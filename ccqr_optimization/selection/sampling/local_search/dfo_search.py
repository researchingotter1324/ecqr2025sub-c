"""Model-based derivative-free local search for acquisition function minimization.

Implements the ``dfo3__adaptive_narrow`` algorithm: a trust-region DFO walk using a
thin-plate RBF surrogate with BOBYQA-style ρ-ratio trust-region updates (Powell 2009;
Conn, Scheinberg & Vicente 2009), a narrow initial trust region for fine-grained local
refinement, and multi-start via diverse epicenter seeding.

Mixed-variable handling:
  - Continuous and integer dimensions: encoded to [0, 1]^d; L-BFGS-B minimizes
    the fitted RBF model within the trust region.
  - Categorical dimensions: enumerated via one-exchange neighbors; one independent
    model minimization is run per categorical slice.

Epicenter selection:
  - X historical base epicenters from the evaluated history (best true performance).
  - Y acquisition-best random base epicenters from the scored candidate pool.
  - Both sets are diversity-filtered via Gower-distance NMS.
  - Merged into a single priority list via Reciprocal Rank Fusion (Cormack et al.,
    SIGIR 2009).

Convention throughout: acquisition values are lower-is-better.

References:
    Powell, M. J. D. (2009). The BOBYQA algorithm for bound constrained optimization
    without derivatives. DAMTP Report NA2009/06.

    Conn, A. R., Scheinberg, K., & Vicente, L. N. (2009). Introduction to
    Derivative-Free Optimization. SIAM.

    Gower, J. C. (1971). A general coefficient of similarity and some of its
    properties. Biometrics, 27(4), 857-871.

    Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). Reciprocal rank
    fusion outperforms condorcet and individual rank learning methods. SIGIR 2009.
"""

import logging
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize as scipy_minimize

from ccqr_optimization.selection.sampling.local_search.base import BaseLocalSearchAlgorithm
from ccqr_optimization.utils.tracking import BaseConfigurationManager
from ccqr_optimization.utils.configurations.utils import create_config_hash
from ccqr_optimization.wrapping import (
    CategoricalRange,
    FloatRange,
    IntRange,
    ParameterRange,
)

logger = logging.getLogger(__name__)

Config = Dict

SQRT12 = np.sqrt(12.0)


def encode(config: Config, space: Dict[str, ParameterRange], cat_fixed: Dict) -> np.ndarray:
    """Encode non-categorical dimensions to [0, 1]^d, fixing categoricals to cat_fixed."""
    vec = []
    for name, p in space.items():
        if isinstance(p, CategoricalRange):
            continue
        v = config[name]
        if isinstance(p, FloatRange):
            if p.log_scale:
                lo, hi = np.log(max(p.min_value, 1e-10)), np.log(p.max_value)
                val = (np.log(max(v, 1e-10)) - lo) / (hi - lo) if hi > lo else 0.5
            else:
                val = (v - p.min_value) / (p.max_value - p.min_value) if p.max_value > p.min_value else 0.5
        else:
            if p.log_scale:
                lo, hi = np.log(max(p.min_value, 1)), np.log(p.max_value)
                val = (np.log(max(v, 1)) - lo) / (hi - lo) if hi > lo else 0.5
            else:
                val = (v - p.min_value) / (p.max_value - p.min_value) if p.max_value > p.min_value else 0.5
        vec.append(float(np.clip(val, 0.0, 1.0)))
    return np.array(vec, dtype=float)


def decode(u: np.ndarray, space: Dict[str, ParameterRange], cat_fixed: Dict) -> Config:
    """Decode a [0, 1]^d vector back to a configuration dict with categoricals from cat_fixed."""
    config = {}
    idx = 0
    for name, p in space.items():
        if isinstance(p, CategoricalRange):
            config[name] = cat_fixed[name]
        elif isinstance(p, FloatRange):
            val = float(np.clip(u[idx], 0.0, 1.0))
            if p.log_scale:
                lo, hi = np.log(max(p.min_value, 1e-10)), np.log(p.max_value)
                raw = float(np.exp(lo + val * (hi - lo)))
            else:
                raw = float(p.min_value + val * (p.max_value - p.min_value))
            config[name] = float(min(p.max_value, max(p.min_value, raw)))
            idx += 1
        else:
            val = float(np.clip(u[idx], 0.0, 1.0))
            if p.log_scale:
                lo, hi = np.log(max(p.min_value, 1)), np.log(p.max_value)
                raw_int = int(round(np.exp(lo + val * (hi - lo))))
            else:
                raw_int = int(round(p.min_value + val * (p.max_value - p.min_value)))
            config[name] = int(min(p.max_value, max(p.min_value, raw_int)))
            idx += 1
    return config


def fit_quadratic(X: np.ndarray, y: np.ndarray, lam: float) -> Tuple[Callable, Callable]:
    """Ridge-regularized quadratic proxy fitted to (X, y).

    Args:
        X: Design matrix of shape (n, d) in [0, 1]^d.
        y: Target values of shape (n,).
        lam: Ridge regularization strength.

    Returns:
        Tuple of (model_fn, grad_fn) callables on [0, 1]^d.

    Raises:
        np.linalg.LinAlgError: If the normal equations are singular.
    """
    n, d = X.shape

    def phi(u: np.ndarray) -> np.ndarray:
        u = np.atleast_1d(u)
        feats = [1.0, *u, *(u ** 2)]
        for i in range(d):
            for j in range(i + 1, d):
                feats.append(u[i] * u[j])
        return np.array(feats)

    Phi = np.vstack([phi(x) for x in X])
    f = Phi.shape[1]
    w = np.linalg.solve(Phi.T @ Phi + lam * np.eye(f), Phi.T @ y)

    def model(u: np.ndarray) -> float:
        return float(phi(u) @ w)

    def grad(u: np.ndarray) -> np.ndarray:
        u = np.array(u)
        g = np.zeros(d)
        i = 1
        g += w[i:i + d]
        i += d
        g += 2.0 * w[i:i + d] * u
        i += d
        for a in range(d):
            for b in range(a + 1, d):
                g[a] += w[i] * u[b]
                g[b] += w[i] * u[a]
                i += 1
        return g

    return model, grad


def fit_rbf(X: np.ndarray, y: np.ndarray, lam: float) -> Tuple[Callable, Callable]:
    """Thin-plate spline RBF surrogate fitted to (X, y) with smoothing ``lam``.

    Falls back to ``fit_quadratic`` if the RBF fit fails on collinear or near-duplicate
    design points. Gradient computed via central finite differences (step 1e-5).

    Args:
        X: Design matrix of shape (n, d) in [0, 1]^d.
        y: Target values of shape (n,).
        lam: Smoothing regularization parameter.

    Returns:
        Tuple of (model_fn, grad_fn) callables on [0, 1]^d.
    """
    try:
        rbf = RBFInterpolator(X, y, kernel="thin_plate_spline", smoothing=lam * len(X))
    except Exception:
        return fit_quadratic(X, y, lam)

    def model(u: np.ndarray) -> float:
        return float(rbf(np.atleast_2d(u))[0])

    def grad(u: np.ndarray) -> np.ndarray:
        u = np.array(u, dtype=float)
        eps = 1e-5
        g = np.zeros_like(u)
        for i in range(len(u)):
            up, um = u.copy(), u.copy()
            up[i] += eps
            um[i] -= eps
            g[i] = (model(up) - model(um)) / (2 * eps)
        return g

    return model, grad


def gower_distance(a: Config, b: Config, space: Dict[str, ParameterRange]) -> float:
    """Gower (1971) mixed-type dissimilarity in [0, 1].

    Per-dimension contributions:
      Categorical:         0 if equal else 1
      Float/Int linear:    |v - v'| / (v_max - v_min)
      Float/Int log-scale: |log v - log v'| / (log v_max - log v_min)

    Args:
        a: First configuration dict.
        b: Second configuration dict.
        space: Search space parameter descriptors.

    Returns:
        Arithmetic mean dissimilarity over non-degenerate dimensions, in [0, 1].
    """
    total, active = 0.0, 0
    for name, p in space.items():
        v1, v2 = a[name], b[name]
        if isinstance(p, CategoricalRange):
            total += 0.0 if v1 == v2 else 1.0
            active += 1
        elif isinstance(p, (FloatRange, IntRange)):
            floor = 1e-10 if isinstance(p, FloatRange) else 1
            if p.log_scale:
                lo, hi = np.log(max(p.min_value, floor)), np.log(p.max_value)
                span = hi - lo
                if span > 0.0:
                    total += abs(np.log(max(v1, floor)) - np.log(max(v2, floor))) / span
                    active += 1
            else:
                span = p.max_value - p.min_value
                if span > 0.0:
                    total += abs(v1 - v2) / span
                    active += 1
    return total / active if active > 0 else 0.0


def diversity_filter(
    candidates: List[Config],
    space: Dict[str, ParameterRange],
    min_dist: float,
    max_keep: int,
    already_kept: Optional[List[Config]] = None,
) -> List[Config]:
    """Greedy Gower-distance diversity filter.

    Accepts each candidate from a best-first sorted list only if it is at least
    ``min_dist`` away from every already-accepted config and from ``already_kept``.

    Args:
        candidates: Pre-sorted list of configurations (best first).
        space: Search space parameter descriptors.
        min_dist: Minimum Gower distance required between any two accepted configs.
        max_keep: Maximum number of configs to keep.
        already_kept: Additional configs to treat as already selected (e.g. from
            another list merged before this call).

    Returns:
        List of accepted configurations, length <= max_keep.
    """
    pool = list(already_kept) if already_kept else []
    kept: List[Config] = []
    for c in candidates:
        if len(kept) >= max_keep:
            break
        if not any(gower_distance(c, s, space) < min_dist for s in pool):
            pool.append(c)
            kept.append(c)
    return kept


def rank_fuse(
    historical: List[Config],
    acq_random: List[Config],
    w_hist: float,
    w_acq: float,
    k: int = 60,
) -> List[Config]:
    """Reciprocal Rank Fusion (Cormack et al., SIGIR 2009).

    RRF(d) = sum_i  w_i / (k + rank_i(d)); higher score = higher priority.

    Args:
        historical: Configs ranked by historical performance (best first).
        acq_random: Configs ranked by acquisition value (best first).
        w_hist: RRF weight for the historical list.
        w_acq: RRF weight for the acquisition list.
        k: Rank smoothing constant.

    Returns:
        Merged list sorted by descending RRF score.
    """
    scores: Dict[int, float] = {}
    id_map: Dict[int, Config] = {}
    for rank, c in enumerate(historical, start=1):
        cid = id(c)
        scores[cid] = (scores[cid] if cid in scores else 0.0) + w_hist / (k + rank)
        id_map[cid] = c
    for rank, c in enumerate(acq_random, start=1):
        cid = id(c)
        scores[cid] = (scores[cid] if cid in scores else 0.0) + w_acq / (k + rank)
        id_map[cid] = c
    return [id_map[h] for h in sorted(scores, key=scores.__getitem__, reverse=True)]


def natural_scales(space: Dict[str, ParameterRange]) -> Dict[str, float]:
    """Exact uninformative perturbation scale per non-categorical parameter.

    Derived from the parameter's parent distribution std:
      Uniform(min, max):           sigma = (max - min) / sqrt(12)
      LogUniform(min, max) in log: sigma = (log max - log min) / sqrt(12)

    Args:
        space: Search space parameter descriptors.

    Returns:
        Mapping from parameter name to natural scale (only non-categorical params).
    """
    out: Dict[str, float] = {}
    for name, p in space.items():
        if isinstance(p, CategoricalRange):
            continue
        floor = 1e-10 if isinstance(p, FloatRange) else 1
        span = (
            (np.log(p.max_value) - np.log(max(p.min_value, floor)))
            if p.log_scale
            else (p.max_value - p.min_value)
        )
        out[name] = span / SQRT12
    return out


def perturb_one(
    config: Config,
    name: str,
    p: ParameterRange,
    natural_scale: float,
    scale: float,
    rng: np.random.Generator,
) -> Config:
    """Return a copy of ``config`` with exactly one parameter perturbed.

    Args:
        config: Source configuration to copy and perturb.
        name: Name of the parameter to perturb.
        p: ParameterRange descriptor for that parameter.
        natural_scale: Pre-computed natural scale for the parameter.
        scale: Trust-region scale factor applied to the noise magnitude.
        rng: NumPy random generator instance.

    Returns:
        New configuration dict differing from ``config`` in exactly one dimension.
    """
    out = config.copy()
    cur = config[name]
    if isinstance(p, CategoricalRange):
        choices = [c for c in p.choices if c != cur]
        if choices:
            out[name] = choices[rng.integers(0, len(choices))]
    elif isinstance(p, FloatRange):
        noise = rng.standard_normal() * scale * natural_scale
        if p.log_scale:
            lo, hi = np.log(max(p.min_value, 1e-10)), np.log(p.max_value)
            raw = float(np.exp(np.clip(np.log(max(cur, 1e-10)) + noise, lo, hi)))
        else:
            raw = float(cur + noise)
        out[name] = float(min(p.max_value, max(p.min_value, raw)))
    elif isinstance(p, IntRange):
        noise = rng.standard_normal() * scale * natural_scale
        if p.log_scale:
            lo, hi = np.log(max(p.min_value, 1)), np.log(p.max_value)
            new = int(round(np.exp(np.clip(np.log(max(cur, 1)) + noise, lo, hi))))
            if new == cur:
                new = cur + (1 if rng.random() > 0.5 else -1)
        else:
            delta = int(round(noise))
            if delta == 0:
                delta = 1 if rng.random() > 0.5 else -1
            new = cur + delta
        out[name] = int(min(p.max_value, max(p.min_value, new)))
    return out


def perturb_batch(
    config: Config,
    space: Dict[str, ParameterRange],
    scales: Dict[str, float],
    n: int,
    scale: float,
    rng: np.random.Generator,
) -> List[Config]:
    """Generate ``n`` single-coordinate stochastic perturbations of ``config``.

    Args:
        config: Source configuration.
        space: Search space parameter descriptors.
        scales: Natural scales per non-categorical parameter (from ``natural_scales``).
        n: Number of perturbed configurations to generate.
        scale: Trust-region scale factor applied to all noise magnitudes.
        rng: NumPy random generator instance.

    Returns:
        List of ``n`` configurations, each differing from ``config`` in one dimension.
        Empty list if no eligible parameters exist.
    """
    eligible = [
        (name, p) for name, p in space.items()
        if not isinstance(p, CategoricalRange) or len(p.choices) > 1
    ]
    if not eligible:
        return []
    names, pranges = zip(*eligible)
    name_scales = [scales[name] for name in names]
    indices = rng.integers(0, len(names), size=n)
    return [
        perturb_one(config, names[i], pranges[i], name_scales[i], scale, rng)
        for i in indices
    ]


class DFOLocalSearch(BaseLocalSearchAlgorithm):
    """Model-based derivative-free local search for acquisition function minimization.

    Implements the ``dfo3__adaptive_narrow`` algorithm: thin-plate RBF surrogate with
    BOBYQA-style rho-ratio trust-region updates and a narrow initial trust region.

    Phase 1 — Candidate scoring: the full random candidate pool is scored in one batch
    call to establish the acquisition baseline.

    Phase 2 — Epicenter selection: up to ``n_historical_epicenters`` configs from the
    evaluated history (best true performance first) and up to ``n_acq_epicenters`` from
    the scored pool (best acquisition first) are diversity-filtered via Gower NMS and
    merged via Reciprocal Rank Fusion.

    Phase 3 — Per-epicenter DFO walk:
    1. Sample ``n_interp`` perturbations; score with the true surrogate.
    2. Fit a thin-plate RBF model to the interpolation set.
    3. Minimize the RBF with L-BFGS-B (3 random restarts). Repeat per categorical slice.
    4. Score novel candidates; accept the best one.
    5. Update interpolation set; compute rho-ratio and update trust region.
    6. Terminate on stall, trust-region collapse, or budget exhaustion.
    """

    def __init__(
        self,
        n_historical_epicenters: int = 2,
        n_acq_epicenters: int = 10,
        min_epicenter_dist: float = 0.05,
        w_historical: float = 0.7,
        w_acq: float = 0.3,
        max_surrogate_calls: int = 10_000,
        epsilon: float = 1e-6,
        delta_init: float = 0.08,
        delta_min: float = 0.001,
        delta_max: float = 0.5,
        gamma_inc: float = 1.5,
        gamma_dec: float = 0.6,
        n_interp_multiplier: float = 3.0,
        n_cat_slices: int = 3,
        lbfgs_maxiter: int = 80,
        tolerance_max: int = 15,
        model_lam: float = 1e-2,
        eta_1: float = 0.10,
        eta_2: float = 0.70,
        random_seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            n_historical_epicenters: Number of base epicenters from evaluated history.
            n_acq_epicenters: Number of base epicenters from the scored random pool.
            min_epicenter_dist: Gower distance NMS threshold between any two epicenters.
            w_historical: RRF weight for the historical epicenter list.
            w_acq: RRF weight for the acquisition-random list. Must satisfy w_historical + w_acq = 1.
            max_surrogate_calls: Total surrogate predict-call budget.
            epsilon: Strict improvement threshold for step acceptance.
            delta_init: Initial trust-region radius in encoded [0, 1]^d space.
            delta_min: Minimum trust-region radius; walk terminates when delta < delta_min.
            delta_max: Maximum trust-region radius.
            gamma_inc: TR expansion factor on a very good step (rho >= eta_2).
            gamma_dec: TR contraction factor on a poor step (rho < eta_1).
            n_interp_multiplier: Interpolation set size = max(d+2, multiplier * (d+1)(d+2)/2).
            n_cat_slices: Maximum number of one-exchange categorical alternatives per round.
            lbfgs_maxiter: Maximum L-BFGS-B iterations for model minimization.
            tolerance_max: Maximum consecutive stall rounds before epicenter shutdown.
            model_lam: RBF smoothing regularization parameter.
            eta_1: rho-ratio lower threshold; rho < eta_1 shrinks TR.
            eta_2: rho-ratio upper threshold; rho >= eta_2 expands TR.
            random_seed: RNG seed for reproducibility.

        Raises:
            ValueError: If w_historical + w_acq != 1.0.
        """
        if abs(w_historical + w_acq - 1.0) > 1e-6:
            raise ValueError(f"w_historical + w_acq must equal 1.0, got {w_historical} + {w_acq}")
        if n_acq_epicenters > 200:
            logger.warning(
                "n_acq_epicenters=%d is large; NMS scan is O(n_candidates x Y x n_params).",
                n_acq_epicenters,
            )

        self.n_historical_epicenters = n_historical_epicenters
        self.n_acq_epicenters = n_acq_epicenters
        self.min_epicenter_dist = min_epicenter_dist
        self.w_historical = w_historical
        self.w_acq = w_acq
        self.max_surrogate_calls = max_surrogate_calls
        self.epsilon = epsilon
        self.delta_init = delta_init
        self.delta_min = delta_min
        self.delta_max = delta_max
        self.gamma_inc = gamma_inc
        self.gamma_dec = gamma_dec
        self.n_interp_multiplier = n_interp_multiplier
        self.n_cat_slices = n_cat_slices
        self.lbfgs_maxiter = lbfgs_maxiter
        self.tolerance_max = tolerance_max
        self.model_lam = model_lam
        self.eta_1 = eta_1
        self.eta_2 = eta_2
        self.random_seed = random_seed

    def optimize(
        self,
        searcher,
        candidates: List[Config],
        config_manager: BaseConfigurationManager,
        search_space: Dict[str, ParameterRange],
        metric_sign: int,
    ) -> Config:
        """Run DFO trust-region local search and return the best configuration found.

        Args:
            searcher: ``QuantileConformalSearcher`` instance; ``predict(X)``
                returns acquisition values (lower-is-better).
            candidates: Random candidate pool. Must be non-empty.
            config_manager: Exposes ``tabularize_configs``, ``searched_configs``,
                ``searched_performances``, and optionally ``searched_config_hashes``
                and ``banned_configurations``.
            search_space: Parameter name to ParameterRange mapping.
            metric_sign: ``+1`` for minimization, ``-1`` for maximization.

        Returns:
            Config with the lowest acquisition value found.

        Raises:
            ValueError: If ``candidates`` is empty.
        """
        if not candidates:
            raise ValueError("candidates must not be empty.")

        rng = np.random.default_rng(self.random_seed)

        cont_int_names = [n for n, p in search_space.items() if not isinstance(p, CategoricalRange)]
        cat_names = [n for n, p in search_space.items() if isinstance(p, CategoricalRange)]
        d = len(cont_int_names)
        min_quad = ((d + 1) * (d + 2)) // 2 if d > 0 else 1
        n_interp = max(d + 2, int(self.n_interp_multiplier * min_quad))
        scales = natural_scales(search_space)

        def predict(cfgs: List[Config]) -> np.ndarray:
            return searcher.predict(config_manager.tabularize_configs(cfgs))

        calls_used = 0
        acq = predict(candidates)
        calls_used += len(candidates)

        baseline = float(np.min(acq))
        best_config: Config = candidates[int(np.argmin(acq))]
        best_acq = baseline

        logger.debug("DFO: Scored %d candidates. Baseline acq = %.6f", len(candidates), baseline)

        starts = self.select_epicenters(candidates, acq, config_manager, search_space, metric_sign)
        if not starts:
            logger.warning("DFO: No epicenters found; returning best random candidate.")
            return self.clamp(best_config, search_space)

        logger.debug("DFO: %d epicenters selected.", len(starts))

        evaluated_hashes: Set[int] = set()
        if hasattr(config_manager, "searched_config_hashes"):
            evaluated_hashes.update(config_manager.searched_config_hashes)
        if hasattr(config_manager, "banned_configurations"):
            evaluated_hashes.update(
                create_config_hash(c) for c in config_manager.banned_configurations
            )

        for idx, start in enumerate(starts):
            if calls_used >= self.max_surrogate_calls:
                break
            logger.debug(
                "DFO: Epicenter %d/%d. Budget remaining: %d",
                idx + 1, len(starts), self.max_surrogate_calls - calls_used,
            )
            found, found_acq, calls_used = self.walk(
                predict_fn=predict,
                start=start,
                baseline=baseline,
                evaluated_hashes=evaluated_hashes,
                calls_used=calls_used,
                search_space=search_space,
                cont_int_names=cont_int_names,
                cat_names=cat_names,
                d=d,
                n_interp=n_interp,
                scales=scales,
                rng=rng,
            )
            if found_acq < best_acq:
                best_acq, best_config = found_acq, found
                logger.debug("DFO: New best acq = %.6f at epicenter %d.", best_acq, idx + 1)

        logger.debug(
            "DFO: Done. Calls: %d/%d. Best acq: %.6f",
            calls_used, self.max_surrogate_calls, best_acq,
        )
        return self.clamp(best_config, search_space)

    def select_epicenters(
        self,
        candidates: List[Config],
        acq: np.ndarray,
        config_manager: BaseConfigurationManager,
        search_space: Dict[str, ParameterRange],
        metric_sign: int,
    ) -> List[Config]:
        """Select, diversity-filter, and rank base epicenters via RRF.

        Args:
            candidates: Candidate pool (same order as ``acq``).
            acq: Acquisition values for each candidate (lower-is-better).
            config_manager: Provides ``searched_configs`` and ``searched_performances``.
            search_space: Search space parameter descriptors.
            metric_sign: ``+1`` for minimization, ``-1`` for maximization.

        Returns:
            Merged list of epicenters sorted by descending RRF score.
        """
        hist_configs: List[Config] = config_manager.searched_configs
        hist_perfs: List[float] = config_manager.searched_performances
        hist_starts: List[Config] = []
        if hist_configs:
            signed = [p * metric_sign for p in hist_perfs]
            sorted_hist = [hist_configs[i] for i in np.argsort(signed)]
            hist_starts = diversity_filter(
                sorted_hist, search_space, self.min_epicenter_dist, self.n_historical_epicenters
            )

        sorted_cands = [candidates[i] for i in np.argsort(acq)]
        acq_starts = diversity_filter(
            sorted_cands, search_space, self.min_epicenter_dist, self.n_acq_epicenters
        )

        return rank_fuse(hist_starts, acq_starts, self.w_historical, self.w_acq)

    def categorical_slices(
        self,
        current: Config,
        cat_names: List[str],
        search_space: Dict[str, ParameterRange],
    ) -> List[Dict]:
        """One-exchange categorical alternatives plus the current categorical assignment.

        Args:
            current: Current configuration.
            cat_names: Names of all categorical parameters.
            search_space: Search space parameter descriptors.

        Returns:
            List of categorical assignment dicts; the first entry is the current assignment.
        """
        slices = [{n: current[n] for n in cat_names}]
        for name in cat_names:
            for alt in [c for c in search_space[name].choices if c != current[name]][: self.n_cat_slices]:
                s = {n: current[n] for n in cat_names}
                s[name] = alt
                slices.append(s)
                if len(slices) >= self.n_cat_slices + 1:
                    return slices
        return slices

    def minimize_model(
        self,
        model_fn: Callable,
        grad_fn: Callable,
        d: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Minimize ``model_fn`` over [0, 1]^d via L-BFGS-B with 3 random restarts.

        Args:
            model_fn: Scalar objective callable on [0, 1]^d.
            grad_fn: Gradient callable on [0, 1]^d.
            d: Dimensionality of the encoded continuous/integer subspace.
            rng: NumPy random generator for restart initialisation.

        Returns:
            Best minimizer found in [0, 1]^d, or the domain centre if all
            restarts produce NaN (degenerate flat model).
        """
        if d == 0:
            return np.array([])

        best_u: Optional[np.ndarray] = None
        best_val = float("inf")
        for _ in range(3):
            res = scipy_minimize(
                model_fn, rng.random(d), jac=grad_fn, method="L-BFGS-B",
                bounds=[(0.0, 1.0)] * d,
                options={"maxiter": self.lbfgs_maxiter, "ftol": 1e-9},
            )
            if not np.isnan(res.fun) and res.fun < best_val:
                best_val, best_u = res.fun, res.x

        if best_u is None:
            return np.full(d, 0.5)
        return np.clip(best_u, 0.0, 1.0)

    @staticmethod
    def clamp(config: Config, search_space: Dict[str, ParameterRange]) -> Config:
        """Clamp all numeric values to their declared bounds.

        Args:
            config: Configuration potentially containing out-of-bound values.
            search_space: Search space parameter descriptors.

        Returns:
            New configuration dict with all values clipped to declared bounds.
        """
        out = dict(config)
        for name, p in search_space.items():
            if isinstance(p, FloatRange):
                out[name] = float(min(p.max_value, max(p.min_value, float(out[name]))))
            elif isinstance(p, IntRange):
                out[name] = int(min(p.max_value, max(p.min_value, int(out[name]))))
        return out

    def walk(
        self,
        predict_fn: Callable[[List[Config]], np.ndarray],
        start: Config,
        baseline: float,
        evaluated_hashes: Set[int],
        calls_used: int,
        search_space: Dict[str, ParameterRange],
        cont_int_names: List[str],
        cat_names: List[str],
        d: int,
        n_interp: int,
        scales: Dict[str, float],
        rng: np.random.Generator,
    ) -> Tuple[Config, float, int]:
        """rho-ratio DFO walk from a single base epicenter.

        Trust-region update (BOBYQA-style):
          rho = actual_improvement / predicted_improvement
          rho >= eta_2: expand TR; eta_1 <= rho < eta_2: keep TR; rho < eta_1: shrink TR.

        Step acceptance is independent of TR update: any step with
        actual_improvement > epsilon is accepted.

        Args:
            predict_fn: Callable that scores a list of configs and returns an ndarray.
            start: Starting configuration for this walk.
            baseline: Acquisition value at ``start`` (lower-is-better).
            evaluated_hashes: Set of configuration hashes already evaluated (avoid repeats).
            calls_used: Number of surrogate calls consumed so far.
            search_space: Search space parameter descriptors.
            cont_int_names: Names of continuous and integer parameters.
            cat_names: Names of categorical parameters.
            d: Number of continuous/integer dimensions.
            n_interp: Interpolation set size.
            scales: Natural scales per non-categorical parameter.
            rng: NumPy random generator.

        Returns:
            Tuple of (best_config_found, best_acq_found, updated_calls_used).
        """
        seen: Set[int] = set(evaluated_hashes)
        current = start
        best_config, best_acq = start, baseline
        current_acq = baseline
        delta = self.delta_init
        stall = 0
        cat_fixed = {n: current[n] for n in cat_names}

        init_pts = perturb_batch(current, search_space, scales, n_interp, delta, rng)
        novel_init = [
            c for c in init_pts
            if create_config_hash(c) not in seen and not seen.add(create_config_hash(c))  # type: ignore[func-returns-value]
        ]
        novel_init = novel_init[:max(0, self.max_surrogate_calls - calls_used)]
        if not novel_init:
            return start, baseline, calls_used

        init_acq = predict_fn(novel_init)
        calls_used += len(novel_init)
        interp_set = [
            (encode(c, search_space, cat_fixed), float(a))
            for c, a in zip(novel_init, init_acq)
        ]

        while stall <= self.tolerance_max and delta >= self.delta_min and calls_used < self.max_surrogate_calls:
            if not interp_set:
                break
            X_enc = np.vstack([pt for pt, _ in interp_set])
            y_vals = np.array([v for _, v in interp_set])

            if len(X_enc) < 2 or X_enc.shape[1] == 0:
                delta = max(self.delta_min, delta * self.gamma_dec)
                stall += 1
                continue

            model_fn, grad_fn = fit_rbf(X_enc, y_vals, self.model_lam)
            u_center = encode(current, search_space, cat_fixed) if d > 0 else np.array([])
            model_at_center = model_fn(u_center) if d > 0 else 0.0

            novel_cands: List[Config] = []
            u_stars: List[np.ndarray] = []
            for cat_slice in self.categorical_slices(current, cat_names, search_space):
                u_star = self.minimize_model(model_fn, grad_fn, d, rng)
                cand = decode(u_star, search_space, cat_slice)
                h = create_config_hash(cand)
                if h not in seen:
                    seen.add(h)
                    novel_cands.append(cand)
                    u_stars.append(u_star)

            if not novel_cands:
                delta = max(self.delta_min, delta * self.gamma_dec)
                stall += 1
                continue

            budget_left = self.max_surrogate_calls - calls_used
            novel_cands, u_stars = novel_cands[:budget_left], u_stars[:budget_left]
            if not novel_cands:
                break

            cand_acq = predict_fn(novel_cands)
            calls_used += len(novel_cands)

            top_idx = int(np.argmin(cand_acq))
            top, top_acq = novel_cands[top_idx], float(cand_acq[top_idx])
            top_u_star = u_stars[top_idx]

            model_at_star = model_fn(top_u_star) if d > 0 else model_at_center
            predicted_imp = model_at_center - model_at_star
            actual_imp = current_acq - top_acq
            rho = actual_imp / predicted_imp if abs(predicted_imp) > 1e-12 else 0.0

            if rho >= self.eta_2:
                delta = min(delta * self.gamma_inc, self.delta_max)
                stall = 0
            elif rho >= self.eta_1:
                stall = 0
            else:
                delta = max(self.delta_min, delta * self.gamma_dec)
                stall += 1

            top_enc = encode(top, search_space, cat_fixed)
            if len(interp_set) >= n_interp:
                interp_set[int(np.argmax([v for _, v in interp_set]))] = (top_enc, top_acq)
            else:
                interp_set.append((top_enc, top_acq))

            if actual_imp > self.epsilon:
                best_acq, best_config = top_acq, top
                current, current_acq = top, top_acq
                cat_fixed = {n: current[n] for n in cat_names}

        logger.debug(
            "DFO walk done: delta=%.4f, stall=%d/%d, calls=%d, best_acq=%.6f",
            delta, stall, self.tolerance_max, calls_used, best_acq,
        )
        return best_config, best_acq, calls_used
