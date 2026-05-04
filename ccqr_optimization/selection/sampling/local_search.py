"""
Model-based derivative-free local search for acquisition function minimization.

Implements the ``dfo3__adaptive_narrow`` algorithm: a trust-region DFO walk
using a thin-plate RBF surrogate with BOBYQA-style ρ-ratio trust-region updates
(Powell 2009; Conn, Scheinberg & Vicente 2009), narrow initial trust region for
fine-grained local refinement, and multi-start via diverse epicenter seeding.

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
    Powell, M. J. D. (2009). The BOBYQA algorithm for bound constrained
    optimization without derivatives. DAMTP Report NA2009/06.

    Conn, A. R., Scheinberg, K., & Vicente, L. N. (2009). Introduction to
    Derivative-Free Optimization. SIAM.

    Gower, J. C. (1971). A general coefficient of similarity and some of its
    properties. Biometrics, 27(4), 857–871.

    Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). Reciprocal rank
    fusion outperforms condorcet and individual rank learning methods. SIGIR 2009.
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize as scipy_minimize

from ccqr_optimization.selection.acquisition import BaseConformalSearcher
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


def fit_quadratic(X: np.ndarray, y: np.ndarray, lam: float):
    """
    Ridge-regularized quadratic proxy fitted to (X, y).

    Features: [1, x, x², xᵢxⱼ for i<j]. Returns (model_fn, grad_fn) callables
    on [0, 1]^d. Raises ``np.linalg.LinAlgError`` if the normal equations are singular.
    """
    n, d = X.shape

    def phi(u):
        u = np.atleast_1d(u)
        feats = [1.0, *u, *(u ** 2)]
        for i in range(d):
            for j in range(i + 1, d):
                feats.append(u[i] * u[j])
        return np.array(feats)

    Phi = np.vstack([phi(x) for x in X])
    f = Phi.shape[1]
    w = np.linalg.solve(Phi.T @ Phi + lam * np.eye(f), Phi.T @ y)

    def model(u):
        return float(phi(u) @ w)

    def grad(u):
        u = np.array(u)
        g = np.zeros(d)
        i = 1
        g += w[i:i + d]; i += d
        g += 2.0 * w[i:i + d] * u; i += d
        for a in range(d):
            for b in range(a + 1, d):
                g[a] += w[i] * u[b]
                g[b] += w[i] * u[a]
                i += 1
        return g

    return model, grad


def fit_rbf(X: np.ndarray, y: np.ndarray, lam: float):
    """
    Thin-plate spline RBF surrogate fitted to (X, y) with smoothing ``lam``.

    Falls back to ``fit_quadratic`` if the RBF fit is singular (collinear or
    near-duplicate design points). Returns (model_fn, grad_fn) callables on [0, 1]^d.
    Gradient is computed via central finite differences (step 1e-5).
    """
    n, d = X.shape
    try:
        rbf = RBFInterpolator(X, y, kernel="thin_plate_spline", smoothing=lam * n)
    except Exception:
        return fit_quadratic(X, y, lam)

    def model(u):
        return float(rbf(np.atleast_2d(u))[0])

    def grad(u):
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
    """
    Gower (1971) mixed-type dissimilarity in [0, 1].

    Per-dimension contributions δ_j:
      Categorical        : δ_j = 0 if equal else 1
      Float/Int linear   : δ_j = |v - v'| / (v_max - v_min)
      Float/Int log-scale: δ_j = |log v - log v'| / (log v_max - log v_min)

    Returns the arithmetic mean over non-degenerate dimensions.
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
    """
    Greedy diversity filter over Gower distance.

    Accepts each candidate from a best-first sorted list only if it is at least
    ``min_dist`` from every already-accepted config and from ``already_kept``.
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
    """
    Reciprocal Rank Fusion (Cormack et al., SIGIR 2009).

    RRF(d) = Σ_i  w_i / (k + rank_i(d)), higher score → higher priority.
    """
    scores: Dict[int, float] = {}
    id_map: Dict[int, Config] = {}
    for rank, c in enumerate(historical, start=1):
        cid = id(c)
        scores[cid] = scores[cid] + w_hist / (k + rank) if cid in scores else w_hist / (k + rank)
        id_map[cid] = c
    for rank, c in enumerate(acq_random, start=1):
        cid = id(c)
        scores[cid] = scores[cid] + w_acq / (k + rank) if cid in scores else w_acq / (k + rank)
        id_map[cid] = c
    return [id_map[h] for h in sorted(scores, key=scores.__getitem__, reverse=True)]


def natural_scales(space: Dict[str, ParameterRange]) -> Dict[str, float]:
    """
    Exact uninformative perturbation scale per non-categorical parameter.

    Derived from the parameter's parent distribution std (no estimation):
      Uniform(min, max)           → σ = (max - min) / √12
      LogUniform(min, max) in log → σ = (log max - log min) / √12
    """
    out: Dict[str, float] = {}
    for name, p in space.items():
        if isinstance(p, CategoricalRange):
            continue
        floor = 1e-10 if isinstance(p, FloatRange) else 1
        span = (np.log(p.max_value) - np.log(max(p.min_value, floor))) if p.log_scale else (p.max_value - p.min_value)
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
    """
    Return a copy of ``config`` with exactly one parameter perturbed.

    Categorical : uniform sample from all other choices.
    Float linear: clip(current + N(0, scale·σ), min, max).
    Float log   : same in log space.
    Int linear  : round(N(0, scale·σ)) with |delta| ≥ 1; clip to bounds.
    Int log     : log-space noise, rounded; force ±1 if rounding leaves value unchanged.

    For integers the delta is rounded (not the result) to stay on the integer grid.
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
    """
    Generate ``n`` single-coordinate stochastic perturbations of ``config``.

    One eligible parameter is chosen uniformly at random per perturbation.
    All non-categorical parameters are eligible. Categoricals are eligible
    if they have more than one choice.
    """
    eligible = [
        (name, p) for name, p in space.items()
        if not isinstance(p, CategoricalRange) or len(p.choices) > 1
    ]
    if not eligible:
        return []
    names, pranges = zip(*eligible)
    name_scales = [scales[name] if name in scales else 0.0 for name in names]
    indices = rng.integers(0, len(names), size=n)
    return [
        perturb_one(config, names[i], pranges[i], name_scales[i], scale, rng)
        for i in indices
    ]


class LocalSearchOptimizer:
    """
    Model-based derivative-free local search for acquisition function minimization.

    Implements the ``dfo3__adaptive_narrow`` algorithm: thin-plate RBF surrogate with
    BOBYQA-style ρ-ratio trust-region updates and a narrow initial trust region.

    Algorithm overview:

    **Phase 1 — Candidate scoring.**
    The full random candidate pool is scored in a single batch surrogate call to
    establish the global acquisition baseline.

    **Phase 2 — Epicenter selection.**
    Up to ``n_historical_epicenters`` configs are drawn from the evaluated history
    (best true performance first, ``metric_sign``-adjusted). Up to
    ``n_acq_epicenters`` configs are drawn from the scored candidate pool
    (acquisition-best first). Both sets are diversity-filtered via Gower NMS.
    The two sets are merged into a single priority list via Reciprocal Rank Fusion.

    **Phase 3 — Per-epicenter DFO walk.**
    For each base epicenter (most-to-least promising):

    1. Build an initial interpolation set of ``n_interp`` points by sampling
       random perturbations within the trust region and scoring them.
    2. Fit a thin-plate RBF model to the interpolation set.
    3. Minimize the RBF model within [0, 1]^d using L-BFGS-B (scipy), with
       3 random restarts. For each one-exchange categorical alternative, run an
       independent minimization with those categoricals fixed.
    4. Score novel candidates with the true surrogate; accept the best one.
    5. Update the interpolation set (replace the point with highest acq value).
    6. Compute the ρ-ratio = actual_improvement / predicted_improvement:
       - ρ ≥ η₂: expand trust region × γ_inc (model was accurate).
       - η₁ ≤ ρ < η₂: keep trust region (acceptable step).
       - ρ < η₁: shrink trust region × γ_dec (model was inaccurate).
    7. Accept the step if actual_improvement > ε (independently of ρ).
    8. Terminate when the stall counter exceeds ``tolerance_max``, the trust
       region shrinks below ``delta_min``, or the budget is exhausted.

    Each base epicenter's walk is independent: its own per-walk seen-hash cache
    is seeded only from truly evaluated configs, so sibling walks do not block
    each other's surrogate evaluations.

    Acquisition convention: lower is better (EI returns −EI; LCB and PLB are
    inherently lower-is-better). No sign flipping anywhere in this module.
    """

    def __init__(
        self,
        search_space: Dict[str, ParameterRange],
        config_manager,
        metric_sign: int = 1,
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
            search_space: Parameter name → ParameterRange mapping.
            config_manager: Must expose ``tabularize_configs``, ``searched_configs``,
                ``searched_performances``, and optionally ``searched_config_hashes``
                and ``banned_configurations``.
            metric_sign: +1 for minimization tasks, −1 for maximization. Applied to
                raw stored performances so that best-first order is always
                "lowest signed performance first."
            n_historical_epicenters: Number of base epicenters from evaluated history.
            n_acq_epicenters: Number of base epicenters from the scored random pool.
                NMS scan is O(n_candidates × Y × n_params); warn threshold: Y > 200.
            min_epicenter_dist: Gower distance NMS threshold ζ between any two epicenters.
            w_historical: RRF weight for the historical epicenter list (w_H).
            w_acq: RRF weight for the acquisition-random list (w_R). w_H + w_R = 1.
            max_surrogate_calls: Total surrogate predict-call budget B_max.
            epsilon: Strict improvement threshold; a step is accepted iff
                actual_improvement > ε.
            delta_init: Initial trust-region radius δ₀ in encoded [0, 1]^d space.
            delta_min: Minimum trust-region radius; walk terminates when δ < δ_min.
            delta_max: Maximum trust-region radius.
            gamma_inc: TR expansion factor on a very good step (ρ ≥ η₂).
            gamma_dec: TR contraction factor on a poor step (ρ < η₁).
            n_interp_multiplier: Interpolation set size = max(d+2, multiplier × (d+1)(d+2)/2).
                Higher values give a more accurate model at the cost of more initial evals.
            n_cat_slices: Maximum number of one-exchange categorical alternatives explored
                per round (plus the current categorical assignment).
            lbfgs_maxiter: Maximum L-BFGS-B iterations for model minimization.
            tolerance_max: Maximum consecutive stall rounds before epicenter shutdown.
            model_lam: RBF smoothing regularization parameter λ.
            eta_1: ρ-ratio lower threshold; ρ < η₁ → shrink TR (BOBYQA standard: 0.10).
            eta_2: ρ-ratio upper threshold; ρ ≥ η₂ → expand TR (BOBYQA standard: 0.70).
            random_seed: RNG seed for reproducibility.
        """
        if abs(w_historical + w_acq - 1.0) > 1e-6:
            raise ValueError(f"w_historical + w_acq must equal 1.0, got {w_historical} + {w_acq}")
        if metric_sign not in (1, -1):
            raise ValueError(f"metric_sign must be +1 or -1, got {metric_sign}")
        if n_acq_epicenters > 200:
            logger.warning(
                "n_acq_epicenters=%d is large. NMS scan is O(n_candidates × Y × n_params); "
                "beyond ~200 epicenters this adds noticeable wall-clock time per BO iteration. "
                "Typical good values: 5–20.",
                n_acq_epicenters,
            )

        self.space = search_space
        self.config_manager = config_manager
        self.metric_sign = metric_sign
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
        self.n_cat_slices = n_cat_slices
        self.lbfgs_maxiter = lbfgs_maxiter
        self.tolerance_max = tolerance_max
        self.model_lam = model_lam
        self.eta_1 = eta_1
        self.eta_2 = eta_2
        self.rng = np.random.default_rng(random_seed)

        self.cont_int_names = [n for n, p in search_space.items() if not isinstance(p, CategoricalRange)]
        self.cat_names = [n for n, p in search_space.items() if isinstance(p, CategoricalRange)]
        d = len(self.cont_int_names)
        min_quad = ((d + 1) * (d + 2)) // 2 if d > 0 else 1
        self.n_interp = max(d + 2, int(n_interp_multiplier * min_quad))
        self.scales = natural_scales(search_space)

    def select_next(self, searcher: BaseConformalSearcher, candidates: List[Config]) -> Config:
        """
        Run local search and return the config with the lowest acquisition value found.

        Args:
            searcher: Fitted conformal searcher; ``predict(X)`` returns acquisition values
                where lower is better (true for EI, LCB, and PLB).
            candidates: Random candidate pool to score and seed epicenter selection from.

        Returns:
            Config with the lowest acquisition value found during search.
        """
        if not candidates:
            raise ValueError("candidates must not be empty.")

        calls_used = 0
        acq = searcher.predict(self.config_manager.tabularize_configs(candidates))
        calls_used += len(candidates)

        baseline = float(np.min(acq))
        best_config: Config = candidates[int(np.argmin(acq))]
        best_acq = baseline

        logger.debug("Scored %d candidates. Baseline acq = %.6f", len(candidates), baseline)

        starts = self.select_epicenters(candidates, acq)
        if not starts:
            logger.warning("No base epicenters found; returning best random candidate.")
            return self.clamp(best_config)

        logger.debug("%d base epicenters selected.", len(starts))

        evaluated_hashes: Set[int] = set()
        if hasattr(self.config_manager, "searched_config_hashes"):
            evaluated_hashes.update(self.config_manager.searched_config_hashes)
        if hasattr(self.config_manager, "banned_configurations"):
            evaluated_hashes.update(
                create_config_hash(c) for c in self.config_manager.banned_configurations
            )

        for idx, start in enumerate(starts):
            if calls_used >= self.max_surrogate_calls:
                break
            logger.debug("Epicenter %d/%d. Budget remaining: %d", idx + 1, len(starts), self.max_surrogate_calls - calls_used)
            found, found_acq, calls_used = self.walk(searcher, start, baseline, evaluated_hashes, calls_used)
            if found_acq < best_acq:
                best_acq, best_config = found_acq, found
                logger.debug("New best acq = %.6f at epicenter %d.", best_acq, idx + 1)

        logger.debug("Local search done. Calls: %d/%d. Best acq: %.6f", calls_used, self.max_surrogate_calls, best_acq)
        return self.clamp(best_config)

    def select_epicenters(self, candidates: List[Config], acq: np.ndarray) -> List[Config]:
        """Select, filter, and rank base epicenters via RRF."""
        hist_configs: List[Config] = self.config_manager.searched_configs
        hist_perfs: List[float] = self.config_manager.searched_performances
        hist_starts: List[Config] = []
        if hist_configs:
            signed = [p * self.metric_sign for p in hist_perfs]
            sorted_hist = [hist_configs[i] for i in np.argsort(signed)]
            hist_starts = diversity_filter(sorted_hist, self.space, self.min_epicenter_dist, self.n_historical_epicenters)

        sorted_cands = [candidates[i] for i in np.argsort(acq)]
        acq_starts = diversity_filter(sorted_cands, self.space, self.min_epicenter_dist, self.n_acq_epicenters)

        return rank_fuse(hist_starts, acq_starts, self.w_historical, self.w_acq)

    def categorical_slices(self, current: Config) -> List[Dict]:
        """
        One-exchange categorical alternatives: current assignment plus up to
        ``n_cat_slices`` alternatives per categorical dimension.
        """
        slices = [{n: current[n] for n in self.cat_names}]
        for name in self.cat_names:
            for alt in [c for c in self.space[name].choices if c != current[name]][:self.n_cat_slices]:
                s = {n: current[n] for n in self.cat_names}
                s[name] = alt
                slices.append(s)
                if len(slices) >= self.n_cat_slices + 1:
                    return slices
        return slices

    def minimize_model(self, model_fn, grad_fn) -> np.ndarray:
        """
        Minimize model_fn over [0, 1]^d via L-BFGS-B with 3 random restarts.

        Raises ``RuntimeError`` if scipy minimization fails to produce any result.
        """
        d = len(self.cont_int_names)
        if d == 0:
            return np.array([])

        best_u, best_val = None, float("inf")
        for _ in range(3):
            res = scipy_minimize(
                model_fn, self.rng.random(d), jac=grad_fn, method="L-BFGS-B",
                bounds=[(0.0, 1.0)] * d, options={"maxiter": self.lbfgs_maxiter, "ftol": 1e-9},
            )
            if res.fun < best_val:
                best_val, best_u = res.fun, res.x

        if best_u is None:
            raise RuntimeError("L-BFGS-B minimization produced no result.")
        return np.clip(best_u, 0.0, 1.0)

    def clamp(self, config: Config) -> Config:
        """Clamp all numeric values to their declared bounds (guards against floating-point drift)."""
        out = dict(config)
        for name, p in self.space.items():
            if isinstance(p, FloatRange):
                out[name] = float(min(p.max_value, max(p.min_value, float(out[name]))))
            elif isinstance(p, IntRange):
                out[name] = int(min(p.max_value, max(p.min_value, int(out[name]))))
        return out

    def walk(
        self,
        searcher: BaseConformalSearcher,
        start: Config,
        baseline: float,
        evaluated_hashes: Set[int],
        calls_used: int,
    ) -> Tuple[Config, float, int]:
        """
        ρ-ratio DFO walk from a single base epicenter.

        Creates a fresh per-walk seen-hash cache seeded only from truly evaluated
        configs, so sibling walks do not block each other's surrogate evaluations.

        Trust-region update (BOBYQA-style, Powell 2009):
          ρ = actual_improvement / predicted_improvement
          ρ ≥ η₂ → expand TR; η₁ ≤ ρ < η₂ → keep TR; ρ < η₁ → shrink TR.

        Step acceptance is independent of TR update: any step with
        actual_improvement > ε is accepted.

        Returns:
            Tuple of (best_config_found, best_acq_found, updated_calls_used).
        """
        seen: Set[int] = set(evaluated_hashes)
        current = start
        best_config, best_acq = start, baseline
        current_acq = baseline
        delta = self.delta_init
        stall = 0
        cat_fixed = {n: current[n] for n in self.cat_names}
        d = len(self.cont_int_names)

        init_pts = perturb_batch(current, self.space, self.scales, self.n_interp, delta, self.rng)
        novel_init = [c for c in init_pts if create_config_hash(c) not in seen and not seen.add(create_config_hash(c))]
        novel_init = novel_init[:max(0, self.max_surrogate_calls - calls_used)]
        if not novel_init:
            return start, baseline, calls_used

        init_acq = searcher.predict(self.config_manager.tabularize_configs(novel_init))
        calls_used += len(novel_init)
        interp_set = [(encode(c, self.space, cat_fixed), float(a)) for c, a in zip(novel_init, init_acq)]

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
            u_center = encode(current, self.space, cat_fixed) if d > 0 else np.array([])
            model_at_center = model_fn(u_center) if d > 0 else 0.0

            novel_cands: List[Config] = []
            u_stars: List[np.ndarray] = []
            for cat_slice in self.categorical_slices(current):
                u_star = self.minimize_model(model_fn, grad_fn)
                cand = decode(u_star, self.space, cat_slice)
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

            cand_acq = searcher.predict(self.config_manager.tabularize_configs(novel_cands))
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

            top_enc = encode(top, self.space, cat_fixed)
            if len(interp_set) >= self.n_interp:
                interp_set[int(np.argmax([v for _, v in interp_set]))] = (top_enc, top_acq)
            else:
                interp_set.append((top_enc, top_acq))

            if actual_imp > self.epsilon:
                best_acq, best_config = top_acq, top
                current, current_acq = top, top_acq
                cat_fixed = {n: current[n] for n in self.cat_names}

        logger.debug("Walk done: delta=%.4f, stall=%d/%d, calls=%d, best_acq=%.6f",
                     delta, stall, self.tolerance_max, calls_used, best_acq)
        return best_config, best_acq, calls_used
