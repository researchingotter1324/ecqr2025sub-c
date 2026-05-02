import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ccqr_optimization.selection.acquisition import BaseConformalSearcher
from ccqr_optimization.utils.configurations.utils import create_config_hash
from ccqr_optimization.wrapping import (
    CategoricalRange,
    FloatRange,
    IntRange,
    ParameterRange,
)

logger = logging.getLogger(__name__)

Config = Dict  # hyperparameter configuration dict: {param_name: value}

_SQRT12 = np.sqrt(12.0)  # std of Uniform(0,1); used in natural-scale computation


def gower_distance(
    a: Config,
    b: Config,
    space: Dict[str, ParameterRange],
) -> float:
    """
    Gower (1971) mixed-type dissimilarity between two configurations, returning d ∈ [0, 1].

    Per-dimension contributions δ_j:
      Categorical        : δ_j = 0 if equal else 1
      Float/Int linear   : δ_j = |v - v'| / (v_max - v_min)
      Float/Int log-scale: δ_j = |log v - log v'| / (log v_max - log v_min)

    The final distance is the arithmetic mean over all non-degenerate (non-zero-range)
    dimensions. Degenerate dimensions contribute nothing to numerator or denominator.

    References:
        Gower, J. C. (1971). A general coefficient of similarity and some of its
        properties. Biometrics, 27(4), 857–871.
        Hallerberg et al. (2023). Mixed-variable Bayesian optimization. arXiv:2206.01409.
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
    Greedy diversity filter (non-maximum suppression over Gower distance).

    Iterates over `candidates` in priority order (best first) and accepts each one only
    if it is at least `min_dist` Gower distance from every already-accepted config and
    from every config in `already_kept`. Stops when `max_keep` configs are accepted.

    Args:
        candidates: Configs pre-sorted best → worst by priority.
        space: Parameter range definitions.
        min_dist: Minimum Gower distance threshold (ζ).
        max_keep: Maximum number of configs to return.
        already_kept: Configs committed by a prior selection step; candidates too close
                      to these are also rejected.

    Returns:
        Up to max_keep diverse configs in priority order.
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


def natural_scales(space: Dict[str, ParameterRange]) -> Dict[str, float]:
    """
    Compute the exact uninformative perturbation scale for each non-categorical parameter,
    derived analytically from the parameter's known parent distribution.

    Since the search space fully specifies each parameter's distribution (Uniform or
    LogUniform), no sample estimation is needed. The natural scale is the standard
    deviation of the parent distribution in its perturbation space:

      Float/Int linear   : Uniform(min, max) → σ = (max - min) / √12
      Float/Int log-scale: LogUniform(min, max) sampled as Uniform in log space →
                           σ = (log(max) - log(min)) / √12

    This gives σ in the appropriate space (raw units for linear; log-units for log-scale).
    The perturbation rule then draws noise from N(0, scale · σ) where `scale` is the
    adaptive multiplier that starts small and grows with failures.

    Categoricals are excluded; they are perturbed via uniform sampling over alternatives.

    Returns:
        {param_name: σ} for all non-categorical parameters.
    """
    scales: Dict[str, float] = {}
    for name, p in space.items():
        if isinstance(p, CategoricalRange):
            continue
        floor = 1e-10 if isinstance(p, FloatRange) else 1
        if p.log_scale:
            span = np.log(p.max_value) - np.log(max(p.min_value, floor))
        else:
            span = p.max_value - p.min_value
        scales[name] = span / _SQRT12
    return scales


def _perturb_one(
    config: Config,
    name: str,
    p: ParameterRange,
    natural_scale: float,
    scale: float,
    rng: np.random.Generator,
) -> Config:
    """
    Return a copy of `config` with exactly one parameter perturbed.

    The noise magnitude is N(0, scale · natural_scale) where natural_scale is the
    parameter's distribution std (see `natural_scales`).

    Categorical: sample uniformly from all other choices.
    Float linear: new = clip(current + N(0, scale·σ), min, max)
    Float log   : same, but noise added in log space.
    Int linear  : delta = round(N(0, scale·σ)); |delta| ≥ 1 enforced;
                  new = clip(current + delta, min, max).
                  The delta is rounded (not the result) to preserve the integer grid.
    Int log     : noise added in log space; result rounded to nearest integer;
                  if rounding yields no change, displace by ±1.
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
            out[name] = float(np.exp(np.clip(np.log(max(cur, 1e-10)) + noise, lo, hi)))
        else:
            out[name] = float(np.clip(cur + noise, p.min_value, p.max_value))

    elif isinstance(p, IntRange):
        noise = rng.standard_normal() * scale * natural_scale
        if p.log_scale:
            lo, hi = np.log(max(p.min_value, 1)), np.log(p.max_value)
            new = int(round(np.exp(np.clip(np.log(max(cur, 1)) + noise, lo, hi))))
            if new == cur:  # rounding left us on the same integer — force displacement
                new = int(np.clip(cur + (1 if rng.random() > 0.5 else -1), p.min_value, p.max_value))
        else:
            delta = int(round(noise))
            if delta == 0:
                delta = 1 if rng.random() > 0.5 else -1
            new = int(np.clip(cur + delta, p.min_value, p.max_value))
        out[name] = new

    return out


def perturb(
    config: Config,
    space: Dict[str, ParameterRange],
    scales: Dict[str, float],
    n: int,
    scale: float,
    rng: np.random.Generator,
) -> List[Config]:
    """
    Generate `n` single-coordinate stochastic perturbations of `config`.

    All non-categorical parameters are eligible (their natural scale is always defined
    since the search space enforces max > min). Categoricals are eligible if they have
    more than one choice. One eligible parameter is chosen uniformly at random per
    perturbation and the type-appropriate noise rule is applied (see `_perturb_one`).

    Returns fewer than `n` configs only if no eligible parameters exist.
    """
    eligible = [
        (name, p) for name, p in space.items()
        if not isinstance(p, CategoricalRange) or len(p.choices) > 1
    ]
    if not eligible:
        return []

    names, pranges = zip(*eligible)
    indices = rng.integers(0, len(names), size=n)
    return [
        _perturb_one(config, names[i], pranges[i], scales.get(names[i], 0.0), scale, rng)
        for i in indices
    ]


def rank_fuse(
    historical: List[Config],
    acq_random: List[Config],
    w_historical: float,
    w_acq: float,
    k: int = 60,
) -> List[Config]:
    """
    Merge two ranked lists of base epicenters into a single priority-ordered list via
    Reciprocal Rank Fusion (RRF).

    Each config d receives a score:
        RRF(d) = Σ_i  w_i / (k + rank_i(d))
    where rank_i is 1-indexed within list i and k=60 is the standard smoothing constant
    (Cormack et al., SIGIR 2009) that dampens outsized influence of the very top ranks.
    Higher RRF score → higher priority. Configs appearing in only one list are scored by
    their single-list contribution alone.

    Args:
        historical: Historical base epicenters, sorted best → worst by true performance.
        acq_random: Acquisition-best random base epicenters, sorted best → worst by acq score.
        w_historical: Weight for the historical list (w_H).
        w_acq: Weight for the acquisition-random list (w_R).
        k: Smoothing constant (default 60, per original paper).

    Returns:
        Merged list sorted by descending RRF score.

    Reference:
        Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009).
        Reciprocal rank fusion outperforms condorcet and individual rank learning methods.
        SIGIR 2009, pp. 758–759.
    """
    scores: Dict[int, float] = {}
    id_to_config: Dict[int, Config] = {}

    for rank, c in enumerate(historical, start=1):
        scores[id(c)] = scores.get(id(c), 0.0) + w_historical / (k + rank)
        id_to_config[id(c)] = c
    for rank, c in enumerate(acq_random, start=1):
        scores[id(c)] = scores.get(id(c), 0.0) + w_acq / (k + rank)
        id_to_config[id(c)] = c

    return [id_to_config[h] for h in sorted(scores, key=scores.__getitem__, reverse=True)]


class LocalSearchOptimizer:
    """
    Stochastic perturbation-based local search for acquisition function minimization.

    All acquisition values follow the codebase convention: lower is better. This is true
    for all supported acquisition functions (EI returns −EI; LCB and PLB are inherently
    lower-is-better). No sign flipping is applied anywhere in this module.

    Terminology:
        base epicenter      — one of the X+Y seed configs selected before search begins.
                              Each gets an independent walk with a fresh per-epicenter
                              surrogate-evaluation cache and a frozen acq baseline.
        iterative epicenter — the single current config from which perturbations are
                              generated within a base epicenter's walk. It advances on
                              strict improvements or lateral moves.

    Algorithm (four phases):
    1. Score the full random candidate pool in one batch call to establish the global
       acquisition baseline (minimum acq value = most promising config seen so far).
    2. Select X historical base epicenters from the evaluated history by true performance
       (metric_sign-adjusted so best-first ordering is correct for both minimize/maximize
       tasks), diversity-filtered via Gower-distance NMS.
    3. Select Y acquisition-best random base epicenters from the scored pool, diversity-
       filtered independently of the historical set. Merge all X+Y base epicenters into a
       single priority list via Reciprocal Rank Fusion (Cormack et al.,
       SIGIR 2009) with configurable weights w_H / w_R.
    4. Run a sequential single-coordinate perturbation walk from each base epicenter:
         - Each round generates Z perturbations of the current iterative epicenter in one
           batch predict call.
         - Perturbations that hash-collide with the walk path (cycle prevention) or with
           configs already surrogate-evaluated in this epicenter's walk are discarded.
         - Strict improvement (best acq < threshold): advance, reset scale; tau continues.
         - Lateral move     (best acq == threshold): advance, increment tau (no scale reset).
         - Failure          (best acq > threshold):  stay, increment tau, grow scale by γ.
         - tau is never reset within a walk; epicenter shuts down when tau > tolerance_max.
       Returns the config with the lowest acquisition value found across all walks.

    Perturbation prior:
        Each numeric parameter is perturbed with noise drawn from N(0, scale · σ), where
        σ is the parameter's natural scale — the exact standard deviation of its parent
        distribution derived from the search space definition:
          Uniform(min, max) → σ = (max - min) / √12
          LogUniform(min, max) sampled in log space → σ = (log max - log min) / √12
        No sample estimation is required. `scale` starts at `initial_perturbation_scale`
        (default 0.1 of σ, for fine-grained local steps) and grows by `scale_growth_factor`
        (default 1.5) per non-improving round, resetting to the initial value on strict
        improvements.
    """

    def __init__(
        self,
        search_space: Dict[str, ParameterRange],
        config_manager,
        metric_sign: int = 1,
        n_historical_epicenters: int = 2,
        n_acq_epicenters: int = 200,
        min_epicenter_dist: float = 0.05,
        w_historical: float = 0.7,
        w_acq: float = 0.3,
        n_perturbations: int = 500,
        initial_perturbation_scale: float = 0.1,
        scale_growth_factor: float = 2,
        max_perturbation_scale: float = 10.0,
        tolerance_max: int = 25,
        max_surrogate_calls: int = 10000,
        random_seed: Optional[int] = None,
    ):
        """
        Args:
            search_space: Parameter name → ParameterRange mapping.
            config_manager: Provides tabularize_configs, searched_configs,
                searched_performances, and optionally searched_config_hashes
                and banned_configurations.
            metric_sign: +1 for minimization tasks, −1 for maximization tasks.
                Raw performances in config_manager are unsigned; metric_sign is applied
                before sorting so the best config is always the one with the lowest
                signed performance.
            n_historical_epicenters: X — historical base epicenters to select.
            n_acq_epicenters: Y — acquisition-best random base epicenters to select.
                The NMS scan is O(n_candidates × Y × n_params); keep Y ≤ ~100 for
                fast iterations. A warning is logged for Y > 200.
            min_epicenter_dist: ζ — minimum Gower distance between any two base epicenters
                (NMS diversity threshold).
            w_historical: w_H — RRF weight for the historical list (0 < w_H < 1).
            w_acq: w_R — RRF weight for the acquisition-random list (0 < w_R < 1).
                w_H + w_R must equal 1.0.
            n_perturbations: Z — perturbations generated per round (one batch call).
            initial_perturbation_scale: σ₀ — starting multiplier on each parameter's
                natural scale. Round-1 noise is N(0, σ₀ · σ_natural). Default 0.1
                (10% of the distribution std) for fine-grained initial steps.
            scale_growth_factor: γ — multiplicative scale growth per non-improving round.
                Default 1.5 (50% growth per failure). Higher than typical 1.1 to compensate
                for the small starting scale and escape ruts faster.
            max_perturbation_scale: σ_max — ceiling on the adaptive scale multiplier.
            tolerance_max: τ_max — max consecutive non-strict-improvement rounds before
                a base epicenter shuts down.
            max_surrogate_calls: B_max — total surrogate predict calls budget.
            random_seed: RNG seed for reproducibility.
        """
        if abs(w_historical + w_acq - 1.0) > 1e-6:
            raise ValueError(
                f"w_historical + w_acq must equal 1.0, got {w_historical} + {w_acq}"
            )
        if metric_sign not in (1, -1):
            raise ValueError(f"metric_sign must be +1 or -1, got {metric_sign}")
        if n_acq_epicenters > 200:
            logger.warning(
                "n_acq_epicenters=%d is large. The NMS diversity filter runs in "
                "O(n_candidates × n_acq_epicenters × n_params); beyond ~200 epicenters "
                "this becomes a noticeable fraction of wall-clock time per BO iteration. "
                "Typical good values: 10–50.",
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
        self.n_perturbations = n_perturbations
        self.initial_perturbation_scale = initial_perturbation_scale
        self.scale_growth_factor = scale_growth_factor
        self.max_perturbation_scale = max_perturbation_scale
        self.tolerance_max = tolerance_max
        self.max_surrogate_calls = max_surrogate_calls
        self._rng = np.random.default_rng(random_seed)
        self._natural_scales = natural_scales(search_space)

    def select_next(
        self,
        searcher: BaseConformalSearcher,
        candidates: List[Config],
    ) -> Config:
        """
        Run local search and return the config with the lowest acquisition value found.

        Args:
            searcher: Fitted conformal searcher; predict(X) returns acquisition values
                      where lower is better (EI, LCB, PLB all follow this convention).
            candidates: Random candidate pool to score and seed epicenter selection from.

        Returns:
            Config with the lowest acquisition value found during search.
        """
        if not candidates:
            raise ValueError("candidates must not be empty.")

        calls_used = 0

        X = self.config_manager.tabularize_configs(candidates)
        acq = searcher.predict(X)
        calls_used += len(candidates)

        baseline = float(np.min(acq))
        best_config: Config = candidates[int(np.argmin(acq))]
        best_acq: float = baseline

        logger.debug("Scored %d candidates. Baseline acq = %.6f", len(candidates), baseline)

        hist_starts = self._historical_starts()
        acq_starts = self._acq_starts(candidates, acq)
        starts = rank_fuse(hist_starts, acq_starts, self.w_historical, self.w_acq)

        if not starts:
            logger.warning("No base epicenters found; returning best random candidate.")
            return best_config

        logger.debug(
            "%d historical + %d acq-random = %d base epicenters (RRF merged).",
            len(hist_starts), len(acq_starts), len(starts),
        )

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
            logger.debug(
                "Base epicenter %d/%d. Budget remaining: %d",
                idx + 1, len(starts), self.max_surrogate_calls - calls_used,
            )
            found, found_acq, calls_used = self._search_from(
                searcher, start, baseline, evaluated_hashes, calls_used
            )
            if found_acq < best_acq:
                best_acq, best_config = found_acq, found
                logger.debug("Global best acq = %.6f at base epicenter %d.", best_acq, idx + 1)

        logger.debug(
            "Local search done. Calls used: %d/%d. Best acq: %.6f",
            calls_used, self.max_surrogate_calls, best_acq,
        )
        return best_config

    def _historical_starts(self) -> List[Config]:
        """
        Select up to n_historical_epicenters configs from the evaluated history, sorted
        best-first by metric_sign-adjusted performance, diversity-filtered via NMS.

        Raw performances are unsigned; multiplying by metric_sign makes the convention
        "lower signed performance = better" for both minimization (+1) and maximization (−1).
        """
        configs: List[Config] = self.config_manager.searched_configs
        perfs: List[float] = self.config_manager.searched_performances
        if not configs:
            return []
        signed = [p * self.metric_sign for p in perfs]
        sorted_configs = [configs[i] for i in np.argsort(signed)]
        return diversity_filter(
            sorted_configs, self.space, self.min_epicenter_dist, self.n_historical_epicenters
        )

    def _acq_starts(
        self,
        candidates: List[Config],
        acq: np.ndarray,
    ) -> List[Config]:
        """
        Select up to n_acq_epicenters configs from candidates sorted by acq score
        (ascending — lower is better), diversity-filtered independently of the
        historical epicenter set.
        """
        sorted_candidates = [candidates[i] for i in np.argsort(acq)]
        return diversity_filter(
            sorted_candidates, self.space, self.min_epicenter_dist, self.n_acq_epicenters
        )

    def _search_from(
        self,
        searcher: BaseConformalSearcher,
        start: Config,
        baseline: float,
        evaluated_hashes: Set[int],
        calls_used: int,
    ) -> Tuple[Config, float, int]:
        """
        Sequential single-coordinate perturbation walk from one base epicenter.

        Creates a fresh per-epicenter surrogate-evaluation cache (seeded from
        `evaluated_hashes`) so this walk does not interfere with sibling walks.

        Round outcomes:
          Strict improvement (best_acq < threshold): advance current, reset scale; tau continues.
          Lateral move       (best_acq == threshold): advance current, increment tau.
          Failure            (best_acq > threshold):  stay, increment tau, grow scale.

        Returns:
            (best_config_found, best_acq_found, updated_calls_used)
        """
        seen: Set[int] = set(evaluated_hashes)
        path: Set[int] = {create_config_hash(start)}

        current: Config = start
        threshold: float = baseline
        scale: float = self.initial_perturbation_scale
        tau: int = 0

        best_config: Config = start
        best_acq: float = float("inf")

        round_num = 0
        while tau <= self.tolerance_max and calls_used < self.max_surrogate_calls:
            round_num += 1

            proposals = perturb(
                current, self.space, self._natural_scales, self.n_perturbations, scale, self._rng
            )

            novel = []
            for c in proposals:
                h = create_config_hash(c)
                if h not in path and h not in seen:
                    seen.add(h)
                    novel.append(c)

            if not novel:
                tau += 1
                scale = min(scale * self.scale_growth_factor, self.max_perturbation_scale)
                continue

            budget_left = self.max_surrogate_calls - calls_used
            if len(novel) > budget_left:
                novel = novel[:budget_left]

            cand_acq = searcher.predict(self.config_manager.tabularize_configs(novel))
            calls_used += len(novel)

            top_idx = int(np.argmin(cand_acq))
            top: Config = novel[top_idx]
            top_acq: float = float(cand_acq[top_idx])

            if top_acq < threshold:
                scale = self.initial_perturbation_scale
                threshold = top_acq
                path.add(create_config_hash(current))
                current = top
                path.add(create_config_hash(top))
                best_config, best_acq = top, top_acq

            elif top_acq == threshold:
                tau += 1
                path.add(create_config_hash(current))
                current = top
                path.add(create_config_hash(top))

            else:
                tau += 1
                scale = min(scale * self.scale_growth_factor, self.max_perturbation_scale)

        if best_acq == float("inf"):
            best_config, best_acq = start, baseline

        logger.debug(
            "Epicenter walk: %d rounds, tau=%d/%d, calls=%d, best_acq=%.6f",
            round_num, tau, self.tolerance_max, calls_used, best_acq,
        )
        return best_config, best_acq, calls_used
