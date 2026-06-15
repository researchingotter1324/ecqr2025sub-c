import itertools
import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ccqr_optimization.selection.sampling.local_search.base import BaseLocalSearchAlgorithm
from ccqr_optimization.utils.tracking import BaseConfigurationManager
from ccqr_optimization.wrapping import (
    CategoricalRange,
    FloatRange,
    IntRange,
    ParameterRange,
)

logger = logging.getLogger(__name__)

Config = Dict

EQ_TOL = 1e-10
SQRT12 = np.sqrt(12.0)

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



def one_exchange_neighborhood(
    config: Config,
    space: Dict[str, ParameterRange],
    scales: Dict[str, float],
    num_continuous_neighbors: int,
    stdev: float,
    rng: np.random.Generator,
) -> List[Config]:
    """Generate all one-exchange neighbours of ``config``.

    For each parameter dimension:
      - Categorical: one neighbour per alternative choice.
      - Float / Int: ``num_continuous_neighbors`` Gaussian perturbations with
        magnitude ``stdev x natural_scale``.

    Args:
        config: Source configuration.
        space: Search space parameter descriptors.
        scales: Natural scales per non-categorical parameter.
        num_continuous_neighbors: Perturbations to generate per continuous/integer dim.
        stdev: Noise magnitude relative to the parameter's natural scale.
        rng: NumPy random generator.

    Returns:
        Flat list of neighbour configs, each differing from ``config`` in one dimension.
    """
    neighbors: List[Config] = []
    for name, p in space.items():
        nat_scale = scales[name] if name in scales else 0.0
        if isinstance(p, CategoricalRange):
            for choice in p.choices:
                if choice != config[name]:
                    neighbor = config.copy()
                    neighbor[name] = choice
                    neighbors.append(neighbor)
        else:
            for _ in range(num_continuous_neighbors):
                neighbors.append(perturb_one(config, name, p, nat_scale, stdev, rng))
    return neighbors


class SmacLocalSearch(BaseLocalSearchAlgorithm):
    """SMAC-style one-exchange neighbourhood local search.

    Performs vectorised, multi-start neighbourhood search on the acquisition surface.
    Trajectories are independent and run sequentially; neighbours within each trajectory
    are scored in a single batch call for efficiency.

    Attributes:
        n_acq_starts: Start points drawn from the scored candidate pool (top by
            acquisition value). Corresponds to ``local_search_iterations`` in SMAC.
        n_historical_starts: Start points drawn from the evaluated history (top by
            observed performance, metric_sign-adjusted).
        n_steps_plateau_walk: Maximum consecutive non-improving rounds before a
            trajectory terminates. Corresponds to SMAC's ``n_steps_plateau_walk``.
        max_steps: Hard cap on total neighbourhood-generation rounds across all
            trajectories. ``None`` means only the plateau counter stops trajectories.
        num_continuous_neighbors: Gaussian perturbations generated per continuous/
            integer parameter per round. Corresponds to SMAC's ``num_neighbors``.
        stdev: Noise magnitude relative to the parameter's natural scale.
        vectorization_min_obtain: Minimum neighbours requested at the start of each
            walk and after a successful improvement.
        vectorization_max_obtain: Maximum neighbours obtainable per round (doubling cap).
        random_seed: Seed for reproducibility.
    """

    def __init__(
        self,
        n_acq_starts: int = 10,
        n_historical_starts: int = 6,
        n_steps_plateau_walk: int = 10,
        max_steps: Optional[int] = None,
        num_continuous_neighbors: int = 8,
        stdev: float = 0.2,
        vectorization_min_obtain: int = 2,
        vectorization_max_obtain: int = 64,
        random_seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            n_acq_starts: Number of start points from the top of the scored candidate pool.
            n_historical_starts: Number of start points from the evaluated history.
            n_steps_plateau_walk: Max consecutive non-improving rounds per trajectory.
            max_steps: Optional global cap on total neighbourhood-generation rounds.
            num_continuous_neighbors: Perturbations per continuous/integer dimension per round.
            stdev: Gaussian noise magnitude relative to the parameter's natural scale.
            vectorization_min_obtain: Batch size at trajectory start and after improvement.
            vectorization_max_obtain: Maximum batch size (doubles on each non-improving round).
            random_seed: RNG seed for reproducibility.
        """
        self.n_acq_starts = n_acq_starts
        self.n_historical_starts = n_historical_starts
        self.n_steps_plateau_walk = n_steps_plateau_walk
        self.max_steps = max_steps
        self.num_continuous_neighbors = num_continuous_neighbors
        self.stdev = stdev
        self.vectorization_min_obtain = vectorization_min_obtain
        self.vectorization_max_obtain = vectorization_max_obtain
        self.random_seed = random_seed

    def optimize(
        self,
        predict_fn: Callable[[List[Config]], np.ndarray],
        candidates: List[Config],
        config_manager: BaseConfigurationManager,
        search_space: Dict[str, ParameterRange],
        metric_sign: int,
    ) -> Config:
        """Run SMAC-style neighbourhood search and return the best configuration found.

        Args:
            predict_fn: Callable that maps a list of configuration dicts to a
                flat ``np.ndarray`` of acquisition values (lower-is-better).
                Built by the sampler as a closure over its estimators.
            candidates: Random candidate pool. Must be non-empty.
            config_manager: Exposes ``tabularize_configs``, ``searched_configs``,
                and ``searched_performances``.
            search_space: Parameter name to ParameterRange mapping.
            metric_sign: ``+1`` for minimization, ``-1`` for maximization.

        Returns:
            Config with the lowest acquisition value found during the search.

        Raises:
            ValueError: If ``candidates`` is empty.
        """
        if not candidates:
            raise ValueError("candidates must not be empty.")

        rng = np.random.default_rng(self.random_seed)
        scales = natural_scales(search_space)

        def predict(cfgs: List[Config]) -> np.ndarray:
            return np.asarray(predict_fn(cfgs)).flatten()

        acq_candidates = predict(candidates)
        baseline = float(np.min(acq_candidates))
        best_config: Config = candidates[int(np.argmin(acq_candidates))]
        best_acq = baseline

        logger.debug(
            "SMAC LS: Scored %d candidates. Baseline acq = %.6f", len(candidates), baseline
        )

        start_points = self.select_starts(
            candidates, acq_candidates, config_manager, metric_sign, rng
        )
        if not start_points:
            logger.warning("SMAC LS: No start points; returning best random candidate.")
            return best_config

        logger.debug("SMAC LS: %d start points selected.", len(start_points))

        global_steps = 0
        for traj_idx, start in enumerate(start_points):
            if self.max_steps is not None and global_steps >= self.max_steps:
                break

            remaining = (
                None if self.max_steps is None else self.max_steps - global_steps
            )
            found, found_acq, steps_taken = self.walk(
                predict_fn=predict,
                start=start,
                search_space=search_space,
                scales=scales,
                rng=rng,
                remaining_steps=remaining,
            )
            global_steps += steps_taken

            if found_acq < best_acq:
                best_acq, best_config = found_acq, found
                logger.debug(
                    "SMAC LS: New best acq = %.6f at trajectory %d.", best_acq, traj_idx + 1
                )

        logger.debug(
            "SMAC LS: Done. Best acq = %.6f over %d trajectories.", best_acq, len(start_points)
        )
        return best_config

    def select_starts(
        self,
        candidates: List[Config],
        acq_candidates: np.ndarray,
        config_manager: BaseConfigurationManager,
        metric_sign: int,
        rng: np.random.Generator,
    ) -> List[Config]:
        """Build the list of trajectory start points.

        Combines:
          - Top ``n_acq_starts`` candidates by acquisition value (ascending).
          - Top ``n_historical_starts`` evaluated configs by signed performance.

        Deduplication by object identity; acquisition-ranked configs appear first.

        Args:
            candidates: Candidate pool.
            acq_candidates: Acquisition scores for each candidate (lower-is-better).
            config_manager: Provides ``searched_configs`` and ``searched_performances``.
            metric_sign: ``+1`` for minimization, ``-1`` for maximization.
            rng: NumPy random generator used for tiebreaking equal acquisition values.

        Returns:
            Deduplicated list of start configurations.
        """
        # Use lexsort with a random tiebreaker, mirroring SMAC's tie-breaking for equal acq values.
        random_tiebreak = rng.random(len(acq_candidates))
        sorted_cand_idx = np.lexsort((random_tiebreak, acq_candidates))
        acq_starts = [candidates[i] for i in sorted_cand_idx[: self.n_acq_starts]]

        hist_configs: List[Config] = config_manager.searched_configs
        hist_perfs: List[float] = config_manager.searched_performances
        hist_starts: List[Config] = []
        if hist_configs:
            signed = [p * metric_sign for p in hist_perfs]
            sorted_hist = [hist_configs[i] for i in np.argsort(signed)]
            hist_starts = sorted_hist[: self.n_historical_starts]

        seen_ids = set()
        result: List[Config] = []
        for c in itertools.chain(acq_starts, hist_starts):
            cid = id(c)
            if cid not in seen_ids:
                seen_ids.add(cid)
                result.append(c)
        return result

    def walk(
        self,
        predict_fn: Callable[[List[Config]], np.ndarray],
        start: Config,
        search_space: Dict[str, ParameterRange],
        scales: Dict[str, float],
        rng: np.random.Generator,
        remaining_steps: Optional[int],
    ) -> Tuple[Config, float, int]:
        """Run one neighbourhood-walk trajectory from ``start``.

        Implements SMAC's per-trajectory logic:
          - Batch-generate ``obtain_n`` one-exchange neighbours each round.
          - Accept strict improvements immediately; collect ties for plateau walking.
          - Double ``obtain_n`` on non-improvement (up to vectorization_max_obtain);
            reset to vectorization_min_obtain on improvement.
          - Increment plateau counter on each non-improving round; stop at
            n_steps_plateau_walk.
          - Honour the optional ``remaining_steps`` hard cap.

        Args:
            predict_fn: Callable that scores a list of configs and returns a flat ndarray.
            start: Starting configuration for this trajectory.
            search_space: Search space parameter descriptors.
            scales: Natural scales per non-categorical parameter.
            rng: NumPy random generator.
            remaining_steps: Maximum rounds allowed for this walk, or None for unlimited.

        Returns:
            Tuple of (best_config, best_acq, steps_taken).
        """
        current = start
        start_acq = float(predict_fn([current])[0])
        current_acq = start_acq
        best_config, best_acq = current, current_acq

        n_no_plateau = 0
        obtain_n = self.vectorization_min_obtain
        equal_acq_pool: List[Config] = []
        steps_taken = 0

        while n_no_plateau < self.n_steps_plateau_walk:
            if remaining_steps is not None and steps_taken >= remaining_steps:
                break

            neighbors = one_exchange_neighborhood(
                current, search_space, scales,
                self.num_continuous_neighbors, self.stdev, rng,
            )
            if not neighbors:
                break

            if len(neighbors) > obtain_n:
                indices = rng.choice(len(neighbors), size=obtain_n, replace=False)
                batch = [neighbors[i] for i in indices]
            else:
                batch = neighbors

            steps_taken += 1

            acq_batch = predict_fn(batch)

            improved = False
            for neighbor, nacq in zip(batch, acq_batch):
                nacq_f = float(nacq)
                if nacq_f < current_acq - EQ_TOL:
                    current = neighbor
                    current_acq = nacq_f
                    if nacq_f < best_acq:
                        best_acq, best_config = nacq_f, neighbor
                    improved = True
                    equal_acq_pool = []
                    break
                elif abs(nacq_f - current_acq) <= EQ_TOL:
                    equal_acq_pool.append(neighbor)

            # Mirror SMAC post-round obtain_n update:
            # reset to 2 on improvement or empty batch; double (capped) otherwise.
            if obtain_n == 0 or improved:
                obtain_n = 2
            else:
                obtain_n = min(obtain_n * 2, self.vectorization_max_obtain)

            if not improved:
                if equal_acq_pool:
                    current = equal_acq_pool[0]
                    equal_acq_pool = []
                n_no_plateau += 1

        logger.debug(
            "SMAC walk done: plateau=%d/%d, steps=%d, best_acq=%.6f",
            n_no_plateau, self.n_steps_plateau_walk, steps_taken, best_acq,
        )
        return best_config, best_acq, steps_taken
