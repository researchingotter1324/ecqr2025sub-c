from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

from ccqr_optimization.utils.configurations.sampling import get_tuning_configurations
from ccqr_optimization.utils.configurations.utils import create_config_hash
from ccqr_optimization.utils.tracking import BaseConfigurationManager
from ccqr_optimization.wrapping import ParameterRange

Config = Dict


def excluded_config_hashes(config_manager: BaseConfigurationManager) -> Set[int]:
    """Hashes of configs that must not be returned as the next evaluation.

    Includes true-objective history and banned (NaN) configurations.
    """
    excluded = set(config_manager.searched_config_hashes)
    excluded.update(
        create_config_hash(cfg) for cfg in config_manager.banned_configurations
    )

    return excluded


class AcquisitionScoreCache:
    """Cache of acquisition scores with a cap on *additional* surrogate evaluations.

    ``score_pool`` scores the random candidate pool and does not consume
    ``max_eval``. Subsequent ``score`` calls consume one unit per uncached
    config. Already-scored configs are free. ``max_eval is None`` means no cap.

    Excluded (already true-evaluated) configs may be scored so they can seed
    walks, but ``best_novel`` never returns them.
    """

    def __init__(
        self,
        predict_fn: Callable[[List[Config]], np.ndarray],
        excluded_hashes: Set[int],
        max_eval: Optional[int],
    ) -> None:
        self.predict_fn = predict_fn
        self.excluded = excluded_hashes
        self.max_eval = max_eval
        self.n_eval = 0
        self.scores: Dict[int, float] = {}
        self.configs: Dict[int, Config] = {}

    def is_novel(self, config: Config) -> bool:
        return create_config_hash(config) not in self.excluded

    def remaining(self) -> Optional[int]:
        if self.max_eval is None:
            remaining_evals = None
        else:
            remaining_evals = max(0, self.max_eval - self.n_eval)

        return remaining_evals

    def budget_exhausted(self) -> bool:
        return self.max_eval is not None and self.n_eval >= self.max_eval

    def score_pool(self, candidates: List[Config]) -> np.ndarray:
        """Score the random pool. Does not consume ``max_eval``."""
        raw = np.asarray(self.predict_fn(candidates)).flatten()
        for cfg, score in zip(candidates, raw):
            cfg_hash = create_config_hash(cfg)
            self.scores[cfg_hash] = float(score)
            self.configs[cfg_hash] = cfg

        return raw

    def score(self, configs: List[Config]) -> np.ndarray:
        """Score configs, consuming budget for uncached entries only.

        Raises:
            ValueError: If scoring the uncached subset would exceed ``max_eval``.
        """
        out = np.empty(len(configs))
        uncached: List[Config] = []
        uncached_idx: List[int] = []
        for idx, cfg in enumerate(configs):
            cfg_hash = create_config_hash(cfg)
            cached = self.scores.get(cfg_hash)
            if cached is not None:
                out[idx] = cached
                continue
            uncached.append(cfg)
            uncached_idx.append(idx)

        if uncached:
            remaining = self.remaining()
            if remaining is not None and len(uncached) > remaining:
                raise ValueError(
                    f"Requested {len(uncached)} new evaluations with only {remaining} remaining."
                )
            new_scores = np.asarray(self.predict_fn(uncached)).flatten()
            self.n_eval += len(uncached)
            for cfg, score in zip(uncached, new_scores):
                cfg_hash = create_config_hash(cfg)
                self.scores[cfg_hash] = float(score)
                self.configs[cfg_hash] = cfg
            for idx, score in zip(uncached_idx, new_scores):
                out[idx] = float(score)

        return out

    def best_novel(self) -> Tuple[Config, float]:
        """Lowest-acquisition novel config and its score."""
        best_hash = None
        best_score = np.inf
        for cfg_hash, score in self.scores.items():
            if cfg_hash in self.excluded:
                continue
            if score < best_score:
                best_score = score
                best_hash = cfg_hash
        if best_hash is None:
            raise ValueError("No novel configuration has been scored.")

        return self.configs[best_hash], best_score


def fill_remaining_eval_budget(
    cache: AcquisitionScoreCache,
    search_space: Dict[str, ParameterRange],
    rng: np.random.Generator,
) -> None:
    """Use leftover ``max_eval`` on random novel configs not yet in ``cache``.

    Called after local search stops (starts exhausted, plateau, etc.) so the
    total additional surrogate budget is still spent when possible.
    """
    if cache.max_eval is not None:
        oversample_factor = 4
        max_empty_rounds = 5
        empty_rounds = 0
        keep_filling = True

        while keep_filling and not cache.budget_exhausted():
            remaining = cache.remaining()
            if remaining is None or remaining <= 0:
                keep_filling = False
            else:
                batch_size = min(remaining, 256)
                seed = int(rng.integers(0, 2**31 - 1))
                proposals = get_tuning_configurations(
                    parameter_grid=search_space,
                    n_configurations=batch_size * oversample_factor,
                    random_state=seed,
                )
                to_score: List[Config] = []
                for cfg in proposals:
                    if not cache.is_novel(cfg):
                        continue
                    if create_config_hash(cfg) in cache.scores:
                        continue
                    to_score.append(cfg)
                    if len(to_score) >= batch_size:
                        break

                if not to_score:
                    empty_rounds += 1
                    if empty_rounds >= max_empty_rounds:
                        keep_filling = False
                else:
                    empty_rounds = 0
                    cache.score(to_score)


class BaseLocalSearchAlgorithm(ABC):
    """Abstract base for local search algorithms over the acquisition surface.

    Subclasses implement distinct search heuristics but share the same
    ``optimize`` interface so that samplers can use them interchangeably.

    All prediction is performed through an opaque ``predict_fn`` callable rather
    than a live searcher reference, so local search algorithms have no dependency
    on the acquisition or sampler layers above them.

    Subclasses accept an optional ``random_state`` seed (plain ``int``/``None``,
    matching this codebase's usual convention) at construction, and use it to
    build their own ``self.rng`` (an ``np.random.Generator``) once, rather than
    drawing from unseeded OS entropy or ambient global ``numpy.random`` state.
    That ``rng`` instance persists and keeps evolving across every subsequent
    ``optimize()`` call for the life of this instance: ``random_state`` is a
    single run-level starting point, not something re-applied (and thus not
    something that resets the draw sequence) on every trial or search restart.
    """

    @abstractmethod
    def optimize(
        self,
        predict_fn: Callable[[List[Dict]], np.ndarray],
        candidates: List[Dict],
        config_manager: BaseConfigurationManager,
        search_space: Dict[str, ParameterRange],
        metric_sign: int,
    ) -> Tuple[Dict, float]:
        """Run local search and return the best novel configuration found.

        Args:
            predict_fn: Callable that maps a list of configuration dicts to a
                flat ``np.ndarray`` of acquisition values (lower-is-better).
                Built by the sampler as a closure over its estimators.
            candidates: Random candidate pool. Must be non-empty and contain only
                configs that have not yet been true-evaluated.
            config_manager: Exposes ``tabularize_configs``, ``searched_configs``,
                and ``searched_performances``.
            search_space: Mapping from parameter name to ``ParameterRange``.
            metric_sign: ``+1`` for minimization, ``-1`` for maximization.

        Returns:
            ``(config, acquisition)`` for the novel configuration with the lowest
            acquisition among the random pool and all locally scored neighbours.
        """
