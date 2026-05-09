"""Bound-based acquisition strategies for conformal prediction optimization.

Implements two acquisition strategies that score candidates using prediction
interval bounds:

    PessimisticLowerBoundSampler: ranks candidates by the raw lower bound of the
        prediction interval (risk-averse, no point estimator required).

    LowerBoundSampler: classical Lower Confidence Bound that combines a point
        estimate with an exploration bonus proportional to the interval
        half-width, with optional decaying ``beta`` schedules.

Both samplers expose:
    score(searcher, X): the acquisition value used by local search
        (lower-is-better).
    select_next(searcher, candidates, ...): top-level selection that scores the
        candidate pool, optionally refines via a local search algorithm, and
        returns the chosen configuration.
"""

from typing import Dict, List, Literal, Optional, Union

import numpy as np

from ccqr_optimization.selection.sampling.local_search.base import BaseLocalSearchAlgorithm
from ccqr_optimization.utils.tracking import BaseConfigurationManager
from ccqr_optimization.selection.sampling.local_search.dfo_search import DFOLocalSearch
from ccqr_optimization.selection.sampling.local_search.smac_search import SmacLocalSearch
from ccqr_optimization.selection.sampling.utils import (
    initialize_single_adapter,
    update_single_interval_width,
)
from ccqr_optimization.wrapping import ParameterRange


class PessimisticLowerBoundSampler:
    """Conservative acquisition strategy using pessimistic lower bounds.

    Ranks candidates by the lower bound of their prediction interval. Optional
    local search refinement is supported via a ``BaseLocalSearchAlgorithm``
    instance passed at construction time.
    """

    def __init__(
        self,
        interval_width: float = 0.8,
        adapter: Optional[Literal["DtACI", "ACI"]] = None,
        local_search_algorithm: Optional[Union[DFOLocalSearch, SmacLocalSearch]] = None,
    ) -> None:
        """
        Args:
            interval_width: Confidence level for prediction intervals (e.g. 0.8 for
                80% intervals). Higher values give wider, more conservative bounds.
            adapter: Interval width adaptation strategy. ``"DtACI"`` is aggressive
                multi-scale adaptation; ``"ACI"`` is conservative; ``None`` disables.
            local_search_algorithm: Optional local search algorithm applied after
                initial candidate scoring. ``None`` returns the best candidate
                from the random pool directly.
        """
        self.interval_width = interval_width
        self.alpha = 1 - interval_width
        self.adapter = initialize_single_adapter(self.alpha, adapter)
        self.local_search_algorithm: Optional[BaseLocalSearchAlgorithm] = local_search_algorithm

    def fetch_alphas(self) -> List[float]:
        """Return the current alpha as a single-element list."""
        return [self.alpha]

    def update_interval_width(self, beta: float) -> None:
        """Update interval width using observed coverage feedback.

        Args:
            beta: Observed coverage rate for the prediction interval.
        """
        self.alpha = update_single_interval_width(self.adapter, self.alpha, beta)

    def score(self, searcher, X: np.ndarray) -> np.ndarray:
        """Acquisition values for ``X`` (lower-is-better).

        Returns the lower bound of the prediction interval in signed
        minimization space. The model is trained on ``metric_sign``-adjusted
        targets, so prediction intervals are already orientation-correct:
        selecting the candidate with the lowest lower bound corresponds to the
        most optimistic estimate of the signed objective, regardless of whether
        the original problem is minimization or maximization. No additional
        ``metric_sign`` multiplication is needed here.

        Used by both top-level selection and any local search algorithm that
        invokes ``searcher.predict``, which delegates to this method.

        Args:
            searcher: Fitted conformal searcher.
            X: Tabularized candidate features, shape (n_candidates, n_features).

        Returns:
            Lower bounds of the prediction interval in signed minimization
            space, shape (n_candidates,). Lower is better.
        """
        intervals = searcher.predict_intervals(X)
        return intervals[0].lower_bounds

    def select_next(
        self,
        searcher,
        candidates: List[Dict],
        config_manager: BaseConfigurationManager,
        search_space: Dict[str, ParameterRange],
        metric_sign: int,
    ) -> Dict:
        """Select the next configuration to evaluate.

        Scores all candidates by their pessimistic lower bound; if a local
        search algorithm is configured, runs it to find a better point than the
        best candidate in the random pool.

        Args:
            searcher: Fitted conformal searcher.
            candidates: Random candidate pool. Must be non-empty.
            config_manager: Configuration manager exposing ``tabularize_configs``.
            search_space: Mapping from parameter name to ``ParameterRange``.
            metric_sign: ``+1`` for minimization, ``-1`` for maximization.
                Not used for acquisition scoring (the model is already trained
                on signed targets). Forwarded to the local search algorithm
                solely so it can rank raw historical performances when
                selecting epicenter / start-point candidates.

        Returns:
            Selected configuration dict.
        """
        X = config_manager.tabularize_configs(candidates)
        scores = self.score(searcher, X)
        if self.local_search_algorithm is None:
            optimum = candidates[int(np.argmin(scores))]
        else:
            optimum = self.local_search_algorithm.optimize(
                searcher=searcher,
                candidates=candidates,
                config_manager=config_manager,
                search_space=search_space,
                metric_sign=metric_sign,
            )
        return optimum


class LowerBoundSampler(PessimisticLowerBoundSampler):
    """Lower Confidence Bound acquisition strategy with adaptive exploration.

    Extends the pessimistic lower bound approach by replacing the raw lower
    bound with ``mu - beta * half_width``, where ``mu`` is a point estimate
    and ``beta`` decays over time. Inherits ``select_next`` from the parent;
    the LCB-specific scoring is supplied by overriding ``score``.

    Decay schedules:
        ``inverse_square_root_decay``: ``beta(t) = sqrt(c / t)``
        ``logarithmic_decay``:         ``beta(t) = sqrt(c * log(t) / t)``
    """

    def __init__(
        self,
        interval_width: float = 0.8,
        adapter: Optional[Literal["DtACI", "ACI"]] = None,
        beta_decay: Optional[
            Literal["inverse_square_root_decay", "logarithmic_decay"]
        ] = "logarithmic_decay",
        c: float = 1,
        beta_max: float = 10,
        local_search_algorithm: Optional[Union[DFOLocalSearch, SmacLocalSearch]] = None,
    ) -> None:
        """
        Args:
            interval_width: Confidence level for prediction intervals.
            adapter: Interval width adaptation strategy. See parent class.
            beta_decay: Exploration parameter decay strategy.
            c: Exploration constant controlling the magnitude of the exploration bonus.
            beta_max: Maximum exploration parameter value for early-iteration stability.
            local_search_algorithm: Optional local search algorithm. See parent class.
        """
        super().__init__(interval_width, adapter, local_search_algorithm)
        self.beta_decay = beta_decay
        self.c = c
        self.t = 1
        self.beta = 1
        self.beta_max = beta_max

    def update_exploration_step(self) -> None:
        """Advance the time step and recompute the exploration parameter."""
        self.t += 1
        if self.beta_decay == "inverse_square_root_decay":
            self.beta = np.sqrt(self.c / self.t)
        elif self.beta_decay == "logarithmic_decay":
            self.beta = np.sqrt((self.c * np.log(self.t)) / self.t)
        elif self.beta_decay is None:
            self.beta = 1
        else:
            raise ValueError(
                "beta_decay must be 'inverse_square_root_decay', 'logarithmic_decay', or None."
            )

    def calculate_lcb_predictions(
        self,
        point_estimates: np.ndarray,
        half_width: np.ndarray,
    ) -> np.ndarray:
        """Compute Lower Confidence Bound acquisition values.

        LCB = mu - beta * half_width. Both ``point_estimates`` and
        ``half_width`` are in signed minimization space (the model is trained
        on ``metric_sign``-adjusted targets), so no additional sign flip is
        needed here. Lower LCB values indicate more promising candidates.

        Args:
            point_estimates: Point predictions in signed minimization space,
                shape (n_candidates,).
            half_width: Half the prediction interval width (exploration bonus),
                shape (n_candidates,).

        Returns:
            LCB values in signed minimization space; lower is better.
        """
        return point_estimates - self.beta * half_width

    def score(self, searcher, X: np.ndarray) -> np.ndarray:
        """LCB acquisition values for ``X`` (lower-is-better).

        Args:
            searcher: Fitted conformal searcher. Must additionally support ``predict_point``.
            X: Tabularized candidate features, shape (n_candidates, n_features).

        Returns:
            LCB values in signed minimization space, shape (n_candidates,).
        """
        intervals = searcher.predict_intervals(X)
        point_estimates = searcher.predict_point(X)
        half_width = np.abs(intervals[0].upper_bounds - intervals[0].lower_bounds) / 2
        return self.calculate_lcb_predictions(point_estimates, half_width)
