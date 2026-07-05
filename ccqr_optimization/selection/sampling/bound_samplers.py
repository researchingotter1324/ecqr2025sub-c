from typing import Dict, List, Literal, Optional, Union

import numpy as np

from ccqr_optimization.selection.conformalization import QuantileConformalEstimator
from ccqr_optimization.selection.estimation import PointEstimator
from ccqr_optimization.selection.sampling.local_search.base import BaseLocalSearchAlgorithm
from ccqr_optimization.selection.sampling.local_search.smac_search import SmacLocalSearch
from ccqr_optimization.selection.sampling.utils import (
    initialize_single_adapter,
    update_single_interval_width,
)
from ccqr_optimization.utils.tracking import BaseConfigurationManager
from ccqr_optimization.wrapping import ParameterRange


class PessimisticLowerBoundSampler:
    """Conservative acquisition strategy using pessimistic lower bounds.

    Ranks candidates by the lower bound of their prediction interval. Holds an
    optional local search algorithm; ``select_next`` applies it when set.
    """

    def __init__(
        self,
        interval_width: float = 0.8,
        adapter: Optional[Literal["DtACI", "ACI"]] = None,
        local_search: Optional[SmacLocalSearch] = None,
    ) -> None:
        """
        Args:
            interval_width: Confidence level for prediction intervals (e.g. 0.8 for
                80% intervals). Higher values give wider, more conservative bounds.
            adapter: Interval width adaptation strategy. ``"DtACI"`` is aggressive
                multi-scale adaptation; ``"ACI"`` is conservative; ``None`` disables.
            local_search: Optional local search algorithm applied after initial
                candidate scoring. ``None`` returns the best-scored candidate
                from the random pool directly.
        """
        self.interval_width = interval_width
        self.alpha = 1 - interval_width
        self.adapter = initialize_single_adapter(alpha=self.alpha, adapter=adapter)
        self.local_search: Optional[BaseLocalSearchAlgorithm] = local_search

    def fetch_alphas(self) -> List[float]:
        """Return the current alpha as a single-element list."""
        return [self.alpha]

    def update_interval_width(self, beta: float) -> None:
        """Update interval width using observed coverage feedback.

        Args:
            beta: Observed coverage rate for the prediction interval.
        """
        self.alpha = update_single_interval_width(
            adapter=self.adapter, alpha=self.alpha, beta=beta
        )

    def score(
        self,
        conformal_estimator: QuantileConformalEstimator,
        X: np.ndarray,
    ) -> np.ndarray:
        """Acquisition values for ``X`` (lower-is-better).

        Returns the lower bound of the prediction interval in signed
        minimization space. The model is trained on ``metric_sign``-adjusted
        targets, so prediction intervals are already orientation-correct.

        Args:
            conformal_estimator: Fitted ``QuantileConformalEstimator``.
            X: Tabularized candidate features, shape (n_candidates, n_features).

        Returns:
            Lower bounds of the prediction interval in signed minimization
            space, shape (n_candidates,). Lower is better.
        """
        intervals = conformal_estimator.predict_intervals(X)
        return intervals[0].lower_bounds

    def select_next(
        self,
        conformal_estimator: QuantileConformalEstimator,
        candidates: List[Dict],
        config_manager: BaseConfigurationManager,
        search_space: Dict[str, ParameterRange],
        metric_sign: int,
    ) -> Dict:
        """Select the next configuration to evaluate.

        Scores candidates by pessimistic lower bound. If a local search algorithm
        is configured, passes a scoring closure to it for neighbourhood refinement;
        otherwise returns the argmin-scored candidate directly.

        Args:
            conformal_estimator: Fitted ``QuantileConformalEstimator``.
            candidates: Random candidate pool. Must be non-empty.
            config_manager: Exposes ``tabularize_configs``, ``searched_configs``,
                and ``searched_performances``.
            search_space: Mapping from parameter name to ``ParameterRange``.
            metric_sign: ``+1`` for minimization, ``-1`` for maximization.

        Returns:
            Selected configuration dict.
        """
        X = config_manager.tabularize_configs(candidates)
        scores = self.score(conformal_estimator=conformal_estimator, X=X)
        if self.local_search is None:
            optimum = candidates[int(np.argmin(scores))]
        else:
            def predict_fn(cfgs: List[Dict]) -> np.ndarray:
                return self.score(
                    conformal_estimator=conformal_estimator,
                    X=config_manager.tabularize_configs(cfgs),
                )

            optimum = self.local_search.optimize(
                predict_fn=predict_fn,
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
    and ``beta`` decays over time. Inherits ``select_next`` from the parent.

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
        local_search: Optional[SmacLocalSearch] = None,
    ) -> None:
        """
        Args:
            interval_width: Confidence level for prediction intervals.
            adapter: Interval width adaptation strategy. See parent class.
            beta_decay: Exploration parameter decay strategy.
            c: Exploration constant controlling the magnitude of the exploration bonus.
            beta_max: Maximum exploration parameter value for early-iteration stability.
            local_search: Optional local search algorithm. See parent class.
        """
        super().__init__(
            interval_width=interval_width,
            adapter=adapter,
            local_search=local_search,
        )
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

        LCB = mu - beta * half_width. Lower LCB values indicate more promising
        candidates.

        Args:
            point_estimates: Point predictions in signed minimization space,
                shape (n_candidates,).
            half_width: Half the prediction interval width,
                shape (n_candidates,).

        Returns:
            LCB values in signed minimization space; lower is better.
        """
        return point_estimates - self.beta * half_width

    def score(
        self,
        conformal_estimator: QuantileConformalEstimator,
        X: np.ndarray,
        point_estimator: PointEstimator,
    ) -> np.ndarray:
        """LCB acquisition values for ``X`` (lower-is-better).

        Args:
            conformal_estimator: Fitted ``QuantileConformalEstimator``.
            X: Tabularized candidate features, shape (n_candidates, n_features).
            point_estimator: Fitted ``PointEstimator`` for point predictions.

        Returns:
            LCB values in signed minimization space, shape (n_candidates,).
        """
        intervals = conformal_estimator.predict_intervals(X)
        point_estimates = point_estimator.predict(X)
        half_width = np.abs(intervals[0].upper_bounds - intervals[0].lower_bounds) / 2
        return self.calculate_lcb_predictions(
            point_estimates=point_estimates, half_width=half_width
        )

    def select_next(
        self,
        conformal_estimator: QuantileConformalEstimator,
        candidates: List[Dict],
        config_manager: BaseConfigurationManager,
        search_space: Dict[str, ParameterRange],
        metric_sign: int,
        point_estimator: PointEstimator,
    ) -> Dict:
        """Select the next configuration to evaluate.

        Scores candidates by LCB. If a local search algorithm is configured,
        passes a scoring closure to it for neighbourhood refinement; otherwise
        returns the argmin-scored candidate directly.

        Args:
            conformal_estimator: Fitted ``QuantileConformalEstimator``.
            candidates: Random candidate pool. Must be non-empty.
            config_manager: Exposes ``tabularize_configs``, ``searched_configs``,
                and ``searched_performances``.
            search_space: Mapping from parameter name to ``ParameterRange``.
            metric_sign: ``+1`` for minimization, ``-1`` for maximization.
            point_estimator: Fitted ``PointEstimator``. Required for LCB scoring.

        Returns:
            Selected configuration dict.
        """
        X = config_manager.tabularize_configs(candidates)
        scores = self.score(
            conformal_estimator=conformal_estimator,
            X=X,
            point_estimator=point_estimator,
        )
        if self.local_search is None:
            optimum = candidates[int(np.argmin(scores))]
        else:
            def predict_fn(cfgs: List[Dict]) -> np.ndarray:
                return self.score(
                    conformal_estimator=conformal_estimator,
                    X=config_manager.tabularize_configs(cfgs),
                    point_estimator=point_estimator,
                )

            optimum = self.local_search.optimize(
                predict_fn=predict_fn,
                candidates=candidates,
                config_manager=config_manager,
                search_space=search_space,
                metric_sign=metric_sign,
            )
        return optimum
