import logging
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
from sklearn.preprocessing import StandardScaler

from ccqr_optimization.selection.conformalization import QuantileConformalEstimator
from ccqr_optimization.selection.estimation import PointEstimator, initialize_estimator
from ccqr_optimization.selection.estimator_configuration import (
    QUANTILE_TO_POINT_ESTIMATOR_MAPPING,
)
from ccqr_optimization.selection.sampling.bound_samplers import (
    LowerBoundSampler,
    PessimisticLowerBoundSampler,
)
from ccqr_optimization.selection.sampling.expected_improvement_samplers import (
    ExpectedImprovementSampler,
)
from ccqr_optimization.selection.sampling.thompson_samplers import ThompsonSampler
from ccqr_optimization.utils.tracking import BaseConfigurationManager
from ccqr_optimization.wrapping import ConformalBounds, ParameterRange

logger = logging.getLogger(__name__)

Sampler = Union[
    LowerBoundSampler,
    ThompsonSampler,
    PessimisticLowerBoundSampler,
    ExpectedImprovementSampler,
]


class MedianQuantileWrapper:
    """Exposes a quantile estimator's median column as a point-prediction interface."""

    def __init__(self, estimator) -> None:
        self.estimator = estimator

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.estimator.predict(X)[:, 0]


class QuantileConformalSearcher:
    """Conformal acquisition function backed by a quantile conformal estimator.

    Owns the full acquisition lifecycle: fitting, prediction, candidate selection,
    and post-observation updates. Local search is configured directly on each
    sampler instance; the searcher is not aware of it.

    Attributes:
        sampler: Active acquisition strategy.
        quantile_estimator_architecture: Architecture key for the quantile model.
        n_pre_conformal_trials: Minimum samples required for conformal mode.
        conformal_estimator: Fitted ``QuantileConformalEstimator``.
        point_estimator: Optional fitted ``PointEstimator``. Present only when
            the sampler requires point predictions.
        X_train: Training features from the most recent ``fit`` call.
        y_train: Training targets from the most recent ``fit`` call.
        last_beta: Most recent coverage feedback for single-alpha samplers.
    """

    def __init__(
        self,
        quantile_estimator_architecture: str,
        sampler: Sampler,
        n_pre_conformal_trials: int = 32,
        n_calibration_folds: int = 3,
        calibration_split_strategy: Literal[
            "cv", "train_test_split", "adaptive"
        ] = "adaptive",
    ) -> None:
        """
        Args:
            quantile_estimator_architecture: Architecture registered in the estimator
                registry; must support simultaneous multi-quantile estimation.
            sampler: Acquisition strategy that defines how prediction intervals are
                converted into acquisition scores. Attach a local search algorithm
                directly to the sampler (e.g. ``LowerBoundSampler(local_search=SmacLocalSearch())``).
                ``ThompsonSampler`` does not accept a local search.
            n_pre_conformal_trials: Minimum total samples required for conformal mode.
                Below this threshold, direct quantile predictions are used.
            n_calibration_folds: Number of folds for cross-validation calibration.
            calibration_split_strategy: One of ``"cv"``, ``"train_test_split"``,
                or ``"adaptive"``.
        """
        self.sampler = sampler
        self.quantile_estimator_architecture = quantile_estimator_architecture
        self.n_pre_conformal_trials = n_pre_conformal_trials
        self.n_calibration_folds = n_calibration_folds
        self.calibration_split_strategy = calibration_split_strategy

        self.scaler = StandardScaler()
        self.point_estimator: Optional[PointEstimator] = None
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.last_beta: Optional[float] = None
        self.conformal_estimator = QuantileConformalEstimator(
            quantile_estimator_architecture=self.quantile_estimator_architecture,
            alphas=self.sampler.fetch_alphas(),
            n_pre_conformal_trials=self.n_pre_conformal_trials,
            n_calibration_folds=self.n_calibration_folds,
            calibration_split_strategy=self.calibration_split_strategy,
        )

    def needs_point_estimator(self) -> bool:
        """Return True iff the configured sampler consumes point estimates.

        ``LowerBoundSampler`` always requires one. ``ThompsonSampler`` requires
        one only when ``enable_optimistic_sampling`` is True. PLB and EI do not.
        """
        lcb_needs_one = isinstance(self.sampler, LowerBoundSampler)
        thompson_needs_one = (
            isinstance(self.sampler, ThompsonSampler)
            and self.sampler.enable_optimistic_sampling
        )
        return lcb_needs_one or thompson_needs_one

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        tuning_iterations: Optional[int] = 0,
        random_state: Optional[int] = None,
    ) -> None:
        """Fit the quantile conformal estimator (and point estimator if required).

        Args:
            X: Input features, shape ``(n_samples, n_features)``.
            y: Target values, shape ``(n_samples,)``.
            tuning_iterations: Number of hyperparameter tuning iterations (0 disables).
            random_state: Random seed for reproducibility.
        """
        self.X_train = X
        self.y_train = y

        if self.needs_point_estimator():
            X_normalized = self.scaler.fit_transform(X)
            if self.quantile_estimator_architecture in QUANTILE_TO_POINT_ESTIMATOR_MAPPING:
                point_arch = QUANTILE_TO_POINT_ESTIMATOR_MAPPING[
                    self.quantile_estimator_architecture
                ]
                point_est = initialize_estimator(
                    estimator_architecture=point_arch,
                    random_state=random_state,
                )
                point_est.fit(X=X_normalized, y=y)
            else:
                quantile_est = initialize_estimator(
                    estimator_architecture=self.quantile_estimator_architecture,
                    random_state=random_state,
                )
                quantile_est.fit(X=X_normalized, y=y, quantiles=[0.5])
                point_est = MedianQuantileWrapper(estimator=quantile_est)

            self.point_estimator = PointEstimator(
                estimator=point_est,
                scaler=self.scaler,
            )

        self.conformal_estimator.fit(
            X=X,
            y=y,
            tuning_iterations=tuning_iterations,
            random_state=random_state,
        )

    def predict_intervals(self, X: np.ndarray) -> List[ConformalBounds]:
        """Return raw conformal prediction intervals for ``X``.

        Args:
            X: Candidate points, shape ``(n_candidates, n_features)``.

        Returns:
            One ``ConformalBounds`` per alpha level configured on the sampler.
        """
        return self.conformal_estimator.predict_intervals(X)

    def predict_point(self, X: np.ndarray) -> np.ndarray:
        """Return point estimates for ``X``.

        Args:
            X: Candidate points, shape ``(n_candidates, n_features)``.

        Returns:
            Point estimates, shape ``(n_candidates,)``.

        Raises:
            RuntimeError: If no point estimator was fit.
        """
        if self.point_estimator is None:
            raise RuntimeError(
                "predict_point called but no point estimator has been fit. "
                "The configured sampler did not request a point estimator at fit() time."
            )
        return self.point_estimator.predict(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Acquisition values (lower-is-better) for ``X``.

        Delegates to ``self.sampler.score``, passing the fitted estimators.

        Args:
            X: Candidate points, shape ``(n_candidates, n_features)``.

        Returns:
            Acquisition values, shape ``(n_candidates,)``.
        """
        if isinstance(self.sampler, LowerBoundSampler):
            return self.sampler.score(
                conformal_estimator=self.conformal_estimator,
                X=X,
                point_estimator=self.point_estimator,
            )
        if isinstance(self.sampler, ThompsonSampler):
            return self.sampler.score(
                conformal_estimator=self.conformal_estimator,
                X=X,
                point_estimator=self.point_estimator,
            )
        return self.sampler.score(conformal_estimator=self.conformal_estimator, X=X)

    def calculate_betas(self, X: np.ndarray, y_true: float) -> List[float]:
        """Calculate coverage feedback for adaptive alpha updating.

        Args:
            X: Configuration where the observation was made, shape ``(n_features,)``.
            y_true: Observed performance value.

        Returns:
            One beta value per alpha level.
        """
        return self.conformal_estimator.calculate_betas(X, y_true)

    def select_next(
        self,
        candidates: List[Dict],
        config_manager: BaseConfigurationManager,
        search_space: Dict[str, ParameterRange],
        metric_sign: int,
    ) -> Dict:
        """Select the next configuration to evaluate.

        Delegates to ``sampler.select_next``, passing the fitted estimators.
        Local search (if configured on the sampler) is invoked internally.

        Args:
            candidates: Random candidate pool. Must be non-empty.
            config_manager: Exposes ``tabularize_configs``, ``searched_configs``,
                and ``searched_performances``.
            search_space: Mapping from parameter name to ``ParameterRange``.
            metric_sign: ``+1`` for minimization, ``-1`` for maximization.

        Returns:
            Selected configuration dict.
        """
        if isinstance(self.sampler, ThompsonSampler):
            return self.sampler.select_next(
                conformal_estimator=self.conformal_estimator,
                candidates=candidates,
                config_manager=config_manager,
                point_estimator=self.point_estimator,
            )
        if isinstance(self.sampler, LowerBoundSampler):
            return self.sampler.select_next(
                conformal_estimator=self.conformal_estimator,
                candidates=candidates,
                config_manager=config_manager,
                search_space=search_space,
                metric_sign=metric_sign,
                point_estimator=self.point_estimator,
            )
        return self.sampler.select_next(
            conformal_estimator=self.conformal_estimator,
            candidates=candidates,
            config_manager=config_manager,
            search_space=search_space,
            metric_sign=metric_sign,
        )

    def get_interval(self, X: np.ndarray) -> Tuple[float, float]:
        """Get prediction interval bounds for a single configuration.

        Only valid when the sampler is a ``PessimisticLowerBoundSampler`` or
        ``LowerBoundSampler`` (i.e. uses a single prediction interval).

        Args:
            X: Input configuration, shape ``(n_features,)``.

        Returns:
            Tuple of ``(lower_bound, upper_bound)``.

        Raises:
            ValueError: If the sampler does not use a single prediction interval.
        """
        if not isinstance(self.sampler, PessimisticLowerBoundSampler):
            raise ValueError(
                "Interval retrieval only supported for PessimisticLowerBoundSampler "
                "and LowerBoundSampler."
            )
        intervals = self.predict_intervals(X.reshape(1, -1))
        return intervals[0].lower_bounds[0], intervals[0].upper_bounds[0]

    def update(self, X: np.ndarray, y_true: float) -> None:
        """Update searcher state with a new observation and adapt coverage levels.

        Args:
            X: Newly evaluated configuration, shape ``(n_features,)``.
            y_true: Observed performance for the configuration.
        """
        if isinstance(self.sampler, ExpectedImprovementSampler):
            self.sampler.update_best_value(y_true)
        if isinstance(self.sampler, LowerBoundSampler):
            self.sampler.update_exploration_step()

        has_nonconformity_scores = self.conformal_estimator.nonconformity_scores is not None
        uses_adaptation = (
            hasattr(self.sampler, "adapter") and self.sampler.adapter is not None
        ) or (
            hasattr(self.sampler, "adapters") and self.sampler.adapters is not None
        )

        if has_nonconformity_scores and uses_adaptation:
            betas = self.calculate_betas(X, y_true)
            if isinstance(self.sampler, (ThompsonSampler, ExpectedImprovementSampler)):
                self.sampler.update_interval_width(betas=betas)
            elif isinstance(self.sampler, PessimisticLowerBoundSampler):
                if len(betas) != 1:
                    raise ValueError("Multiple betas returned for single beta sampler.")
                self.last_beta = betas[0]
                self.sampler.update_interval_width(beta=betas[0])
            self.conformal_estimator.update_alphas(self.sampler.fetch_alphas())
