"""Conformal acquisition function backed by a quantile conformal estimator.

``QuantileConformalSearcher`` is the single searcher class. It owns:

  - The sampler (one of ``LowerBoundSampler``, ``PessimisticLowerBoundSampler``,
    ``ExpectedImprovementSampler``, ``ThompsonSampler``).
  - Fitting logic (quantile conformal estimator + optional point estimator).
  - Prediction surfaces (``predict_intervals``, ``predict_point``, ``predict``).
  - Coordination: ``select_next`` delegates to the sampler; ``update`` adapts
    coverage levels after each observation.

Dependency flow (top → bottom, no upward edges):

    wrapping / utils
         │
         ▼
    samplers / local_search
         │
         ▼
    acquisition              (imports samplers for Sampler type and dispatch)
         │
         ▼
    tuning
"""

import logging
from typing import Dict, List, Literal, Optional, Protocol, Tuple, Union

import numpy as np
from sklearn.preprocessing import StandardScaler

from ccqr_optimization.selection.conformalization import QuantileConformalEstimator
from ccqr_optimization.selection.estimation import initialize_estimator
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

DEFAULT_IG_SAMPLER_RANDOM_STATE = 1234

PointEstimatorArchitecture = Literal["gbm", "rf", "knn", "kr", "pens"]

Sampler = Union[
    LowerBoundSampler,
    ThompsonSampler,
    PessimisticLowerBoundSampler,
    ExpectedImprovementSampler,
]


class PointEstimator(Protocol):
    """Minimal interface any point estimator must expose.

    Satisfied by sklearn ``BaseEstimator`` subclasses, ``MedianQuantileWrapper``,
    and any other object that can map a feature matrix to a 1-D prediction array.
    """

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return point predictions for ``X``, shape ``(n_samples,)``."""
        ...


class MedianQuantileWrapper:
    """Adapter exposing a quantile estimator's median quantile as a point estimator."""

    def __init__(self, estimator: PointEstimator) -> None:
        """
        Args:
            estimator: A fitted quantile estimator whose ``predict`` returns
                an array of shape ``(n_samples, n_quantiles)``.
        """
        self.estimator = estimator

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the first-quantile (median) predictions.

        Args:
            X: Input features, shape ``(n_samples, n_features)``.

        Returns:
            Median predictions, shape ``(n_samples,)``.
        """
        return self.estimator.predict(X)[:, 0]


class QuantileConformalSearcher:
    """Conformal acquisition function backed by a quantile conformal estimator.

    Owns the full lifecycle: construction, fitting, prediction, candidate
    selection, and post-observation state updates.

    A point estimator is fit only when the sampler requires one:
    ``LowerBoundSampler`` always needs one; ``ThompsonSampler`` needs one only
    when ``enable_optimistic_sampling`` is True; all others do not.

    Attributes:
        sampler: Active acquisition strategy.
        quantile_estimator_architecture: Architecture key for the quantile model.
        n_pre_conformal_trials: Minimum samples required for conformal mode.
        conformal_estimator: Fitted ``QuantileConformalEstimator``.
        point_estimator: Optional fitted point estimator.
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
            sampler: Acquisition strategy that defines scoring and selection behavior.
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
                self.point_estimator = point_est
            else:
                quantile_est = initialize_estimator(
                    estimator_architecture=self.quantile_estimator_architecture,
                    random_state=random_state,
                )
                quantile_est.fit(X=X_normalized, y=y, quantiles=[0.5])
                self.point_estimator = MedianQuantileWrapper(quantile_est)

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
        return self.point_estimator.predict(self.scaler.transform(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Acquisition values (lower-is-better) for ``X``.

        Delegates to ``self.sampler.score``. Entry point for local search
        algorithms that score novel candidates during a neighbourhood walk.

        Args:
            X: Candidate points, shape ``(n_candidates, n_features)``.

        Returns:
            Acquisition values, shape ``(n_candidates,)``.
        """
        return self.sampler.score(self, X)

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

        Dispatches to the sampler's ``select_next`` with the full argument set.
        All samplers accept the same signature; those that do not use
        ``search_space`` or ``metric_sign`` ignore them.

        Args:
            candidates: Random candidate pool. Must be non-empty.
            config_manager: Configuration manager exposing ``tabularize_configs``,
                ``searched_configs``, and ``searched_performances``.
            search_space: Mapping from parameter name to ``ParameterRange``.
            metric_sign: ``+1`` for minimization, ``-1`` for maximization.

        Returns:
            Selected configuration dict.
        """
        return self.sampler.select_next(
            searcher=self,
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
            ValueError: If the sampler does not use a single prediction interval,
                or if the conformal estimator is not fitted.
        """
        if not isinstance(self.sampler, PessimisticLowerBoundSampler):
            raise ValueError(
                "Interval retrieval only supported for PessimisticLowerBoundSampler "
                "and LowerBoundSampler."
            )
        if self.conformal_estimator is None:
            raise ValueError(
                "Conformal estimator not initialized. Call fit() before getting interval."
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
