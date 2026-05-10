from typing import Dict, List, Literal, Optional

import numpy as np

from ccqr_optimization.selection.conformalization import QuantileConformalEstimator
from ccqr_optimization.selection.estimation import PointEstimator
from ccqr_optimization.selection.sampling.utils import (
    flatten_conformal_bounds,
    initialize_multi_adapters,
    initialize_quantile_alphas,
    update_multi_interval_widths,
    validate_even_quantiles,
)
from ccqr_optimization.utils.tracking import BaseConfigurationManager
from ccqr_optimization.wrapping import ConformalBounds


class ThompsonSampler:
    """Thompson sampling acquisition strategy for conformal prediction optimization.

    Randomly draws values from quantile-based prediction intervals to simulate
    posterior sampling. Optionally caps drawn values by point estimates to
    encourage exploitation in promising regions.
    """

    def __init__(
        self,
        n_quantiles: int = 4,
        adapter: Optional[Literal["DtACI", "ACI"]] = None,
        enable_optimistic_sampling: bool = False,
    ) -> None:
        """
        Args:
            n_quantiles: Number of quantiles for interval construction. Must be even
                for symmetric pairing. Typical values: 4-8.
            adapter: Interval width adaptation strategy. ``"DtACI"`` is aggressive
                multi-scale adaptation; ``"ACI"`` is conservative; ``None`` disables.
            enable_optimistic_sampling: When True, sampled values are capped by point
                estimates to encourage exploitation of promising regions. A
                ``PointEstimator`` must be passed at ``score``/``select_next`` time.
        """
        validate_even_quantiles(n_quantiles=n_quantiles, sampler_name="Thompson")

        self.n_quantiles = n_quantiles
        self.enable_optimistic_sampling = enable_optimistic_sampling

        self.alphas = initialize_quantile_alphas(n_quantiles=n_quantiles)
        self.adapters = initialize_multi_adapters(alphas=self.alphas, adapter=adapter)

    def fetch_alphas(self) -> List[float]:
        """Return current alpha values, ordered from lowest to highest confidence."""
        return self.alphas

    def update_interval_width(self, betas: List[float]) -> None:
        """Update interval widths using observed coverage rates.

        Args:
            betas: Observed coverage rates for each interval, one per alpha level.
        """
        self.alphas = update_multi_interval_widths(
            adapters=self.adapters, alphas=self.alphas, betas=betas
        )

    def calculate_thompson_predictions(
        self,
        predictions_per_interval: List[ConformalBounds],
        point_predictions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate Thompson sampling predictions by randomly sampling from intervals.

        Args:
            predictions_per_interval: ConformalBounds for each confidence level.
            point_predictions: Optional point estimates. When provided and
                ``enable_optimistic_sampling`` is True, sampled values are capped
                at point estimates to encourage exploitation.

        Returns:
            Array of shape (n_observations,) with sampled predictions.
        """
        all_bounds = flatten_conformal_bounds(
            predictions_per_interval=predictions_per_interval
        )
        n_observations = len(predictions_per_interval[0].lower_bounds)
        n_intervals = all_bounds.shape[1]

        idx = np.random.randint(0, n_intervals, size=n_observations)
        sampled_bounds = np.array([all_bounds[i, idx[i]] for i in range(n_observations)])

        if self.enable_optimistic_sampling and point_predictions is not None:
            sampled_bounds = np.minimum(sampled_bounds, point_predictions)

        return sampled_bounds

    def score(
        self,
        conformal_estimator: QuantileConformalEstimator,
        X: np.ndarray,
        point_estimator: Optional[PointEstimator] = None,
    ) -> np.ndarray:
        """Thompson acquisition values for ``X`` (lower-is-better).

        Stochastic: repeated calls on the same ``X`` return different scores.
        When ``enable_optimistic_sampling`` is True, ``point_estimator`` must be
        provided or sampled values will not be capped.

        Args:
            conformal_estimator: Fitted ``QuantileConformalEstimator``.
            X: Tabularized candidate features, shape (n_candidates, n_features).
            point_estimator: Fitted ``PointEstimator``. Required only when
                ``enable_optimistic_sampling`` is True.

        Returns:
            Sampled bounds in signed minimization space, shape (n_candidates,).
        """
        intervals = conformal_estimator.predict_intervals(X)
        if self.enable_optimistic_sampling and point_estimator is not None:
            point_predictions = point_estimator.predict(X)
        else:
            point_predictions = None
        return self.calculate_thompson_predictions(
            predictions_per_interval=intervals, point_predictions=point_predictions
        )

    def select_next(
        self,
        conformal_estimator: QuantileConformalEstimator,
        candidates: List[Dict],
        config_manager: BaseConfigurationManager,
        point_estimator: Optional[PointEstimator] = None,
    ) -> Dict:
        """Select the next configuration via Thompson sampling.

        Thompson sampling does not support local search. Returns the candidate
        with the lowest sampled acquisition value.

        Args:
            conformal_estimator: Fitted ``QuantileConformalEstimator``.
            candidates: Random candidate pool. Must be non-empty.
            config_manager: Exposes ``tabularize_configs``.
            point_estimator: Fitted ``PointEstimator``. Required only when
                ``enable_optimistic_sampling`` is True.

        Returns:
            Selected configuration dict.
        """
        X = config_manager.tabularize_configs(candidates)
        scores = self.score(
            conformal_estimator=conformal_estimator,
            X=X,
            point_estimator=point_estimator,
        )
        return candidates[int(np.argmin(scores))]
