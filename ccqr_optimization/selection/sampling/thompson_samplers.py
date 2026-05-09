"""Thompson sampling strategy for conformal prediction acquisition.

Implements Thompson sampling using quantile-based conformal prediction intervals
as approximations to posterior distributions. Randomly draws values from
prediction intervals to balance exploration and exploitation.

Thompson sampling does not support local search. Its ``select_next`` accepts
the full uniform signature shared by all samplers but ignores ``search_space``
and ``metric_sign``.
"""

from typing import Dict, List, Literal, Optional

import numpy as np

from ccqr_optimization.utils.tracking import BaseConfigurationManager
from ccqr_optimization.selection.sampling.utils import (
    flatten_conformal_bounds,
    initialize_multi_adapters,
    initialize_quantile_alphas,
    update_multi_interval_widths,
    validate_even_quantiles,
)
from ccqr_optimization.wrapping import ConformalBounds, ParameterRange


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
                estimates to encourage exploitation of promising regions. Requires
                the searcher to expose ``predict_point``.
        """
        validate_even_quantiles(n_quantiles, "Thompson")

        self.n_quantiles = n_quantiles
        self.enable_optimistic_sampling = enable_optimistic_sampling

        self.alphas = initialize_quantile_alphas(n_quantiles)
        self.adapters = initialize_multi_adapters(self.alphas, adapter)

    def fetch_alphas(self) -> List[float]:
        """Return current alpha values, ordered from lowest to highest confidence."""
        return self.alphas

    def update_interval_width(self, betas: List[float]) -> None:
        """Update interval widths using observed coverage rates.

        Args:
            betas: Observed coverage rates for each interval, one per alpha level.
        """
        self.alphas = update_multi_interval_widths(self.adapters, self.alphas, betas)

    def calculate_thompson_predictions(
        self,
        predictions_per_interval: List[ConformalBounds],
        point_predictions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate Thompson sampling predictions by randomly sampling from intervals.

        Args:
            predictions_per_interval: ConformalBounds for each confidence level.
                All bounds must have the same number of observations.
            point_predictions: Optional point estimates. When provided and
                ``enable_optimistic_sampling`` is True, sampled values are capped
                at point estimates to encourage exploitation.

        Returns:
            Array of shape (n_observations,) with sampled predictions.
        """
        all_bounds = flatten_conformal_bounds(predictions_per_interval)
        n_observations = len(predictions_per_interval[0].lower_bounds)
        n_intervals = all_bounds.shape[1]

        idx = np.random.randint(0, n_intervals, size=n_observations)
        sampled_bounds = np.array([all_bounds[i, idx[i]] for i in range(n_observations)])

        if self.enable_optimistic_sampling and point_predictions is not None:
            sampled_bounds = np.minimum(sampled_bounds, point_predictions)

        return sampled_bounds

    def score(self, searcher, X: np.ndarray) -> np.ndarray:
        """Thompson acquisition values for ``X`` (lower-is-better).

        Used by ``select_next`` and by ``searcher.predict`` (which delegates to
        this method). Note that Thompson sampling is stochastic, so repeated
        calls on the same ``X`` return different scores.

        The model is trained on ``metric_sign``-adjusted targets
        (``metric_sign * raw_performance``), so sampled bounds are already in
        signed minimization space — no additional sign flip is needed here.
        For maximization, ``metric_sign = -1`` means lower sampled values
        correspond to higher (better) raw performance.

        When ``enable_optimistic_sampling`` is True, samples are capped from
        above by the point estimate via ``np.minimum``. In minimization space
        this prevents draws that are overly pessimistic (too high), encouraging
        exploitation of regions the model predicts as low.

        Args:
            searcher: Fitted conformal searcher. Must additionally support
                ``predict_point`` when ``enable_optimistic_sampling`` is True.
            X: Tabularized candidate features, shape (n_candidates, n_features).

        Returns:
            Sampled bounds in signed minimization space, shape (n_candidates,).
            Lower values are better; ``select_next`` uses ``argmin``.
        """
        intervals = searcher.predict_intervals(X)
        if self.enable_optimistic_sampling:
            point_predictions = searcher.predict_point(X)
        else:
            point_predictions = None
        return self.calculate_thompson_predictions(intervals, point_predictions)

    def select_next(
        self,
        searcher,
        candidates: List[Dict],
        config_manager: BaseConfigurationManager,
        search_space: Dict[str, ParameterRange] = None,
        metric_sign: int = None,
    ) -> Dict:
        """Select the next configuration to evaluate via Thompson sampling.

        Thompson sampling does not support local search, so ``search_space``
        and ``metric_sign`` are unused. Acquisition scores are directionally
        correct without any ``metric_sign`` multiplication because the model
        is trained on sign-adjusted targets (``metric_sign * raw_performance``).

        Args:
            searcher: Fitted conformal searcher.
            candidates: Random candidate pool. Must be non-empty.
            config_manager: Configuration manager exposing ``tabularize_configs``.
            search_space: Unused. Accepted for interface uniformity.
            metric_sign: Unused. Accepted for interface uniformity.

        Returns:
            Selected configuration dict.
        """
        X = config_manager.tabularize_configs(candidates)
        scores = self.score(searcher, X)
        return candidates[int(np.argmin(scores))]
