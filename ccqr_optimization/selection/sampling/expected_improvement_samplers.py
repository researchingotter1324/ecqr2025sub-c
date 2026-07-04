from typing import Dict, List, Literal, Optional


import numpy as np

from ccqr_optimization.selection.conformalization import QuantileConformalEstimator
from ccqr_optimization.selection.sampling.local_search.base import BaseLocalSearchAlgorithm
from ccqr_optimization.selection.sampling.local_search.smac_search import SmacLocalSearch
from ccqr_optimization.selection.sampling.utils import (
    flatten_conformal_bounds,
    initialize_multi_adapters,
    initialize_quantile_alphas,
    update_multi_interval_widths,
    validate_even_quantiles,
)
from ccqr_optimization.utils.math import monotone_rearrange
from ccqr_optimization.utils.tracking import BaseConfigurationManager
from ccqr_optimization.wrapping import ConformalBounds, ParameterRange

_EI_ZERO_THRESHOLD = 1e-10

def discretized_ei(
    quantile_values: np.ndarray,
    quantile_levels: np.ndarray,
    target: float,
) -> np.ndarray:
    """Compute discretized quantile expected improvement for minimization.

    Implements:

        EI_hat(x) = sum_{i=1}^{n-1} h_i * A_min(q_i(x), q_{i+1}(x); f*)

    where h_i = u_{i+1} - u_i and A_min is the piecewise contribution:

        A_min(a, b; f*) = f* - (a+b)/2       if b <= f*   (whole segment improves)
                        = 0                   if a >= f*   (no improvement)
                        = (f*-a)^2 / 2(b-a)  if a < f* < b (partial crossing)

    Args:
        quantile_values: Predicted quantiles in ascending order,
            shape (n_candidates, n_quantiles).
        quantile_levels: Quantile levels in ascending order, shape (n_quantiles,).
        target: Current best (target) value f*.

    Returns:
        EI values, shape (n_candidates,). All values are non-negative.
    """
    prob_masses = np.diff(quantile_levels)
    left_vals = quantile_values[:, :-1]
    right_vals = quantile_values[:, 1:]

    contrib_full = target - (left_vals + right_vals) / 2.0

    denom = np.where(right_vals > left_vals, right_vals - left_vals, 1.0)
    contrib_cross = (np.maximum(target - left_vals, 0.0) ** 2) / (2.0 * denom)

    contribution = np.where(
        right_vals <= target,
        contrib_full,
        np.where(left_vals >= target, 0.0, contrib_cross),
    )

    return np.sum(prob_masses * contribution, axis=1)


class ExpectedImprovementSampler:
    """Expected Improvement acquisition strategy using conformal prediction intervals.

    EI is approximated via discretized integration over predicted quantiles of the
    conditional predictive distribution, with intra-quantile linear interpolation.
    No extrapolation is performed beyond the outermost supplied quantiles.

    Holds an optional local search algorithm. When ``local_search`` is set,
    ``select_next`` runs the algorithm to refine the best candidate from the
    random pool; otherwise it returns the argmin-scored candidate directly.
    """

    def __init__(
        self,
        n_quantiles: int = 4,
        adapter: Optional[Literal["DtACI", "ACI"]] = None,
        current_best_value: float = float("inf"),
        target_type: Literal["incumbent", "median"] = "incumbent",
        local_search: Optional[SmacLocalSearch] = None,
    ) -> None:
        """
        Args:
            n_quantiles: Number of quantiles for interval construction. Must be even
                for symmetric pairing. Typical values: 4-8.
            adapter: Interval width adaptation strategy. ``"DtACI"`` is aggressive
                multi-scale adaptation; ``"ACI"`` is conservative; ``None`` disables.
            current_best_value: Initial best observed value for improvement computation.
                Updated automatically via ``update_best_value``.
            target_type: The target value to pitch the expected improvement against.
                ``"incumbent"`` uses the absolute best value observed so far.
                ``"median"`` uses the median of all observed values (a softer target).
            local_search: Optional local search algorithm applied after initial candidate
                scoring. ``None`` returns the best-scored candidate from the random pool
                directly.
        """
        validate_even_quantiles(n_quantiles=n_quantiles, sampler_name="Expected Improvement")

        self.n_quantiles = n_quantiles
        self.current_best_value = current_best_value
        self.target_type = target_type
        self.local_search: Optional[BaseLocalSearchAlgorithm] = local_search

        self.alphas = initialize_quantile_alphas(n_quantiles=n_quantiles)
        self.adapters = initialize_multi_adapters(alphas=self.alphas, adapter=adapter)

        self.ei_score_history: List[np.ndarray] = []
        self.y_history: List[float] = []

        self.last_ei_collapsed: Optional[int] = None
        self.last_perc_zero_ei: Optional[float] = None

    def update_best_value(self, value: float) -> None:
        """Update the current best observed value and historical values.

        ``value`` must be in signed minimization space (i.e.
        ``metric_sign * raw_performance``), matching the space in which the
        model is trained and the conformal intervals are produced.

        Args:
            value: Newly observed signed performance (``metric_sign * raw``).
        """
        self.current_best_value = min(self.current_best_value, value)
        self.y_history.append(value)

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

    def get_target_value(self) -> float:
        """Determine the target value for expected improvement based on target_type."""
        if self.target_type == "median" and self.y_history:
            return float(np.percentile(self.y_history, 50.0))

        return self.current_best_value

    def calculate_expected_improvement(
        self,
        predictions_per_interval: List[ConformalBounds],
    ) -> np.ndarray:
        """Calculate Expected Improvement for each candidate.

        Implements the discretized quantile EI approximation for minimization.

        Args:
            predictions_per_interval: ConformalBounds for each confidence level,
                outermost first, in signed minimization space.

        Returns:
            Array of shape (n_observations,) with negated EI (lower-is-better).
        """
        all_bounds = flatten_conformal_bounds(predictions_per_interval=predictions_per_interval)

        if np.isinf(self.current_best_value) and self.current_best_value > 0:
            return np.full(all_bounds.shape[0], -np.inf)

        quantile_values = monotone_rearrange(all_bounds)
        quantile_levels = np.linspace(
            1 / (self.n_quantiles + 1), self.n_quantiles / (self.n_quantiles + 1), self.n_quantiles
        )

        target = self.get_target_value()
        ei = discretized_ei(
            quantile_values=quantile_values,
            quantile_levels=quantile_levels,
            target=target,
        )

        return -ei

    def score(
        self,
        conformal_estimator: QuantileConformalEstimator,
        X: np.ndarray,
    ) -> np.ndarray:
        """EI acquisition values for ``X`` (lower-is-better, i.e. negated EI).

        Args:
            conformal_estimator: Fitted ``QuantileConformalEstimator``.
            X: Tabularized candidate features, shape (n_candidates, n_features).

        Returns:
            Negated EI, shape (n_candidates,).
        """
        intervals = conformal_estimator.predict_intervals(X)
        ei_scores = self.calculate_expected_improvement(predictions_per_interval=intervals)
        self.ei_score_history.append(-ei_scores)
        return ei_scores

    def select_next(
        self,
        conformal_estimator: QuantileConformalEstimator,
        candidates: List[Dict],
        config_manager: BaseConfigurationManager,
        search_space: Dict[str, ParameterRange],
        metric_sign: int,
    ) -> Dict:
        """Select the next configuration to evaluate.

        Scores all candidates by EI. If a local search algorithm is configured,
        passes a scoring closure to it for neighbourhood refinement; otherwise
        returns the argmin-scored candidate directly.

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
            winner_idx = int(np.argmin(scores))
            optimum = candidates[winner_idx]
            winner_ei = -scores[winner_idx]
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
            winner_score = self.score(
                conformal_estimator=conformal_estimator,
                X=config_manager.tabularize_configs([optimum]),
            )
            winner_ei = -winner_score[0]

        self.last_ei_collapsed = 1 if winner_ei <= _EI_ZERO_THRESHOLD else 0

        all_ei = np.concatenate(self.ei_score_history)
        n_zero = int(np.sum(all_ei <= _EI_ZERO_THRESHOLD))
        self.last_perc_zero_ei = 100.0 * n_zero / len(all_ei)

        return optimum
