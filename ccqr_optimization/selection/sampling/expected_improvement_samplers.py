from typing import Dict, List, Literal, Optional, Union

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
from ccqr_optimization.utils.tracking import BaseConfigurationManager
from ccqr_optimization.wrapping import ConformalBounds, ParameterRange


class ExpectedImprovementSampler:
    """Expected Improvement acquisition strategy using conformal prediction intervals.

    EI is estimated via Monte Carlo sampling from quantile-based conformal
    prediction intervals, providing a principled exploration-exploitation
    balance without explicit posterior models.

    Holds an optional local search algorithm. When ``local_search`` is set,
    ``select_next`` runs the algorithm to refine the best candidate from the
    random pool; otherwise it returns the argmin-scored candidate directly.
    """

    def __init__(
        self,
        n_quantiles: int = 4,
        adapter: Optional[Literal["DtACI", "ACI"]] = None,
        current_best_value: float = float("inf"),
        num_ei_samples: int = 20,
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
            num_ei_samples: Number of Monte Carlo samples for EI estimation. Typical: 10-50.
            local_search: Optional local search algorithm applied after initial candidate
                scoring. ``None`` returns the best-scored candidate from the random pool
                directly.
        """
        validate_even_quantiles(n_quantiles=n_quantiles, sampler_name="Expected Improvement")

        self.n_quantiles = n_quantiles
        self.current_best_value = current_best_value
        self.num_ei_samples = num_ei_samples
        self.local_search: Optional[BaseLocalSearchAlgorithm] = local_search

        self.alphas = initialize_quantile_alphas(n_quantiles=n_quantiles)
        self.adapters = initialize_multi_adapters(alphas=self.alphas, adapter=adapter)

    def update_best_value(self, value: float) -> None:
        """Update the current best observed value.

        ``value`` must be in signed minimization space (i.e.
        ``metric_sign * raw_performance``), matching the space in which the
        model is trained and the conformal intervals are produced.

        Args:
            value: Newly observed signed performance (``metric_sign * raw``).
        """
        self.current_best_value = min(self.current_best_value, value)

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

    def calculate_expected_improvement(
        self,
        predictions_per_interval: List[ConformalBounds],
    ) -> np.ndarray:
        """Calculate Expected Improvement for each candidate via Monte Carlo sampling.

        Methodology:
            1. Flatten prediction intervals into a matrix representation.
            2. Randomly sample from intervals for each observation.
            3. Compute improvements: ``max(0, current_best - sampled_value)``.
               Because the model is trained on ``metric_sign``-adjusted targets
               and ``current_best_value`` is maintained in the same signed space,
               no additional ``metric_sign`` multiplication is needed.
            4. Estimate EI as the sample mean across draws.
            5. Return negated EI so that lower values indicate higher improvement
               (lower-is-better convention).

        Args:
            predictions_per_interval: ConformalBounds for each confidence level,
                in signed minimization space.

        Returns:
            Array of shape (n_observations,) with negated EI (lower-is-better).
        """
        all_bounds = flatten_conformal_bounds(predictions_per_interval=predictions_per_interval)
        n_observations = len(predictions_per_interval[0].lower_bounds)

        idxs = np.random.randint(0, all_bounds.shape[1], size=(n_observations, self.num_ei_samples))

        realizations = np.zeros((n_observations, self.num_ei_samples))
        for i in range(n_observations):
            realizations[i] = all_bounds[i, idxs[i]]

        improvements = np.maximum(0, self.current_best_value - realizations)
        expected_improvements = np.mean(improvements, axis=1)

        return -expected_improvements

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
        return self.calculate_expected_improvement(predictions_per_interval=intervals)

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
