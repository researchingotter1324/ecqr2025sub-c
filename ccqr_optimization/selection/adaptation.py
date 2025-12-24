import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def pinball_loss(beta: float, theta: float, alpha: float) -> float:
    return alpha * (beta - theta) - min(0, beta - theta)


class DtACI:
    def __init__(
        self,
        alpha: float = 0.1,
        gamma_values: Optional[list[float]] = None,
        use_weighted_average: bool = True,
    ):
        """Initialize DtACI adapter for coverage-level adaptation.

        Args:
            alpha: Target miscoverage level (α ∈ (0,1))
            gamma_values: Learning rates for each expert. If None, uses default values.
            use_weighted_average: If True, uses deterministic weighted average.
                If False, uses random sampling.
        """
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")

        self.alpha = alpha
        self.alpha_t = alpha
        self.use_weighted_average = use_weighted_average

        if gamma_values is None:
            gamma_values = [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.128]

        if any(gamma <= 0 for gamma in gamma_values):
            raise ValueError("All gamma values must be positive")

        self.k = len(gamma_values)
        self.gamma_values = np.asarray(gamma_values)
        self.alpha_t_candidates = np.array([alpha] * self.k)

        # Parameters for update mechanics
        self.interval = 50
        self.sigma = 1 / (2 * self.interval)
        self.eta = (
            np.sqrt(3 / self.interval)
            * np.sqrt(np.log(self.interval * self.k) + 2)
            / ((1 - alpha) ** 2 * alpha**2)
        )

        self.weights = np.ones(self.k) / self.k
        self.update_count = 0
        self.beta_history = []
        self.alpha_history = []
        self.weight_history = []

    def update(self, beta: float) -> float:
        """Update alpha values based on empirical coverage feedback.

        Updates expert weights based on pinball losses and adjusts each expert's
        alpha value using gradient steps. Returns a new alpha value selected or
        averaged according to the configuration.

        Args:
            beta: Empirical coverage feedback (β_t ∈ [0,1])

        Returns:
            Updated miscoverage level
        """
        if not 0 <= beta <= 1:
            raise ValueError(f"beta must be in [0, 1], got {beta}")

        self.update_count += 1
        self.beta_history.append(beta)

        # Compute pinball losses for each expert
        # From paper: ℓ(β_t, α_t^i) where β_t is empirical coverage and α_t^i is expert's alpha
        losses = np.array(
            [
                pinball_loss(beta=beta, theta=alpha_val, alpha=self.alpha)
                for alpha_val in self.alpha_t_candidates
            ]
        )

        updated_weights = self.weights * np.exp(-self.eta * losses)
        sum_of_updated_weights = np.sum(updated_weights)
        self.weights = (1 - self.sigma) * updated_weights + (
            (self.sigma * sum_of_updated_weights) / self.k
        )

        # Update each expert's alpha using gradient step
        # err_indicators = 1 if breach (beta < alpha), 0 if coverage (beta >= alpha)
        err_indicators = (beta < self.alpha_t_candidates).astype(float)
        self.alpha_t_candidates = self.alpha_t_candidates + self.gamma_values * (
            self.alpha - err_indicators
        )
        self.alpha_t_candidates = np.clip(self.alpha_t_candidates, 0.001, 0.999)

        if np.sum(self.weights) > 0:
            normalized_weights = self.weights / np.sum(self.weights)
        else:
            normalized_weights = np.ones(self.k) / self.k
            logger.warning("All expert weights became zero, reverting to uniform")

        if self.use_weighted_average:
            # Deterministic weighted average (Algorithm 2)
            self.alpha_t = np.sum(normalized_weights * self.alpha_t_candidates)
        else:
            # Random sampling (Algorithm 1)
            chosen_idx = np.random.choice(self.k, p=normalized_weights)
            self.alpha_t = self.alpha_t_candidates[chosen_idx]

        self.alpha_history.append(self.alpha_t)
        self.weight_history.append(normalized_weights.copy())

        return self.alpha_t
