"""
Utility functions for sampling strategies in conformal prediction.

This module provides shared functionality used across different sampler implementations,
including alpha initialization strategies, adapter configuration for interval width
adjustment, and common preprocessing utilities. The module implements quantile-based
alpha initialization following symmetric quantile pairing methodology and provides
standardized interfaces for interval width adaptation using coverage rate feedback.

Key architectural components:
- Quantile-based alpha value initialization using symmetric pairing
- Multi-adapter configuration for complex sampling strategies
- Interval width update mechanisms with coverage rate feedback
- Validation utilities for sampling parameter constraints
- Conformal bounds preprocessing for efficient computation

Integration context:
The utilities in this module are designed to be used by all sampling strategy
implementations, providing consistent interfaces for common operations while
allowing each sampler to implement its specific acquisition logic.
"""

from typing import Optional, List, Literal
import warnings
from ccqr_optimization.selection.adaptation import DtACI
from ccqr_optimization.wrapping import ConformalBounds
import numpy as np


def initialize_quantile_alphas(n_quantiles: int) -> List[float]:
    """Initialize alpha values from symmetric quantile pairing.

    Creates alpha values by pairing quantiles symmetrically. Returns alpha values
    in decreasing order (increasing confidence levels).

    Args:
        n_quantiles: Number of quantiles. Must be even.

    Returns:
        List of alpha values. Length is n_quantiles // 2.

    Raises:
        ValueError: If n_quantiles is not even.
    """
    if n_quantiles % 2 != 0:
        raise ValueError("Number of quantiles must be even.")

    starting_quantiles = [
        round(i / (n_quantiles + 1), 2) for i in range(1, n_quantiles + 1)
    ]
    alphas = []
    half_length = len(starting_quantiles) // 2

    for i in range(half_length):
        lower, upper = starting_quantiles[i], starting_quantiles[-(i + 1)]
        alphas.append(1 - (upper - lower))
    return alphas


def initialize_multi_adapters(
    alphas: List[float], adapter: Optional[Literal["DtACI", "ACI"]] = None
) -> Optional[List[DtACI]]:
    """Initialize adapters for each alpha value.

    Creates individual adapter instances for each alpha level.

    Args:
        alphas: List of alpha values requiring adapters.
        adapter: Adaptation strategy ("DtACI" or "ACI") or None.

    Returns:
        List of adapter instances or None if no adaptation.

    Raises:
        ValueError: If adapter type is not recognized.
    """
    if adapter is None:
        return None
    elif adapter == "DtACI":
        return [
            DtACI(
                alpha=alpha,
                gamma_values=[0.001, 0.002, 0.004, 0.008, 0.0160, 0.032, 0.064, 0.128],
            )
            for alpha in alphas
        ]
    elif adapter == "ACI":
        return [DtACI(alpha=alpha, gamma_values=[0.005]) for alpha in alphas]
    else:
        raise ValueError("adapter must be None, 'DtACI', or 'ACI'")


def initialize_single_adapter(
    alpha: float, adapter: Optional[Literal["DtACI", "ACI"]] = None
) -> Optional[DtACI]:
    """Initialize an adapter for a single alpha value.

    Args:
        alpha: Miscoverage rate for the interval.
        adapter: Adaptation strategy ("DtACI" or "ACI") or None.

    Returns:
        Adapter instance or None if no adaptation.

    Raises:
        ValueError: If adapter type is not recognized.
    """
    if adapter is None:
        return None
    elif adapter == "DtACI":
        return DtACI(
            alpha=alpha,
            gamma_values=[0.001, 0.002, 0.004, 0.008, 0.0160, 0.032, 0.064, 0.128],
        )
    elif adapter == "ACI":
        return DtACI(alpha=alpha, gamma_values=[0.005])
    else:
        raise ValueError("adapter must be None, 'DtACI', or 'ACI'")


def update_multi_interval_widths(
    adapters: Optional[List[DtACI]], alphas: List[float], betas: List[float]
) -> List[float]:
    """Update alpha values for multiple intervals using coverage feedback.

    Args:
        adapters: List of adapter instances or None.
        alphas: Current alpha values for each interval.
        betas: Observed coverage rates for each interval.

    Returns:
        Updated alpha values or original alphas if no adapters.
    """
    if adapters:
        updated_alphas = []
        for i, (adapter, beta) in enumerate(zip(adapters, betas)):
            updated_alpha = adapter.update(beta=beta)
            updated_alphas.append(updated_alpha)
        return updated_alphas
    else:
        return alphas


def update_single_interval_width(
    adapter: Optional[DtACI], alpha: float, beta: float
) -> float:
    """Update alpha value for a single interval using coverage feedback.

    Args:
        adapter: Adapter instance or None.
        alpha: Current alpha value.
        beta: Observed coverage rate.

    Returns:
        Updated alpha value or original alpha if no adapter.

    Warns:
        UserWarning: If no adapter was initialized.
    """
    if adapter is not None:
        return adapter.update(beta=beta)
    else:
        warnings.warn(
            "'update_interval_width()' method was called, but no adapter was initialized."
        )
        return alpha


def validate_even_quantiles(n_quantiles: int, sampler_name: str = "sampler") -> None:
    """Validate that n_quantiles is even for symmetric pairing.

    Args:
        n_quantiles: Number of quantiles to validate.
        sampler_name: Name of the sampler for error messages.

    Raises:
        ValueError: If n_quantiles is not even.
    """
    if n_quantiles % 2 != 0:
        raise ValueError(f"Number of {sampler_name} quantiles must be even.")


def flatten_conformal_bounds(
    predictions_per_interval: List[ConformalBounds],
) -> np.ndarray:
    """Flatten ConformalBounds objects into matrix form.

    Returns shape (n_observations, n_intervals * 2) with columns alternating
    between lower and upper bounds for each interval.

    Args:
        predictions_per_interval: List of ConformalBounds objects.

    Returns:
        Flattened bounds array.
    """
    n_points = len(predictions_per_interval[0].lower_bounds)
    all_bounds = np.zeros((n_points, len(predictions_per_interval) * 2))
    for i, interval in enumerate(predictions_per_interval):
        all_bounds[:, i * 2] = interval.lower_bounds.flatten()
        all_bounds[:, i * 2 + 1] = interval.upper_bounds.flatten()
    return all_bounds
