from typing import List, Optional

import numpy as np


def monotone_rearrange(
    predictions: np.ndarray,
    quantile_levels: Optional[List[float]] = None,
) -> np.ndarray:
    """Apply Chernozhukov et al. monotone rearrangement to quantile predictions.

    Sorts prediction values row-wise to enforce monotonicity across quantile
    levels, then remaps columns back to their original order. When
    ``quantile_levels`` are omitted (or are already in ascending order), the
    function reduces to a plain row-wise sort.

    Args:
        predictions: Array of shape ``(n_samples, n_quantiles)`` where each
            column corresponds to one entry of ``quantile_levels``.
        quantile_levels: Quantile levels for each column, in any order. When
            ``None``, columns are assumed to be in ascending order and the
            result is simply the row-sorted ``predictions``.

    Returns:
        Array with the same shape as ``predictions``, with values rearranged
        so that columns associated with higher quantile levels are
        monotonically non-decreasing relative to lower quantile levels.
    """
    sorted_preds = np.sort(predictions, axis=1)
    if quantile_levels is None:
        return sorted_preds

    sorted_idx = np.argsort(quantile_levels)
    reverse_idx = np.empty_like(sorted_idx)
    reverse_idx[sorted_idx] = np.arange(len(quantile_levels))

    return sorted_preds[:, reverse_idx]
