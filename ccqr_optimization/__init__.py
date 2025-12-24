import warnings

# Silence known noisy warning from statsmodels' quantile regression
try:
    from statsmodels.tools.sm_exceptions import IterationLimitWarning
except Exception:
    IterationLimitWarning = None

if IterationLimitWarning is not None:
    warnings.filterwarnings("ignore", category=IterationLimitWarning)

__all__ = []
