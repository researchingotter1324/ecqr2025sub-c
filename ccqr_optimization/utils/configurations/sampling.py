from typing import Dict, List, Optional, Literal
import math
import logging
import random
import numpy as np
from scipy.stats import qmc
from ccqr_optimization.wrapping import (
    IntRange,
    FloatRange,
    CategoricalRange,
    ParameterRange,
)
from ccqr_optimization.utils.configurations.utils import create_config_hash

logger = logging.getLogger(__name__)


def _sobol_row_to_config(
    row: np.ndarray,
    numeric_params: List,
    categorical_params: List,
) -> Dict:
    """Convert a Sobol sequence row to a configuration dictionary.

    Args:
        row: Single Sobol sample in [0, 1]^d.
        numeric_params: List of (index, name, ParameterRange) tuples for numeric params.
        categorical_params: List of (index, name, ParameterRange) tuples for categorical params.

    Returns:
        Configuration dictionary with mapped parameter values.
    """
    config: Dict = {}
    for dim, (_, name, pr) in enumerate(numeric_params):
        if isinstance(pr, IntRange):
            if pr.log_scale:
                lmin = np.log(max(pr.min_value, 1))
                lmax = np.log(pr.max_value)
                value = int(np.round(np.exp(lmin + row[dim] * (lmax - lmin))))
                config[name] = max(pr.min_value, min(value, pr.max_value))
            else:
                n_ints = pr.max_value - pr.min_value + 1
                value = pr.min_value + int(np.floor(row[dim] * n_ints))
                config[name] = max(pr.min_value, min(value, pr.max_value))
        else:
            if pr.log_scale:
                lmin = np.log(max(pr.min_value, 1e-10))
                lmax = np.log(pr.max_value)
                config[name] = float(np.exp(lmin + row[dim] * (lmax - lmin)))
            else:
                config[name] = float(
                    pr.min_value + row[dim] * (pr.max_value - pr.min_value)
                )
    for _, name, pr in categorical_params:
        value = random.choice(pr.choices)
        if all(isinstance(choice, bool) for choice in pr.choices):
            value = bool(value)
        config[name] = value
    return config


def get_tuning_configurations(
    parameter_grid: Dict[str, ParameterRange],
    n_configurations: int,
    random_state: Optional[int] = None,
    sampling_method: Literal["uniform", "sobol"] = "sobol",
) -> List[Dict]:
    """Generate unique parameter configurations via uniform or Sobol sampling.

    Args:
        parameter_grid: Parameter ranges dictionary.
        n_configurations: Number of configurations to generate.
        random_state: Random seed.
        sampling_method: Either 'uniform' or 'sobol' (default: 'sobol').

    Returns:
        List of unique configuration dictionaries.
    """
    if sampling_method == "sobol":
        samples = _sobol_sampling(
            parameter_grid=parameter_grid,
            n_configurations=n_configurations,
            random_state=random_state,
        )
    elif sampling_method == "uniform":
        samples = _uniform_sampling(
            parameter_grid=parameter_grid,
            n_configurations=n_configurations,
            random_state=random_state,
        )
    else:
        raise ValueError(
            f"Invalid sampling method: {sampling_method}. Must be 'uniform' or 'sobol'."
        )

    return samples


def _uniform_sampling(
    parameter_grid: Dict[str, ParameterRange],
    n_configurations: int,
    random_state: Optional[int] = None,
) -> List[Dict]:
    """Generate configurations using uniform random sampling.

    Args:
        parameter_grid: Parameter ranges dictionary.
        n_configurations: Number of configurations to generate.
        random_state: Random seed.

    Returns:
        List of unique configuration dictionaries.
    """
    configurations: List[Dict] = []
    configurations_set = set()
    if random_state is not None:
        random.seed(a=random_state)
        np.random.seed(seed=random_state)

    param_names = sorted(parameter_grid.keys())
    max_attempts = min(n_configurations * 3, 50000)
    attempts = 0
    while len(configurations) < n_configurations and attempts < max_attempts:
        config = {}
        for name in param_names:
            param_range = parameter_grid[name]
            if isinstance(param_range, IntRange):
                if param_range.log_scale:
                    lmin = np.log(max(param_range.min_value, 1))
                    lmax = np.log(param_range.max_value)
                    config[name] = int(np.round(np.exp(random.uniform(lmin, lmax))))
                    config[name] = max(
                        param_range.min_value, min(config[name], param_range.max_value)
                    )
                else:
                    config[name] = random.randint(
                        param_range.min_value, param_range.max_value
                    )
            elif isinstance(param_range, FloatRange):
                if param_range.log_scale:
                    lmin = np.log(max(param_range.min_value, 1e-10))
                    lmax = np.log(param_range.max_value)
                    config[name] = float(np.exp(random.uniform(lmin, lmax)))
                else:
                    config[name] = random.uniform(
                        param_range.min_value, param_range.max_value
                    )
            elif isinstance(param_range, CategoricalRange):
                value = random.choice(param_range.choices)
                if all(isinstance(choice, bool) for choice in param_range.choices):
                    value = bool(value)
                config[name] = value
        config_hash = create_config_hash(config)
        if config_hash not in configurations_set:
            configurations_set.add(config_hash)
            configurations.append(config)
        attempts += 1

    if len(configurations) < n_configurations:
        logger.warning(
            f"Could only generate {len(configurations)} unique configurations "
        )
    return configurations


def _sobol_sampling(
    parameter_grid: Dict[str, ParameterRange],
    n_configurations: int,
    random_state: Optional[int] = None,
) -> List[Dict]:
    """Generate configurations using low-discrepancy Sobol sequence sampling.

    Args:
        parameter_grid: Parameter ranges dictionary.
        n_configurations: Number of configurations to generate.
        random_state: Random seed.

    Returns:
        List of unique configuration dictionaries.
    """
    configurations: List[Dict] = []
    configurations_set = set()
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    param_names = sorted(parameter_grid.keys())
    param_ranges = [parameter_grid[name] for name in param_names]
    numeric_params = [
        (i, name, pr)
        for i, (name, pr) in enumerate(zip(param_names, param_ranges))
        if isinstance(pr, (IntRange, FloatRange))
    ]
    categorical_params = [
        (i, name, pr)
        for i, (name, pr) in enumerate(zip(param_names, param_ranges))
        if isinstance(pr, CategoricalRange)
    ]

    if not numeric_params:
        raise ValueError("Sobol sampling requires at least one numeric parameter.")

    if n_configurations <= 0:
        raise ValueError(
            "n_configurations must be a positive integer for Sobol sampling"
        )

    sobol_engine = qmc.Sobol(d=len(numeric_params), scramble=True, seed=random_state)
    m = math.ceil(math.log2(n_configurations)) if n_configurations > 1 else 0
    n_sobol = 2 ** m

    if n_sobol != n_configurations:
        m_down = math.floor(math.log2(n_configurations))
        n_sobol_down = 2 ** m_down
        logger.warning(
            f"n_configurations={n_configurations} is not a power of 2. "
            f"Using {n_sobol_down} configurations (2^{m_down})."
        )
        m = m_down

    samples = sobol_engine.random_base2(m)

    for row in samples:
        config = _sobol_row_to_config(row, numeric_params, categorical_params)
        config_hash = create_config_hash(config)
        if config_hash not in configurations_set:
            configurations_set.add(config_hash)
            configurations.append(config)

    return configurations
