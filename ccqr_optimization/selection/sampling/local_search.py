import random
from typing import Dict, Iterator, List, Optional

import numpy as np

from ccqr_optimization.selection.acquisition import BaseConformalSearcher
from ccqr_optimization.utils.configurations.utils import create_config_hash
from ccqr_optimization.wrapping import (
    CategoricalRange,
    FloatRange,
    IntRange,
    ParameterRange,
)


def get_one_exchange_neighborhood(
    configuration: Dict,
    search_space: Dict[str, ParameterRange],
    n_continuous_neighbors: int = 8,
    stdev_multiplier: float = 0.2,
    random_state: Optional[int] = None,
) -> Iterator[Dict]:
    """
    Generate a 1-exchange neighborhood for a given configuration.

    Yields neighboring configurations by perturbing exactly one parameter at a time.
    For continuous parameters, samples from a truncated normal distribution.
    For discrete parameters, samples and rounds, falling back to adjacent values if necessary.
    For categorical parameters, yields all other available choices in random order.

    Args:
        configuration: The base configuration to perturb.
        search_space: The parameter search space definition.
        n_continuous_neighbors: Number of neighbors to generate for numeric parameters.
        stdev_multiplier: Fraction of the parameter range to use as standard deviation.
        random_state: Random seed for reproducibility.

    Yields:
        Neighboring configuration dictionaries.
    """
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    param_names = list(search_space.keys())
    random.shuffle(param_names)

    for param_name in param_names:
        param_range = search_space[param_name]
        current_value = configuration[param_name]

        if isinstance(param_range, CategoricalRange):
            choices = [c for c in param_range.choices if c != current_value]
            random.shuffle(choices)
            for choice in choices:
                neighbor = configuration.copy()
                neighbor[param_name] = choice
                yield neighbor

        elif isinstance(param_range, FloatRange):
            yield from _generate_float_neighbors(
                configuration,
                param_name,
                param_range,
                current_value,
                n_continuous_neighbors,
                stdev_multiplier,
            )

        elif isinstance(param_range, IntRange):
            yield from _generate_int_neighbors(
                configuration,
                param_name,
                param_range,
                current_value,
                n_continuous_neighbors,
                stdev_multiplier,
            )


def _generate_float_neighbors(
    configuration: Dict,
    param_name: str,
    param_range: FloatRange,
    current_value: float,
    n_neighbors: int,
    stdev_multiplier: float,
) -> Iterator[Dict]:
    if param_range.log_scale:
        lmin = np.log(max(param_range.min_value, 1e-10))
        lmax = np.log(param_range.max_value)
        lval = np.log(max(current_value, 1e-10))
        std = stdev_multiplier * (lmax - lmin)

        samples = np.random.normal(loc=lval, scale=std, size=n_neighbors * 2)
        samples = samples[(samples >= lmin) & (samples <= lmax)]
        samples = np.exp(samples)
    else:
        vmin = param_range.min_value
        vmax = param_range.max_value
        std = stdev_multiplier * (vmax - vmin)

        samples = np.random.normal(loc=current_value, scale=std, size=n_neighbors * 2)
        samples = samples[(samples >= vmin) & (samples <= vmax)]

    yielded_count = 0
    for sample in samples:
        if yielded_count >= n_neighbors:
            break
        if sample != current_value:
            neighbor = configuration.copy()
            neighbor[param_name] = float(sample)
            yield neighbor
            yielded_count += 1


def _generate_int_neighbors(
    configuration: Dict,
    param_name: str,
    param_range: IntRange,
    current_value: int,
    n_neighbors: int,
    stdev_multiplier: float,
) -> Iterator[Dict]:
    yielded_values = {current_value}

    if param_range.log_scale:
        lmin = np.log(max(param_range.min_value, 1))
        lmax = np.log(param_range.max_value)
        lval = np.log(max(current_value, 1))
        std = stdev_multiplier * (lmax - lmin)

        samples = np.random.normal(loc=lval, scale=std, size=n_neighbors * 5)
        samples = samples[(samples >= lmin) & (samples <= lmax)]
        samples = np.round(np.exp(samples)).astype(int)
    else:
        vmin = param_range.min_value
        vmax = param_range.max_value
        std = stdev_multiplier * (vmax - vmin)

        samples = np.random.normal(loc=current_value, scale=std, size=n_neighbors * 5)
        samples = samples[(samples >= vmin) & (samples <= vmax)]
        samples = np.round(samples).astype(int)

    yielded_count = 0
    for sample in samples:
        if yielded_count >= n_neighbors:
            break
        if sample not in yielded_values:
            neighbor = configuration.copy()
            neighbor[param_name] = int(sample)
            yield neighbor
            yielded_values.add(sample)
            yielded_count += 1

    if yielded_count == 0:
        for offset in [-1, 1]:
            adj_val = current_value + offset
            if (
                param_range.min_value <= adj_val <= param_range.max_value
                and adj_val not in yielded_values
            ):
                neighbor = configuration.copy()
                neighbor[param_name] = int(adj_val)
                yield neighbor
                yielded_values.add(adj_val)


class LocalSearchOptimizer:
    """
    Optimizes the acquisition function using a vectorized, multi-trajectory local search.
    """

    def __init__(
        self,
        search_space: Dict[str, ParameterRange],
        config_manager,
        max_steps: int = 500,
        n_steps_plateau_walk: int = 24,
        vectorization_min_obtain: int = 2,
        vectorization_max_obtain: int = 96,
        n_continuous_neighbors: int = 12,
        stdev_multiplier: float = 0.2,
    ):
        self.search_space = search_space
        self.config_manager = config_manager
        self.max_steps = max_steps
        self.n_steps_plateau_walk = n_steps_plateau_walk
        self.vectorization_min_obtain = vectorization_min_obtain
        self.vectorization_max_obtain = vectorization_max_obtain
        self.n_continuous_neighbors = n_continuous_neighbors
        self.stdev_multiplier = stdev_multiplier

    def maximize(
        self,
        searcher: BaseConformalSearcher,
        starting_points: List[Dict],
    ) -> Dict:
        """
        Run parallel local searches from the given starting points to find the configuration
        that minimizes the acquisition function value.
        """
        if not starting_points:
            raise ValueError("Local search requires at least one starting point.")

        n_trajectories = len(starting_points)
        active = np.ones(n_trajectories, dtype=bool)
        n_no_plateau_walk = np.zeros(n_trajectories, dtype=int)
        obtain_n = np.full(n_trajectories, self.vectorization_min_obtain, dtype=int)

        centers = list(starting_points)

        centers_transformed = self.config_manager.tabularize_configs(centers)
        current_acq_values = searcher.predict(centers_transformed)

        plateau_lists = [[] for _ in range(n_trajectories)]

        iterators = [
            get_one_exchange_neighborhood(
                centers[i],
                self.search_space,
                self.n_continuous_neighbors,
                self.stdev_multiplier,
            )
            for i in range(n_trajectories)
        ]

        evaluated_hashes = set()
        if hasattr(self.config_manager, "searched_config_hashes"):
            evaluated_hashes.update(self.config_manager.searched_config_hashes)
        if hasattr(self.config_manager, "banned_configurations"):
            evaluated_hashes.update(
                create_config_hash(c) for c in self.config_manager.banned_configurations
            )

        for c in centers:
            evaluated_hashes.add(create_config_hash(c))

        step = 0
        while np.any(active) and step < self.max_steps:
            step += 1

            batch_configs = []
            batch_trajectory_indices = []
            iterator_exhausted = np.zeros(n_trajectories, dtype=bool)

            for i in range(n_trajectories):
                if not active[i]:
                    continue

                collected = 0
                while collected < obtain_n[i]:
                    try:
                        neighbor = next(iterators[i])
                        nhash = create_config_hash(neighbor)
                        if nhash not in evaluated_hashes:
                            evaluated_hashes.add(nhash)
                            batch_configs.append(neighbor)
                            batch_trajectory_indices.append(i)
                            collected += 1
                    except StopIteration:
                        iterator_exhausted[i] = True
                        break

            if not batch_configs:
                for i in range(n_trajectories):
                    if active[i] and iterator_exhausted[i]:
                        self._handle_exhaustion(
                            i,
                            active,
                            centers,
                            current_acq_values,
                            plateau_lists,
                            n_no_plateau_walk,
                            iterators,
                            obtain_n,
                        )
                continue

            X_batch = self.config_manager.tabularize_configs(batch_configs)
            acq_values = searcher.predict(X_batch)

            improved = np.zeros(n_trajectories, dtype=bool)

            for idx, traj_idx in enumerate(batch_trajectory_indices):
                if improved[traj_idx]:
                    continue

                val = acq_values[idx]
                neighbor = batch_configs[idx]

                if val < current_acq_values[traj_idx]:
                    centers[traj_idx] = neighbor
                    current_acq_values[traj_idx] = val
                    improved[traj_idx] = True
                    plateau_lists[traj_idx].clear()
                    obtain_n[traj_idx] = self.vectorization_min_obtain
                    iterators[traj_idx] = get_one_exchange_neighborhood(
                        neighbor,
                        self.search_space,
                        self.n_continuous_neighbors,
                        self.stdev_multiplier,
                    )
                elif val == current_acq_values[traj_idx]:
                    plateau_lists[traj_idx].append(neighbor)

            for i in range(n_trajectories):
                if not active[i]:
                    continue

                if not improved[i]:
                    if iterator_exhausted[i]:
                        self._handle_exhaustion(
                            i,
                            active,
                            centers,
                            current_acq_values,
                            plateau_lists,
                            n_no_plateau_walk,
                            iterators,
                            obtain_n,
                        )
                    else:
                        obtain_n[i] = min(obtain_n[i] * 2, self.vectorization_max_obtain)

        searched_hashes = set()
        if hasattr(self.config_manager, "searched_config_hashes"):
            searched_hashes = self.config_manager.searched_config_hashes

        valid_indices = []
        for i, center in enumerate(centers):
            if create_config_hash(center) not in searched_hashes:
                valid_indices.append(i)

        if not valid_indices:
            best_idx = np.argmin(current_acq_values)
            return centers[best_idx]

        best_valid_idx = valid_indices[np.argmin(current_acq_values[valid_indices])]
        return centers[best_valid_idx]

    def _handle_exhaustion(
        self,
        i,
        active,
        centers,
        current_acq_values,
        plateau_lists,
        n_no_plateau_walk,
        iterators,
        obtain_n,
    ):
        if plateau_lists[i] and n_no_plateau_walk[i] < self.n_steps_plateau_walk:
            centers[i] = plateau_lists[i].pop(0)
            n_no_plateau_walk[i] += 1
            obtain_n[i] = self.vectorization_min_obtain
            iterators[i] = get_one_exchange_neighborhood(
                centers[i],
                self.search_space,
                self.n_continuous_neighbors,
                self.stdev_multiplier,
            )
        else:
            active[i] = False
