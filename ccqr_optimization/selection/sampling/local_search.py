import logging
from typing import Dict, List

import numpy as np

from ccqr_optimization.selection.acquisition import BaseConformalSearcher
from ccqr_optimization.utils.configurations.utils import create_config_hash
from ccqr_optimization.wrapping import (
    CategoricalRange,
    FloatRange,
    IntRange,
    ParameterRange,
)

logger = logging.getLogger(__name__)


def normalized_distance(c1: Dict, c2: Dict, search_space: Dict[str, ParameterRange]) -> float:
    """
    Computes a normalized distance between two configurations in [0, 1].
    Used for Non-Maximum Suppression to ensure epicenter diversity.
    """
    dist = 0.0
    for name, prange in search_space.items():
        v1, v2 = c1[name], c2[name]
        if isinstance(prange, CategoricalRange):
            dist += 0.0 if v1 == v2 else 1.0
        elif isinstance(prange, FloatRange):
            if prange.log_scale:
                l1, l2 = np.log(max(v1, 1e-10)), np.log(max(v2, 1e-10))
                lmin, lmax = np.log(max(prange.min_value, 1e-10)), np.log(prange.max_value)
                if lmax > lmin:
                    dist += abs(l1 - l2) / (lmax - lmin)
            else:
                vmin, vmax = prange.min_value, prange.max_value
                if vmax > vmin:
                    dist += abs(v1 - v2) / (vmax - vmin)
        elif isinstance(prange, IntRange):
            if prange.log_scale:
                l1, l2 = np.log(max(v1, 1)), np.log(max(v2, 1))
                lmin, lmax = np.log(max(prange.min_value, 1)), np.log(prange.max_value)
                if lmax > lmin:
                    dist += abs(l1 - l2) / (lmax - lmin)
            else:
                vmin, vmax = prange.min_value, prange.max_value
                if vmax > vmin:
                    dist += abs(v1 - v2) / (vmax - vmin)
    return dist / max(1, len(search_space))


def generate_pattern_neighborhood(
    config: Dict,
    search_space: Dict[str, ParameterRange],
    step_size: float
) -> List[Dict]:
    """
    Generates a Coordinate Search / Pattern Search neighborhood.
    For continuous and integer spaces, generates +/- step_size variations (pseudo-gradients).
    For categorical spaces, generates all 1-exchange variations.
    
    Evaluating this entire neighborhood in one batch effectively computes the steepest 
    ascent/descent direction across the mixed discrete/continuous space, acting as an 
    efficient proxy for partial derivatives in a non-differentiable tree-based landscape.
    """
    neighbors = []
    for param_name, param_range in search_space.items():
        current_val = config[param_name]
        
        if isinstance(param_range, CategoricalRange):
            for choice in param_range.choices:
                if choice != current_val:
                    neighbor = config.copy()
                    neighbor[param_name] = choice
                    neighbors.append(neighbor)
                    
        elif isinstance(param_range, FloatRange):
            if param_range.log_scale:
                lmin = np.log(max(param_range.min_value, 1e-10))
                lmax = np.log(param_range.max_value)
                lval = np.log(max(current_val, 1e-10))
                delta = step_size * (lmax - lmin)
                
                for direction in [-1, 1]:
                    new_lval = lval + direction * delta
                    new_lval = np.clip(new_lval, lmin, lmax)
                    new_val = float(np.exp(new_lval))
                    if new_val != current_val:
                        neighbor = config.copy()
                        neighbor[param_name] = new_val
                        neighbors.append(neighbor)
            else:
                vmin = param_range.min_value
                vmax = param_range.max_value
                delta = step_size * (vmax - vmin)
                
                for direction in [-1, 1]:
                    new_val = current_val + direction * delta
                    new_val = float(np.clip(new_val, vmin, vmax))
                    if new_val != current_val:
                        neighbor = config.copy()
                        neighbor[param_name] = new_val
                        neighbors.append(neighbor)
                        
        elif isinstance(param_range, IntRange):
            if param_range.log_scale:
                lmin = np.log(max(param_range.min_value, 1))
                lmax = np.log(param_range.max_value)
                lval = np.log(max(current_val, 1))
                delta = step_size * (lmax - lmin)
                
                for direction in [-1, 1]:
                    new_lval = lval + direction * delta
                    new_lval = np.clip(new_lval, lmin, lmax)
                    new_val = int(np.round(np.exp(new_lval)))
                    if new_val == current_val:
                        new_val = current_val + direction
                        new_val = int(np.clip(new_val, param_range.min_value, param_range.max_value))
                    if new_val != current_val:
                        neighbor = config.copy()
                        neighbor[param_name] = new_val
                        neighbors.append(neighbor)
            else:
                vmin = param_range.min_value
                vmax = param_range.max_value
                delta = step_size * (vmax - vmin)
                
                for direction in [-1, 1]:
                    step = int(np.round(delta))
                    if step == 0:
                        step = 1
                    new_val = current_val + direction * step
                    new_val = int(np.clip(new_val, vmin, vmax))
                    if new_val != current_val:
                        neighbor = config.copy()
                        neighbor[param_name] = new_val
                        neighbors.append(neighbor)
    return neighbors


class LocalSearchTrajectory:
    def __init__(self, initial_config: Dict, initial_score: float, initial_step_size: float):
        self.current_config = initial_config
        self.current_score = initial_score
        self.step_size = initial_step_size
        self.steps_since_improvement = 0
        self.active = True


class LocalSearchOptimizer:
    """
    Optimizes the acquisition function using an Adaptive Pattern Search (Coordinate Search)
    with Non-Maximum Suppression for diverse epicenter spawning.
    
    Academic Justification:
    Tree-based quantile regression models produce piece-wise constant surfaces, meaning
    finite-difference gradients are zero almost everywhere. Gradient-based methods like LBFGS 
    are highly prone to getting stuck in flat regions. Instead, this optimizer uses an 
    Adaptive Pattern Search with a large initial step size to bridge decision boundaries. 
    The step size shrinks adaptively only when local improvements cannot be found, allowing 
    convergence to a robust local minimum without the need for differentiable surrogate functions.
    
    Furthermore, to satisfy runtime bounds, it tightly controls the prediction budget, and evaluates
    entire pattern neighborhoods simultaneously leveraging the vectorized prediction efficiency
    of scikit-learn estimators.
    """

    def __init__(
        self,
        search_space: Dict[str, ParameterRange],
        config_manager,
        max_parallel_trajectories: int = 16,
        initial_step_size: float = 0.2,
        min_step_size: float = 1e-3,
        step_shrink_factor: float = 0.5,
        step_growth_factor: float = 1.0,
        patience: int = 5,
        max_predict_budget: int = 15000,
        min_epicenter_distance: float = 0.05,
    ):
        self.search_space = search_space
        self.config_manager = config_manager
        self.max_parallel_trajectories = max_parallel_trajectories
        self.initial_step_size = initial_step_size
        self.min_step_size = min_step_size
        self.step_shrink_factor = step_shrink_factor
        self.step_growth_factor = step_growth_factor
        self.patience = patience
        self.max_predict_budget = max_predict_budget
        self.min_epicenter_distance = min_epicenter_distance

    def _nms_filter_epicenters(
        self, candidate_configs: List[Dict], candidate_scores: np.ndarray
    ) -> List[Dict]:
        """
        Filters candidates to ensure diverse epicenters, prioritizing better scores.
        Avoids clustering of starting paths by enforcing a minimum distance constraint.
        """
        sorted_indices = np.argsort(candidate_scores)
        selected_configs = []
        
        for idx in sorted_indices:
            candidate = candidate_configs[idx]
            
            too_close = False
            for selected in selected_configs:
                if normalized_distance(candidate, selected, self.search_space) < self.min_epicenter_distance:
                    too_close = True
                    break
                    
            if not too_close:
                selected_configs.append(candidate)
                
        return selected_configs

    def maximize(
        self,
        searcher: BaseConformalSearcher,
        candidate_configs: List[Dict],
        candidate_scores: np.ndarray,
    ) -> Dict:
        """
        Run parallel adaptive pattern searches from diverse epicenters.
        Returns the best configuration found.
        """
        if not candidate_configs:
            raise ValueError("Local search requires at least one candidate configuration.")

        # 1. Filter diverse epicenters using NMS
        diverse_epicenters = self._nms_filter_epicenters(candidate_configs, candidate_scores)
        
        # We will keep a queue of epicenters to spawn new trajectories when old ones die
        epicenter_queue = list(diverse_epicenters)
        
        # 2. Initialize tracking
        evaluated_hashes = set()
        if hasattr(self.config_manager, "searched_config_hashes"):
            evaluated_hashes.update(self.config_manager.searched_config_hashes)
        if hasattr(self.config_manager, "banned_configurations"):
            evaluated_hashes.update(
                create_config_hash(c) for c in self.config_manager.banned_configurations
            )
            
        best_overall_config = None
        best_overall_score = float('inf')
        
        trajectories: List[LocalSearchTrajectory] = []
        
        predicts_used = 0
        
        # Initial spawn
        while len(trajectories) < self.max_parallel_trajectories and epicenter_queue:
            config = epicenter_queue.pop(0)
            chash = create_config_hash(config)
            
            X_center = self.config_manager.tabularize_configs([config])
            score = float(searcher.predict(X_center)[0])
            
            if score < best_overall_score:
                best_overall_score = score
                best_overall_config = config
                
            evaluated_hashes.add(chash)
            trajectories.append(LocalSearchTrajectory(config, score, self.initial_step_size))
            predicts_used += 1

        # 3. Main Search Loop
        while trajectories and predicts_used < self.max_predict_budget:
            active_trajectories = [t for t in trajectories if t.active]
            if not active_trajectories:
                # Try to spawn new ones if all are dead
                while len(trajectories) < self.max_parallel_trajectories and epicenter_queue:
                    config = epicenter_queue.pop(0)
                    chash = create_config_hash(config)
                    if chash not in evaluated_hashes:
                        X_center = self.config_manager.tabularize_configs([config])
                        score = float(searcher.predict(X_center)[0])
                        predicts_used += 1
                        if score < best_overall_score:
                            best_overall_score = score
                            best_overall_config = config
                        evaluated_hashes.add(chash)
                        trajectories.append(LocalSearchTrajectory(config, score, self.initial_step_size))
                        active_trajectories.append(trajectories[-1])
                
                if not active_trajectories:
                    break  # Completely out of epicenters and all dead
            
            # Generate batch of neighbors across all active trajectories
            batch_configs = []
            batch_traj_indices = []
            
            for i, traj in enumerate(active_trajectories):
                neighbors = generate_pattern_neighborhood(
                    traj.current_config, self.search_space, traj.step_size
                )
                for neighbor in neighbors:
                    nhash = create_config_hash(neighbor)
                    if nhash not in evaluated_hashes:
                        evaluated_hashes.add(nhash)
                        batch_configs.append(neighbor)
                        batch_traj_indices.append(i)
                        
            if not batch_configs:
                # All active trajectories generated neighbors that were already evaluated
                # or no neighbors could be generated. Mark them for step size reduction.
                for traj in active_trajectories:
                    traj.steps_since_improvement += 1
                    traj.step_size *= self.step_shrink_factor
                    if traj.step_size < self.min_step_size or traj.steps_since_improvement >= self.patience:
                        traj.active = False
                continue
                
            # Evaluate batch
            if predicts_used + len(batch_configs) > self.max_predict_budget:
                # Truncate batch to respect budget
                allowed = self.max_predict_budget - predicts_used
                batch_configs = batch_configs[:allowed]
                batch_traj_indices = batch_traj_indices[:allowed]
                
            if not batch_configs:
                break
                
            X_batch = self.config_manager.tabularize_configs(batch_configs)
            acq_values = searcher.predict(X_batch)
            predicts_used += len(batch_configs)
            
            # Group results by trajectory
            traj_best_neighbor = [None] * len(active_trajectories)
            traj_best_score = [float('inf')] * len(active_trajectories)
            
            for idx, t_idx in enumerate(batch_traj_indices):
                val = acq_values[idx]
                if val < traj_best_score[t_idx]:
                    traj_best_score[t_idx] = val
                    traj_best_neighbor[t_idx] = batch_configs[idx]
                    
            # Update trajectories
            for i, traj in enumerate(active_trajectories):
                best_neighbor = traj_best_neighbor[i]
                best_score = traj_best_score[i]
                
                if best_neighbor is not None and best_score < traj.current_score:
                    # Improvement found (steepest descent direction)
                    traj.current_config = best_neighbor
                    traj.current_score = best_score
                    traj.steps_since_improvement = 0
                    traj.step_size = min(1.0, traj.step_size * self.step_growth_factor)
                    
                    if best_score < best_overall_score:
                        best_overall_score = best_score
                        best_overall_config = best_neighbor
                else:
                    # No improvement, shrink step size to localize search
                    traj.steps_since_improvement += 1
                    traj.step_size *= self.step_shrink_factor
                    
                # Prune if stuck or converged
                if traj.step_size < self.min_step_size or traj.steps_since_improvement >= self.patience:
                    traj.active = False
                    
            # Cull trajectories that converge to the same configuration to prevent duplicated work
            active_hashes = {}
            for traj in active_trajectories:
                if traj.active:
                    chash = create_config_hash(traj.current_config)
                    if chash in active_hashes:
                        traj.active = False
                    else:
                        active_hashes[chash] = traj

            # Spawn new trajectories to replace dead ones from the diversified pool
            dead_count = sum(1 for t in trajectories if not t.active)
            active_count = len(trajectories) - dead_count
            
            while active_count < self.max_parallel_trajectories and epicenter_queue:
                config = epicenter_queue.pop(0)
                chash = create_config_hash(config)
                if chash not in evaluated_hashes:
                    X_center = self.config_manager.tabularize_configs([config])
                    score = float(searcher.predict(X_center)[0])
                    predicts_used += 1
                    if score < best_overall_score:
                        best_overall_score = score
                        best_overall_config = config
                    evaluated_hashes.add(chash)
                    trajectories.append(LocalSearchTrajectory(config, score, self.initial_step_size))
                    active_count += 1
                    
        return best_overall_config if best_overall_config is not None else candidate_configs[0]
