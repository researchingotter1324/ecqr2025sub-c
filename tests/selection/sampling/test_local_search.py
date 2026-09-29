import numpy as np
import pytest

from ccqr_optimization.selection.sampling.local_search.mies_search import MiesLocalSearch
from ccqr_optimization.selection.sampling.local_search.smac_search import SmacLocalSearch
from ccqr_optimization.utils.configurations.utils import create_config_hash
from ccqr_optimization.utils.tracking import DynamicConfigurationManager
from ccqr_optimization.wrapping import IntRange


@pytest.mark.parametrize("max_eval", [0, 5, 20])
def test_smac_pool_is_scored_once_and_additional_evals_respect_cap(
    local_search_space, max_eval
):
    mgr = DynamicConfigurationManager(
        search_space=local_search_space,
        n_candidate_configurations=24,
        random_state=0,
    )
    candidates = mgr.get_searchable_configurations()
    call_sizes = []

    def predict_fn(cfgs):
        call_sizes.append(len(cfgs))
        return np.array([cfg["x"] + 0.01 * cfg["k"] for cfg in cfgs])

    search = SmacLocalSearch(
        n_acq_starts=3,
        n_historical_starts=2,
        n_steps_plateau_walk=4,
        max_eval=max_eval,
        vectorization_max_obtain=64,
        random_state=0,
    )
    chosen, acq = search.optimize(
        predict_fn=predict_fn,
        candidates=candidates,
        config_manager=mgr,
        search_space=local_search_space,
        metric_sign=1,
    )

    assert call_sizes[0] == len(candidates)
    assert sum(call_sizes[1:]) <= max_eval
    assert create_config_hash(chosen) not in mgr.searched_config_hashes
    assert acq == pytest.approx(chosen["x"] + 0.01 * chosen["k"])


def test_smac_fills_remaining_eval_budget_when_walks_finish_early(local_search_space):
    mgr = DynamicConfigurationManager(
        search_space=local_search_space,
        n_candidate_configurations=24,
        random_state=0,
    )
    candidates = mgr.get_searchable_configurations()
    call_sizes = []

    def predict_fn(cfgs):
        call_sizes.append(len(cfgs))
        return np.array([cfg["x"] + 0.01 * cfg["k"] for cfg in cfgs])

    max_eval = 20
    search = SmacLocalSearch(
        n_acq_starts=1,
        n_historical_starts=0,
        n_steps_plateau_walk=1,
        max_eval=max_eval,
        num_continuous_neighbors=2,
        random_state=0,
    )
    search.optimize(
        predict_fn=predict_fn,
        candidates=candidates,
        config_manager=mgr,
        search_space=local_search_space,
        metric_sign=1,
    )

    assert call_sizes[0] == len(candidates)
    additional = sum(call_sizes[1:])
    assert additional <= max_eval
    assert additional >= max_eval - 2
    assert max(call_sizes[1:]) > 5


def test_smac_never_returns_a_historically_sampled_config(local_search_space):
    mgr = DynamicConfigurationManager(
        search_space=local_search_space,
        n_candidate_configurations=24,
        random_state=1,
    )
    incumbent = {"x": 0.0, "k": 0, "c": "a"}
    mgr.mark_as_searched(incumbent, -100.0)
    candidates = mgr.get_searchable_configurations()

    def predict_fn(cfgs):
        return np.array([cfg["x"] for cfg in cfgs])

    search = SmacLocalSearch(
        n_acq_starts=4,
        n_historical_starts=6,
        n_steps_plateau_walk=5,
        max_eval=40,
        random_state=1,
    )
    chosen, _ = search.optimize(
        predict_fn=predict_fn,
        candidates=candidates,
        config_manager=mgr,
        search_space=local_search_space,
        metric_sign=1,
    )

    assert create_config_hash(chosen) != create_config_hash(incumbent)
    assert create_config_hash(chosen) not in mgr.searched_config_hashes


def test_smac_filters_discrete_neighbors_that_collide_with_history():
    space = {"k": IntRange(min_value=0, max_value=2)}
    mgr = DynamicConfigurationManager(
        search_space=space,
        n_candidate_configurations=8,
        random_state=0,
    )
    mgr.mark_as_searched({"k": 0}, 0.0)
    mgr.mark_as_searched({"k": 1}, 1.0)
    candidates = [{"k": 2}]

    def predict_fn(cfgs):
        return np.array([float(cfg["k"]) for cfg in cfgs])

    search = SmacLocalSearch(
        n_acq_starts=1,
        n_historical_starts=2,
        n_steps_plateau_walk=6,
        max_eval=20,
        num_continuous_neighbors=4,
        random_state=0,
    )
    chosen, _ = search.optimize(
        predict_fn=predict_fn,
        candidates=candidates,
        config_manager=mgr,
        search_space=space,
        metric_sign=1,
    )

    assert chosen == {"k": 2}


@pytest.mark.parametrize("max_eval", [0, 8, 25])
def test_mies_pool_is_scored_once_and_additional_evals_respect_cap(
    local_search_space, max_eval
):
    mgr = DynamicConfigurationManager(
        search_space=local_search_space,
        n_candidate_configurations=16,
        random_state=0,
    )
    candidates = mgr.get_searchable_configurations()
    call_sizes = []

    def predict_fn(cfgs):
        call_sizes.append(len(cfgs))
        return np.array([cfg["x"] + 0.01 * cfg["k"] for cfg in cfgs])

    search = MiesLocalSearch(
        mu_=3,
        lambda_=4,
        max_eval=max_eval,
        random_state=0,
    )
    chosen, _ = search.optimize(
        predict_fn=predict_fn,
        candidates=candidates,
        config_manager=mgr,
        search_space=local_search_space,
        metric_sign=1,
    )

    assert call_sizes[0] == len(candidates)
    assert sum(call_sizes[1:]) <= max_eval
    assert create_config_hash(chosen) not in mgr.searched_config_hashes


def test_mies_never_returns_a_historically_sampled_config(local_search_space):
    mgr = DynamicConfigurationManager(
        search_space=local_search_space,
        n_candidate_configurations=16,
        random_state=2,
    )
    incumbent = {"x": 0.0, "k": 0, "c": "a"}
    mgr.mark_as_searched(incumbent, -100.0)
    candidates = mgr.get_searchable_configurations()

    def predict_fn(cfgs):
        return np.array([cfg["x"] for cfg in cfgs])

    search = MiesLocalSearch(mu_=3, lambda_=4, max_eval=30, random_state=2)
    chosen, _ = search.optimize(
        predict_fn=predict_fn,
        candidates=candidates,
        config_manager=mgr,
        search_space=local_search_space,
        metric_sign=1,
    )

    assert create_config_hash(chosen) != create_config_hash(incumbent)
    assert create_config_hash(chosen) not in mgr.searched_config_hashes
