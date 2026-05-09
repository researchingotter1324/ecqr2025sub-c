"""Abstract base class shared by all local search algorithms.

Local search modules import ``BaseConfigurationManager`` from ``utils.tracking``.
Nothing in this module or its submodules imports from ``selection.acquisition``,
so the dependency graph remains acyclic:

    wrapping / utils  ──►  local_search / samplers
                                   │
                                   ▼
                              acquisition

Convention: acquisition values are lower-is-better throughout.
"""

from abc import ABC, abstractmethod
from typing import Dict, List

from ccqr_optimization.utils.tracking import BaseConfigurationManager
from ccqr_optimization.wrapping import ParameterRange


class BaseLocalSearchAlgorithm(ABC):
    """Abstract base for local search algorithms over the acquisition surface.

    Subclasses implement distinct search heuristics but share the same
    ``optimize`` interface so that samplers can use them interchangeably without
    knowing the concrete type. All mutable runtime context (searcher,
    candidates, config_manager, search_space) is passed at call time so a
    single algorithm instance can be reused across tuning iterations without
    carrying stale state.
    """

    @abstractmethod
    def optimize(
        self,
        searcher,
        candidates: List[Dict],
        config_manager: BaseConfigurationManager,
        search_space: Dict[str, ParameterRange],
        metric_sign: int,
    ) -> Dict:
        """Run local search and return the best configuration found.

        Args:
            searcher: Fitted conformal searcher. ``predict(X)`` returns
                acquisition values (lower-is-better) for a tabularized feature
                matrix of shape ``(n_candidates, n_features)``.
            candidates: Random candidate pool. Must be non-empty.
            config_manager: Configuration manager exposing ``tabularize_configs``,
                ``searched_configs``, and ``searched_performances``.
            search_space: Mapping from parameter name to ``ParameterRange``.
            metric_sign: ``+1`` for minimization, ``-1`` for maximization.

        Returns:
            The configuration with the lowest acquisition value found.
        """
