from abc import ABC, abstractmethod
from typing import Callable, Dict, List

import numpy as np

from ccqr_optimization.utils.tracking import BaseConfigurationManager
from ccqr_optimization.wrapping import ParameterRange


class BaseLocalSearchAlgorithm(ABC):
    """Abstract base for local search algorithms over the acquisition surface.

    Subclasses implement distinct search heuristics but share the same
    ``optimize`` interface so that samplers can use them interchangeably.

    All prediction is performed through an opaque ``predict_fn`` callable rather
    than a live searcher reference, so local search algorithms have no dependency
    on the acquisition or sampler layers above them.
    """

    @abstractmethod
    def optimize(
        self,
        predict_fn: Callable[[List[Dict]], np.ndarray],
        candidates: List[Dict],
        config_manager: BaseConfigurationManager,
        search_space: Dict[str, ParameterRange],
        metric_sign: int,
    ) -> Dict:
        """Run local search and return the best configuration found.

        Args:
            predict_fn: Callable that maps a list of configuration dicts to a
                flat ``np.ndarray`` of acquisition values (lower-is-better).
                Built by the sampler as a closure over its estimators.
            candidates: Random candidate pool. Must be non-empty.
            config_manager: Exposes ``tabularize_configs``, ``searched_configs``,
                and ``searched_performances``.
            search_space: Mapping from parameter name to ``ParameterRange``.
            metric_sign: ``+1`` for minimization, ``-1`` for maximization.

        Returns:
            The configuration with the lowest acquisition value found.
        """
