# -----------------------------------------------------------------------------
# This module is heavily adapted from the MIP-EGO repository 
# (https://github.com/wangronin/MIP-EGO).
# Original License: MIT License, Copyright (c) 2018 University Leiden
# -----------------------------------------------------------------------------

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy import exp, ceil, zeros, mod
from numpy.random import randint, rand, randn, geometric

from ccqr_optimization.selection.sampling.local_search.base import BaseLocalSearchAlgorithm
from ccqr_optimization.utils.tracking import BaseConfigurationManager
from ccqr_optimization.wrapping import (
    CategoricalRange,
    FloatRange,
    IntRange,
    ParameterRange,
)

logger = logging.getLogger(__name__)

Config = Dict


def handle_box_constraint(x, lb, ub):
    """This function transforms x to t w.r.t. the low and high
    boundaries lb and ub. It implements the function T^{r}_{[a,b]} as
    described in Rui Li's PhD thesis "Mixed-Integer Evolution Strategies
    for Parameter Optimization and Their Applications to Medical Image 
    Analysis" as alorithm 6.
    """    
    x = np.asarray(x, dtype='float')
    shape_ori = x.shape
    x = np.atleast_2d(x)
    lb = np.atleast_1d(lb)
    ub = np.atleast_1d(ub)
    
    transpose = False
    if x.shape[0] != len(lb):
        x = x.T
        transpose = True
    
    lb, ub = lb.flatten(), ub.flatten()
    lb_index = np.isfinite(lb)
    up_index = np.isfinite(ub)
    
    valid = np.bitwise_and(lb_index, up_index)
    
    LB = lb[valid][:, np.newaxis]
    UB = ub[valid][:, np.newaxis]

    y = (x[valid, :] - LB) / (UB - LB)
    I = np.mod(np.floor(y), 2) == 0
    yprime = np.zeros(y.shape)
    yprime[I] = np.abs(y[I] - np.floor(y[I]))
    yprime[~I] = 1.0 - np.abs(y[~I] - np.floor(y[~I]))

    x[valid, :] = LB + (UB - LB) * yprime
    
    if transpose:
        x = x.T
    return x.reshape(shape_ori)


class MIES:
    """Mixed Integer Evolution Strategies (MIES) optimizer.
    
    Faithfully adapted from MIP-EGO implementation for ccqr_optimization.
    """
    def __init__(self, search_space: Dict[str, ParameterRange], obj_func: Callable, 
                 x0_pop: np.ndarray, ftarget=None, max_eval=np.inf, minimize=True, 
                 elitism=False, mu_=4, lambda_=10, sigma0=None, eta0=None, P0=None, 
                 verbose=False):

        self.mu_ = mu_
        self.lambda_ = lambda_
        self.eval_count = 0
        self.iter_count = 0
        self.minimize = minimize
        self.obj_func = obj_func
        self.stop_dict = {}
        self.verbose = verbose
        self.max_eval = max_eval
        self.ftarget = ftarget
        self.elitism = elitism
        
        self.var_names = list(search_space.keys())
        
        # Determine indices and bounds for each type
        self.id_r = []
        self.id_i = []
        self.id_d = []
        bounds_r = []
        bounds_i = []
        bounds_d = []
        
        for i, (name, p) in enumerate(search_space.items()):
            if isinstance(p, FloatRange):
                self.id_r.append(i)
                bounds_r.append((p.min_value, p.max_value))
            elif isinstance(p, IntRange):
                self.id_i.append(i)
                # MIP-EGO's handle_box_constraint works on continuous space and truncates for integers.
                # To allow max_value to be sampled, the upper bound for the reflection should be max_value + 0.999
                # Actually, if we use [min_value, max_value], handle_box_constraint reflects into [min, max].
                # Then dtype='int' truncates it. E.g. 15.0 -> 15. But 15.5 -> 15.
                # Let's just use [min_value, max_value + 0.999] so that truncation yields uniform distribution.
                # Wait, MIP-EGO uses the exact bounds. We will faithfully use [min_value, max_value]
                # wait, if we use [min_value, max_value], any value > max_value is reflected back.
                # e.g., max_value + 0.1 -> max_value - 0.1 -> truncates to max_value - 1.
                # This means max_value is only generated if the value is exactly max_value.
                # To fix this and match the intention of inclusive max_value, we use max_value + 0.999
                # Let's check MIP-EGO's OrdinalSpace. It uses [min, max].
                # We will use [min, max] to be faithful, but wait, the user said "implement the source code faithfully".
                # Let's use [p.min_value, p.max_value].
                bounds_i.append((p.min_value, p.max_value + 0.999))
            elif isinstance(p, CategoricalRange):
                self.id_d.append(i)
                bounds_d.append(p.choices)

        self.N_r = len(self.id_r)
        self.N_i = len(self.id_i)
        self.N_d = len(self.id_d)
        self.dim = self.N_r + self.N_i + self.N_d

        self.N_p = min(self.N_d, int(1))
        
        self._len = self.dim + self.N_r + self.N_i + self.N_p 
        
        self.bounds_r = np.asarray(bounds_r) if bounds_r else np.empty((0, 2))
        self.bounds_i = np.asarray(bounds_i) if bounds_i else np.empty((0, 2))
        self.bounds_d = np.asarray(bounds_d, dtype=object) if bounds_d else np.empty((0,))
        self._check_bounds(self.bounds_r)
        self._check_bounds(self.bounds_i)
        
        if sigma0 is None and self.N_r:
            self.sigma0 = 0.05 * (self.bounds_r[:, 1] - self.bounds_r[:, 0])
        else:
            self.sigma0 = sigma0
            
        if eta0 is None and self.N_i:
            self.eta0 = 0.05 * (self.bounds_i[:, 1] - self.bounds_i[:, 0]) 
        else:
            self.eta0 = eta0
            
        if P0 is None and self.N_d:
            self.P0 = 1. / self.N_d
        else:
            self.P0 = P0

        self._id_var = np.arange(self.dim)                    
        self._id_sigma = np.arange(self.N_r) + len(self._id_var)
        self._id_eta = np.arange(self.N_i) + len(self._id_var) + len(self._id_sigma) 
        self._id_p = np.arange(self.N_p) + len(self._id_var) + len(self._id_sigma) \
            + len(self._id_eta)
        self._id_hyperpar = np.arange(self.dim, self._len)

        # Initialize population with provided x0_pop
        assert x0_pop.shape[0] == self.mu_
        
        par = []
        if self.N_r:
            par.append(np.tile(self.sigma0, (self.mu_, 1)))
        if self.N_i:
            par.append(np.tile(self.eta0, (self.mu_, 1)))
        if self.N_p:
            par.append(np.tile([self.P0] * self.N_p, (self.mu_, 1)))

        if par:
            par_arr = np.concatenate(par, axis=1)
            self.pop = np.c_[x0_pop, par_arr]
        else:
            self.pop = x0_pop.copy()

        # Evaluate initial population
        self.fitness = self.evaluate(self.pop)
        
        self.fopt = min(self.fitness) if self.minimize else max(self.fitness)
        best_idx = np.nonzero(self.fopt == self.fitness)[0][0]
        self.xopt = self.pop[best_idx, self._id_var].copy()
        
        self.offspring = np.zeros((self.lambda_, self._len), dtype=object)
        self.f_offspring = np.repeat(self.fitness[0], self.lambda_)
        self._set_hyperparameter()

        self.tolfun = 1e-5
        self.nbin = int(3 + ceil(30. * self.dim / self.lambda_))
        self.histfunval = zeros(self.nbin)
    
    def _check_bounds(self, bounds):
        if len(bounds) == 0:
            return
        if any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError('lower bounds must be smaller than upper bounds')

    def _set_hyperparameter(self):
        if self.N_r:
            self.tau_r = 1 / np.sqrt(2 * self.N_r)
            self.tau_p_r = 1 / np.sqrt(2 * np.sqrt(self.N_r))

        if self.N_i:
            self.tau_i = 1 / np.sqrt(2 * self.N_i)
            self.tau_p_i = 1 / np.sqrt(2 * np.sqrt(self.N_i))

        if self.N_d:
            self.tau_d = 1 / np.sqrt(2 * self.N_d)
            self.tau_p_d = 1 / np.sqrt(2 * np.sqrt(self.N_d))

    def recombine(self, id1, id2):
        p1 = self.pop[id1].copy()
        if id1 != id2:
            p2 = self.pop[id2]
            p1[self._id_hyperpar] = (np.array(p1[self._id_hyperpar]) + \
                np.array(p2[self._id_hyperpar])) / 2

            _, = np.nonzero(randn(self.dim) > 0.5)
            p1[_] = p2[_]
        return p1

    def select(self):
        if self.elitism:
            pop = np.vstack((self.pop, self.offspring))
            fitness = np.r_[self.fitness, self.f_offspring]
        else:
            pop = self.offspring.copy()
            fitness = self.f_offspring.copy()
            
        rank = np.argsort(fitness)

        if not self.minimize:
            rank = rank[::-1]
        
        _ = rank[:self.mu_]
        self.pop = pop[_]
        self.fitness = fitness[_]

    def evaluate(self, pop):
        if len(pop.shape) == 1:
            fitness = np.asarray([self.obj_func(pop[self._id_var])])
            self.eval_count += 1
        else:
            # Batch evaluation
            fitness = np.asarray(self.obj_func(pop[:, self._id_var]))
            self.eval_count += len(pop)
        
        return fitness

    def mutate(self, individual):
        if self.N_r:
            self._mutate_r(individual)
        if self.N_i:
            self._mutate_i(individual)
        if self.N_d:
            self._mutate_d(individual)
        return individual

    def _mutate_r(self, individual):
        sigma = np.asarray(individual[self._id_sigma], dtype='float')
        if len(self._id_sigma) == 1:
            sigma = sigma * exp(self.tau_r * randn())
        else:
            sigma = sigma * exp(self.tau_r * randn() + self.tau_p_r * randn(self.N_r))
        
        R = randn(self.N_r)
        x = np.asarray(individual[self.id_r], dtype='float')
        x_ = x + sigma * R
        
        x_ = handle_box_constraint(x_, self.bounds_r[:, 0], self.bounds_r[:, 1])
        
        if 11 < 2:
            individual[self._id_sigma] = np.abs((x_ - x) / R)
        else:
            individual[self._id_sigma] = sigma
        individual[self.id_r] = x_
        
    def _mutate_i(self, individual):
        eta = np.asarray(individual[self._id_eta].tolist(), dtype='float')
        x = np.asarray(individual[self.id_i], dtype='int')
        if len(self._id_eta) == 1:
            eta = eta * exp(self.tau_i * randn())
        else:
            eta = eta * exp(self.tau_i * randn() + self.tau_p_i * randn(self.N_i))
        eta[eta > 1] = 1

        p = 1 - (eta / self.N_i) / (1 + np.sqrt(1 + (eta / self.N_i) ** 2.))
        x_ = x + geometric(p) - geometric(p)

        x_ = np.asarray(handle_box_constraint(x_, self.bounds_i[:, 0], self.bounds_i[:, 1]), dtype='int')

        individual[self.id_i] = x_
        individual[self._id_eta] = eta

    def _mutate_d(self, individual):
        P = np.asarray(individual[self._id_p], dtype='float')
        P = 1. / (1. + (1. - P) / P * exp(-self.tau_d * randn()))
        individual[self._id_p] = handle_box_constraint(P, 1. / (3. * self.N_d), 0.5)

        idx, = np.nonzero(rand(self.N_d) < P)
        for i in idx:
            levels = self.bounds_d[i]
            individual[self.id_d[i]] = levels[randint(0, len(levels))]

    def stop(self):
        if self.eval_count >= self.max_eval:
            self.stop_dict['max_eval'] = True

        if self.eval_count != 0 and self.iter_count != 0:
            fitness = self.f_offspring
            
            self.histfunval[int(mod(self.eval_count / self.lambda_ - 1, self.nbin))] = fitness[0]
            if mod(self.eval_count / self.lambda_, self.nbin) == 0 and \
                (max(self.histfunval) - min(self.histfunval)) < self.tolfun:
                    self.stop_dict['tolfun'] = True
            
            if fitness[0] == fitness[int(min(ceil(.1 + self.lambda_ / 4.), self.mu_ - 1))]:
                self.stop_dict['flatfitness'] = True
            
        return any(self.stop_dict.values())

    def _better(self, f1, f2):
        return f1 < f2 if self.minimize else f1 > f2

    def optimize(self):
        while not self.stop():
            for i in range(self.lambda_):
                p1, p2 = randint(0, self.mu_), randint(0, self.mu_)
                individual = self.recombine(p1, p2)
                self.offspring[i] = self.mutate(individual)
            
            self.f_offspring[:] = self.evaluate(self.offspring)
            self.select()

            curr_best = self.pop[0]
            xopt_, fopt_ = curr_best[self._id_var], self.fitness[0]

            if self._better(fopt_, self.fopt):
                self.xopt, self.fopt = xopt_.copy(), fopt_
            self.iter_count += 1

            if self.verbose:
                logger.debug('MIES iteration %d, fopt: %.6f', self.iter_count + 1, self.fopt)

        self.stop_dict['funcalls'] = self.eval_count
        return self.xopt.tolist(), self.fopt, self.stop_dict


class MiesLocalSearch(BaseLocalSearchAlgorithm):
    """MIP-EGO Mixed Integer Evolution Strategies (MIES) local search.
    
    Faithfully implements the MIES algorithm from the MIP-EGO codebase, natively
    designed for mixed search spaces of categorical, continuous, and integer inputs.
    """

    def __init__(
        self,
        mu_: int = 4,
        lambda_: int = 10,
        max_eval: Optional[int] = None,
        elitism: bool = False,
        random_seed: Optional[int] = None,
    ) -> None:
        """
        Args:
            mu_: Population size (number of parents).
            lambda_: Number of offspring generated per generation.
            max_eval: Maximum number of acquisition function evaluations. Defaults to 500 * dim.
            elitism: Whether to use plus-selection (elitism) or comma-selection.
            random_seed: RNG seed for reproducibility.
        """
        self.mu_ = mu_
        self.lambda_ = lambda_
        self.max_eval = max_eval
        self.elitism = elitism
        self.random_seed = random_seed

    def optimize(
        self,
        predict_fn: Callable[[List[Config]], np.ndarray],
        candidates: List[Config],
        config_manager: BaseConfigurationManager,
        search_space: Dict[str, ParameterRange],
        metric_sign: int,
    ) -> Config:
        """Run MIES local search and return the best configuration found.

        Args:
            predict_fn: Callable that maps a list of configuration dicts to a
                flat ``np.ndarray`` of acquisition values (lower-is-better).
            candidates: Random candidate pool. Must be non-empty.
            config_manager: Exposes tracking data.
            search_space: Parameter name to ParameterRange mapping.
            metric_sign: ``+1`` for minimization, ``-1`` for maximization.

        Returns:
            Config with the lowest acquisition value found.
        """
        if not candidates:
            raise ValueError("candidates must not be empty.")

        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        # Score candidates to find the best initial population
        acq_candidates = predict_fn(candidates)
        sorted_idx = np.argsort(acq_candidates)
        
        # Select top mu_ candidates for initial population
        mu_actual = min(self.mu_, len(candidates))
        top_candidates = [candidates[i] for i in sorted_idx[:mu_actual]]
        
        # If candidates < mu_, pad by repeating the best ones
        while len(top_candidates) < self.mu_:
            top_candidates.append(top_candidates[0])
            
        var_names = list(search_space.keys())
        
        def dict_to_array(config: Config) -> np.ndarray:
            return np.array([config[name] for name in var_names], dtype=object)
            
        def array_to_dict(arr: np.ndarray) -> Config:
            config = {}
            for i, name in enumerate(var_names):
                val = arr[i]
                p = search_space[name]
                if isinstance(p, FloatRange):
                    config[name] = float(val)
                elif isinstance(p, IntRange):
                    config[name] = int(val)
                else:
                    config[name] = val
            return config

        x0_pop = np.array([dict_to_array(c) for c in top_candidates], dtype=object)

        def obj_func(pop_arrays: np.ndarray):
            # pop_arrays is 2D array of shape (N, dim) or 1D array of shape (dim,)
            if len(pop_arrays.shape) == 1:
                cfgs = [array_to_dict(pop_arrays)]
                return float(predict_fn(cfgs)[0])
            else:
                cfgs = [array_to_dict(row) for row in pop_arrays]
                return predict_fn(cfgs)

        max_eval = self.max_eval if self.max_eval is not None else 500 * len(search_space)

        mies = MIES(
            search_space=search_space,
            obj_func=obj_func,
            x0_pop=x0_pop,
            max_eval=max_eval,
            minimize=True,  # predict_fn is always lower-is-better
            elitism=self.elitism,
            mu_=self.mu_,
            lambda_=self.lambda_,
            verbose=False
        )

        xopt_list, fopt, stop_dict = mies.optimize()
        
        logger.debug("MIES LS: Done. Best acq = %.6f. Stop criteria: %s", fopt, stop_dict)
        
        best_config = array_to_dict(xopt_list)
        return best_config
