"""Quantile regression estimators for distributional prediction.

This module provides quantile regression implementations using different algorithmic
approaches: multi-fit estimators that train separate models per quantile, and single-fit
estimators that model the full conditional distribution. Includes gradient boosting,
random forest, neural network, and Gaussian process variants optimized for uncertainty
quantification in conformal prediction frameworks.
"""

from typing import Dict, List, Union, Optional
import numpy as np
from ccqr_optimization.utils.math import monotone_rearrange
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.ensemble._forest import _generate_sample_indices, _get_n_samples_bootstrap
from sklearn.neighbors import NearestNeighbors
from statsmodels.regression.quantile_regression import QuantReg
from sklearn.base import clone
from abc import ABC, abstractmethod
from scipy.stats import norm
from scipy.linalg import solve_triangular, cholesky, LinAlgError
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import QuantileRegressor as SKLearnQuantileRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process.kernels import (
    RBF,
    Matern,
    RationalQuadratic,
    ExpSineSquared,
    ConstantKernel as C,
    Kernel,
    WhiteKernel,
)
from sklearn.utils.validation import check_array
import warnings
import copy
import logging


class BaseMultiFitQuantileEstimator(ABC):
    """Abstract base for quantile estimators training separate models per quantile."""

    def fit(self, X: np.array, y: np.array, quantiles: List[float]):
        """Fit separate models for each quantile level.

        Args:
            X: Training features with shape (n_samples, n_features).
            y: Training targets with shape (n_samples,).
            quantiles: List of quantile levels in [0, 1] to fit models for.

        Returns:
            Self for method chaining.
        """
        self.quantiles = quantiles
        self.trained_estimators = []
        for quantile in quantiles:
            quantile_estimator = self._fit_quantile_estimator(X, y, quantile)
            self.trained_estimators.append(quantile_estimator)
        return self

    @abstractmethod
    def _fit_quantile_estimator(self, X: np.array, y: np.array, quantile: float):
        """Fit a single model for the specified quantile level.

        Args:
            X: Training features with shape (n_samples, n_features).
            y: Training targets with shape (n_samples,).
            quantile: Quantile level in [0, 1] to fit model for.

        Returns:
            Fitted estimator for the quantile level.
        """

    def predict(self, X: np.array) -> np.array:
        """Generate predictions for all fitted quantile levels.

        Args:
            X: Features for prediction with shape (n_samples, n_features).

        Returns:
            Quantile predictions with shape (n_samples, n_quantiles).

        Raises:
            RuntimeError: If called before fitting any models.
        """
        if not self.trained_estimators:
            raise RuntimeError("Model must be fitted before prediction")

        y_pred = np.column_stack(
            [estimator.predict(X) for estimator in self.trained_estimators]
        )
        return monotone_rearrange(y_pred, self.quantiles)


class BaseSingleFitQuantileEstimator(ABC):
    """Abstract base for quantile estimators that model the full conditional distribution."""

    def fit(self, X: np.ndarray, y: np.ndarray, quantiles: List[float]):
        """Fit a single model to capture the conditional distribution.

        Args:
            X: Training features with shape (n_samples, n_features).
            y: Training targets with shape (n_samples,).
            quantiles: List of quantile levels in [0, 1] to extract later.

        Returns:
            Self for method chaining.
        """
        self.quantiles = quantiles
        self._fit_implementation(X, y)
        return self

    @abstractmethod
    def _fit_implementation(self, X: np.ndarray, y: np.ndarray):
        """Implement the model fitting logic for the conditional distribution.

        Args:
            X: Training features with shape (n_samples, n_features).
            y: Training targets with shape (n_samples,).
        """

    @abstractmethod
    def _get_candidate_local_distribution(self, X: np.ndarray) -> np.ndarray:
        """Extract candidate distribution samples for quantile computation.

        Args:
            X: Features with shape (n_samples, n_features).

        Returns:
            Distribution samples with shape (n_samples, n_candidates).
        """

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate quantile predictions from the fitted conditional distribution.

        Args:
            X: Features for prediction with shape (n_samples, n_features).

        Returns:
            Quantile predictions with shape (n_samples, n_quantiles).
        """
        candidate_distribution = self._get_candidate_local_distribution(X)
        quantile_preds = np.quantile(candidate_distribution, self.quantiles, axis=1).T
        return quantile_preds


class QuantRegWrapper:
    """Wrapper for statsmodels quantile regression results to provide scikit-learn interface.

    Adapts statsmodels QuantReg fitted results to provide a predict method compatible
    with the estimator framework. Handles intercept management for proper matrix
    multiplication during prediction.

    Args:
        results: Fitted QuantReg results object from statsmodels.
        has_intercept: Whether an intercept term was added to the design matrix.
    """

    def __init__(self, results, has_intercept):
        self.results = results
        self.has_intercept = has_intercept

    def predict(self, X):
        """Generate predictions using the fitted quantile regression coefficients.

        Args:
            X: Features for prediction with shape (n_samples, n_features).

        Returns:
            Predictions with shape (n_samples,).
        """
        if self.has_intercept:
            X_with_intercept = np.column_stack([np.ones(len(X)), X])
        else:
            X_with_intercept = X

        return X_with_intercept @ self.results.params


class QuantileLasso(BaseMultiFitQuantileEstimator):
    """Linear quantile regression with L1 regularization.

    Args:
        max_iter: Maximum optimization iterations.
        p_tol: Convergence tolerance.
        random_state: Random seed.
    """

    def __init__(
        self,
        max_iter: int = 1000,
        p_tol: float = 1e-6,
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.max_iter = max_iter
        self.p_tol = p_tol
        self.random_state = random_state

    def _fit_quantile_estimator(self, X: np.array, y: np.array, quantile: float):
        """Fit linear quantile regression for a specific quantile level.

        Args:
            X: Training features with shape (n_samples, n_features).
            y: Training targets with shape (n_samples,).
            quantile: Quantile level in [0, 1] to fit model for.

        Returns:
            QuantRegWrapper containing fitted model for the quantile.
        """
        has_added_intercept = not np.any(np.all(X == 1, axis=0))
        if has_added_intercept:
            X_with_intercept = np.column_stack([np.ones(len(X)), X])
        else:
            X_with_intercept = X

        # Add small regularization to prevent numerical issues
        n_features = X_with_intercept.shape[1]
        regularization = 1e-8 * np.eye(n_features)
        X_with_intercept.T @ X_with_intercept + regularization

        if self.random_state is not None:
            np.random.seed(self.random_state)

        try:
            model = QuantReg(y, X_with_intercept)
            result = model.fit(q=quantile, max_iter=self.max_iter, p_tol=self.p_tol)
            return QuantRegWrapper(result, has_added_intercept)
        except np.linalg.LinAlgError:
            # Fallback to robust coordinate descent quantile regression
            warnings.warn(
                f"SVD convergence failed for quantile {quantile}. "
                "Using coordinate descent fallback solution."
            )

            # Use coordinate descent for robust quantile regression
            params = self._coordinate_descent_quantile_regression(
                X_with_intercept, y, quantile
            )

            # Create a mock result object compatible with QuantRegWrapper
            class MockQuantRegResult:
                def __init__(self, params):
                    self.params = params

            mock_result = MockQuantRegResult(params)
            return QuantRegWrapper(mock_result, has_added_intercept)

    def _coordinate_descent_quantile_regression(
        self, X: np.ndarray, y: np.ndarray, quantile: float
    ) -> np.ndarray:
        """Coordinate descent algorithm for quantile regression with regularization.

        Implements a robust coordinate descent solver for quantile regression that
        handles numerical instability better than general-purpose optimizers.
        Uses adaptive step sizes and convergence checking for stability.

        Args:
            X: Design matrix with shape (n_samples, n_features).
            y: Target values with shape (n_samples,).
            quantile: Quantile level in [0, 1].

        Returns:
            Coefficient vector with shape (n_features,).
        """
        n_samples, n_features = X.shape

        # Initialize coefficients with robust least squares estimate
        try:
            # Try regularized least squares initialization
            XtX = X.T @ X + 1e-6 * np.eye(n_features)
            Xty = X.T @ y
            beta = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            # Fallback to zero initialization if solve fails
            beta = np.zeros(n_features)

        # Coordinate descent parameters
        max_iter = self.max_iter
        tolerance = self.p_tol
        lambda_reg = 1e-6  # Small L2 regularization for stability

        # Pre-compute frequently used values
        X_norms_sq = np.sum(X**2, axis=0) + lambda_reg

        for iteration in range(max_iter):
            beta_old = beta.copy()

            # Update each coefficient in turn
            for j in range(n_features):
                # Compute residual without j-th feature
                residual = y - X @ beta + X[:, j] * beta[j]

                # Compute coordinate-wise gradient components
                r_pos = residual >= 0
                r_neg = ~r_pos

                # Subgradient of quantile loss w.r.t. beta[j]
                grad_pos = -quantile * np.sum(X[r_pos, j])
                grad_neg = -(quantile - 1) * np.sum(X[r_neg, j])
                gradient = grad_pos + grad_neg

                # Add L2 regularization gradient
                gradient += lambda_reg * beta[j]

                # Update using coordinate descent step
                # For quantile regression, we use a simple gradient step with adaptive step size
                step_size = 1.0 / X_norms_sq[j]
                beta[j] -= step_size * gradient

                # Apply soft thresholding for implicit L1 regularization
                # This helps with numerical stability
                thresh = 1e-8
                if abs(beta[j]) < thresh:
                    beta[j] = 0.0

            # Check convergence
            param_change = np.linalg.norm(beta - beta_old)
            if param_change < tolerance:
                break

        return beta


class QuantileGBM(BaseMultiFitQuantileEstimator):
    """Gradient boosting quantile regression using scikit-learn backend.

    Implements quantile regression using gradient boosting with the quantile loss
    function. Each quantile level trains a separate GBM model with the alpha
    parameter set to the target quantile. Provides robust non-linear quantile
    estimation with automatic feature selection and interaction detection.

    Args:
        learning_rate: Step size for gradient descent updates.
        n_estimators: Number of boosting stages (trees) to fit.
        min_samples_split: Minimum samples required to split internal nodes.
        min_samples_leaf: Minimum samples required at leaf nodes.
        max_depth: Maximum depth of individual trees.
        subsample: Fraction of samples used for fitting individual trees.
        max_features: Number of features considered for best split.
        random_state: Seed for reproducible tree construction.
    """

    def __init__(
        self,
        learning_rate: float,
        n_estimators: int,
        min_samples_split: Union[float, int],
        min_samples_leaf: Union[float, int],
        max_depth: int,
        subsample: float = 1.0,
        max_features: Union[str, float, int] = None,
        random_state: int = None,
    ):
        super().__init__()
        self.base_estimator = GradientBoostingRegressor(
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            subsample=subsample,
            max_features=max_features,
            random_state=random_state,
            loss="quantile",
        )

    def _fit_quantile_estimator(self, X: np.array, y: np.array, quantile: float):
        """Fit gradient boosting model for a specific quantile level.

        Args:
            X: Training features with shape (n_samples, n_features).
            y: Training targets with shape (n_samples,).
            quantile: Quantile level in [0, 1] to fit model for.

        Returns:
            Fitted GradientBoostingRegressor for the quantile.
        """
        estimator = clone(self.base_estimator)
        estimator.set_params(alpha=quantile)
        estimator.fit(X, y)
        return estimator


class QuantileForest(BaseSingleFitQuantileEstimator):
    """Random forest quantile regression using tree ensemble distributions.

    Implements quantile regression by fitting a single random forest and using
    the distribution of tree predictions to estimate quantiles. This approach
    captures epistemic uncertainty through ensemble diversity and provides
    naturally monotonic quantiles from the empirical tree distribution.

    Args:
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of individual trees.
        max_features: Fraction of features considered for best split.
        min_samples_split: Minimum samples required to split internal nodes.
        bootstrap: Whether to use bootstrap sampling for tree training.
        random_state: Seed for reproducible tree construction.
    """

    def __init__(
        self,
        n_estimators: int = 25,
        max_depth: int = 5,
        max_features: float = 0.8,
        min_samples_leaf: int = 1,
        min_samples_split: int = 2,
        bootstrap: bool = True,
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.base_estimator = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            bootstrap=bootstrap,
            random_state=random_state,
        )

    def _fit_implementation(self, X: np.ndarray, y: np.ndarray):
        """Fit the random forest on the training data.

        Args:
            X: Training features with shape (n_samples, n_features).
            y: Training targets with shape (n_samples,).

        Returns:
            Self for method chaining.
        """
        self.fitted_model = self.base_estimator
        self.fitted_model.fit(X, y)
        return self

    def _get_candidate_local_distribution(self, X: np.ndarray) -> np.ndarray:
        """Extract tree prediction distributions for quantile computation.

        Uses apply() to obtain leaf assignments and then looks up each tree's
        stored leaf mean in a single vectorised pass, avoiding the O(n_trees)
        Python-loop overhead of calling individual estimator.predict() per tree.

        Args:
            X: Features with shape (n_samples, n_features).

        Returns:
            Tree predictions with shape (n_samples, n_estimators).
        """
        # apply() returns (n_samples, n_estimators) leaf node ids in one batched call
        leaf_ids = self.fitted_model.apply(X)  # (n_samples, n_trees)
        n_samples, n_trees = leaf_ids.shape

        sub_preds = np.empty((n_samples, n_trees), dtype=np.float64)
        for b, estimator in enumerate(self.fitted_model.estimators_):
            # tree_.value has shape (n_nodes, n_outputs, max_n_classes);
            # index with leaf ids to get the stored mean for each sample
            sub_preds[:, b] = estimator.tree_.value[leaf_ids[:, b], 0, 0]

        return sub_preds


class QuantileKNN(BaseSingleFitQuantileEstimator):
    """K-nearest neighbors quantile regression using local empirical distributions.

    Implements quantile regression by finding k nearest neighbors for each
    prediction point and using their target value distribution to estimate
    quantiles. This non-parametric approach adapts locally to data density
    and provides natural uncertainty quantification in sparse regions.

    Args:
        n_neighbors: Number of nearest neighbors to use for quantile estimation.
    """

    def __init__(self, n_neighbors: int = 5):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None
        self.nn_model = NearestNeighbors(
            n_neighbors=n_neighbors, algorithm="ball_tree", leaf_size=40
        )

    def _fit_implementation(self, X: np.ndarray, y: np.ndarray):
        """Fit the k-NN model by storing training data and building search index.

        Args:
            X: Training features with shape (n_samples, n_features).
            y: Training targets with shape (n_samples,).

        Returns:
            Self for method chaining.
        """
        self.X_train = X
        self.y_train = y
        self.nn_model.fit(X)
        return self

    def _get_candidate_local_distribution(self, X: np.ndarray) -> np.ndarray:
        """Get neighbor target distributions for quantile computation.

        Args:
            X: Features with shape (n_samples, n_features).

        Returns:
            Neighbor targets with shape (n_samples, n_neighbors).
        """
        _, indices = self.nn_model.kneighbors(X)
        neighbor_preds = self.y_train[indices]
        return neighbor_preds


class QuantileGP(BaseSingleFitQuantileEstimator):
    """Gaussian process quantile regression with robust uncertainty quantification.

    Implements quantile regression using Gaussian processes that model the complete
    conditional distribution p(y|x). Provides analytical quantile computation from
    Gaussian posteriors with proper noise handling and robust hyperparameter optimization.

    All features are treated as continuous using kernels with Automatic Relevance
    Determination (ARD). Categorical features should be one-hot encoded prior to
    being passed to this class.

    Key improvements over basic sklearn GP usage:
    - Proper noise handling without post-hoc kernel modification
    - Robust numerical implementation with Cholesky decomposition
    - Analytical quantile computation for efficiency
    - Batched prediction for memory efficiency
    - Consistent kernel usage between training and prediction
    - ARD kernels for automatic feature relevance determination

    Args:
        kernel: GP kernel specification. Accepts string names ("rbf", "matern",
            "rational_quadratic", "exp_sine_squared") with sensible defaults, or
            custom Kernel objects. Defaults to Matern(nu=1.5).
        noise_variance: Explicit noise variance. If "optimize", will be learned.
            If numeric, uses fixed value. Default is "optimize".
        alpha: Regularization parameter for numerical stability. Range: [1e-12, 1e-6].
        n_restarts_optimizer: Number of restarts for hyperparameter optimization.
        random_state: Seed for reproducible optimization and prediction.
        batch_size: Batch size for prediction to manage memory usage.
        optimize_hyperparameters: Whether to optimize kernel hyperparameters.
            If False, uses kernel as-is.
        prior_lengthscale_concentration: For future custom optimization (unused).
        prior_lengthscale_rate: For future custom optimization (unused).
        prior_noise_concentration: For future custom optimization (unused).
        prior_noise_rate: For future custom optimization (unused).

    Attributes:
        quantiles: List of quantile levels fitted during training.
        X_train_: Training features.
        y_train_: Training targets (normalized).
        kernel_: Fitted kernel with optimized hyperparameters.
        noise_variance_: Fitted noise variance.
        chol_factor_: Cholesky decomposition of kernel matrix.
        alpha_: Precomputed weights for prediction.
        y_train_mean_: Mean of training targets.
        y_train_std_: Standard deviation of training targets.
    """

    def __init__(
        self,
        kernel: Optional[Union[str, Kernel]] = None,
        noise_variance: Optional[Union[str, float]] = "optimize",
        alpha: float = 1e-10,
        n_restarts_optimizer: int = 5,
        random_state: Optional[int] = None,
        batch_size: Optional[int] = None,
        optimize_hyperparameters: bool = True,
        prior_lengthscale_concentration: float = 2.0,
        prior_lengthscale_rate: float = 1.0,
        prior_noise_concentration: float = 1.1,
        prior_noise_rate: float = 30.0,
    ):
        super().__init__()
        self.kernel = kernel
        self.noise_variance = noise_variance
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer
        self.random_state = random_state
        self.batch_size = batch_size
        self.optimize_hyperparameters = optimize_hyperparameters
        self.prior_lengthscale_concentration = prior_lengthscale_concentration
        self.prior_lengthscale_rate = prior_lengthscale_rate
        self.prior_noise_concentration = prior_noise_concentration
        self.prior_noise_rate = prior_noise_rate
        self._ppf_cache = {}

        # Fitted attributes
        self.X_train_ = None
        self.X_train_mean_ = None
        self.X_train_std_ = None
        self.y_train_ = None
        self.kernel_ = None
        self.noise_variance_ = None
        self.chol_factor_ = None
        self.alpha_ = None
        self.y_train_mean_ = None
        self.y_train_std_ = None
        # Eigendecomposition fallback attributes
        self.eigenvals_ = None
        self.eigenvecs_ = None

    def _get_kernel_object(
        self,
        kernel_spec: Optional[Union[str, Kernel]] = None,
        n_features: Optional[int] = None,
    ) -> Kernel:
        """Convert kernel specification to scikit-learn kernel object with ARD support.

        Creates kernels with per-feature length scales for Automatic Relevance
        Determination (ARD). This allows the model to automatically learn the
        importance of each feature by optimizing individual length scales.

        Args:
            kernel_spec: Kernel specification (string name, kernel object, or None).
            n_features: Number of features for ARD initialization. If None, uses scalar length scale.

        Returns:
            Scikit-learn kernel object with proper ARD bounds for optimization.

        Raises:
            ValueError: If unknown kernel name provided or invalid kernel type.
        """
        # Initialize length scale for ARD
        if n_features is not None and n_features > 1:
            # ARD: one length scale per feature
            length_scale = np.ones(n_features)
            length_scale_bounds = (1e-2, 1e2)
        else:
            # Scalar length scale for single feature or unspecified
            length_scale = 1.0
            length_scale_bounds = (1e-2, 1e2)

        # Default to Matern kernel with ARD
        if kernel_spec is None:
            return C(1.0, (1e-3, 1e3)) * Matern(
                length_scale=length_scale,
                length_scale_bounds=length_scale_bounds,
                nu=2.5,
            )

        # String specifications with ARD support
        elif isinstance(kernel_spec, str):
            kernel_map = {
                "rbf": C(1.0, (1e-3, 1e3))
                * RBF(
                    length_scale=length_scale, length_scale_bounds=length_scale_bounds
                ),
                "matern": C(1.0, (1e-3, 1e3))
                * Matern(
                    length_scale=length_scale,
                    length_scale_bounds=length_scale_bounds,
                    nu=2.5,
                ),
                "rational_quadratic": C(1.0, (1e-3, 1e3))
                * RationalQuadratic(
                    length_scale=length_scale,
                    length_scale_bounds=length_scale_bounds,
                    alpha=1.0,
                    alpha_bounds=(1e-3, 1e3),
                ),
                "exp_sine_squared": C(1.0, (1e-3, 1e3))
                * ExpSineSquared(
                    length_scale=length_scale,
                    length_scale_bounds=length_scale_bounds,
                    periodicity=1.0,
                    periodicity_bounds=(1e-2, 1e2),
                ),
            }

            if kernel_spec not in kernel_map:
                raise ValueError(f"Unknown kernel name: {kernel_spec}")
            return kernel_map[kernel_spec]

        # Kernel object - make a deep copy for safety
        elif isinstance(kernel_spec, Kernel):
            return copy.deepcopy(kernel_spec)

        else:
            raise ValueError(
                f"Kernel must be a string name, Kernel object, or None. Got: {type(kernel_spec)}"
            )

    def _optimize_hyperparameters(self) -> None:
        """Optimize kernel hyperparameters and noise variance using sklearn's optimization."""
        if not self.optimize_hyperparameters:
            return

        # Determine alpha value for optimization
        # If noise_variance is "optimize", use a small alpha and let GP optimize noise
        # If noise_variance is fixed, use it as alpha
        if self.noise_variance == "optimize":
            # We use a small fixed alpha for numerical stability, WhiteKernel handles the actual noise
            kernel_to_fit = self.kernel_ + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1e1))
            alpha_for_opt = max(self.alpha, 1e-6)
        else:
            kernel_to_fit = self.kernel_
            alpha_for_opt = self.noise_variance_ + self.alpha

        # Use sklearn's GaussianProcessRegressor for hyperparameter optimization
        # This provides robust optimization with proper parameter mapping
        temp_gp = GaussianProcessRegressor(
            kernel=kernel_to_fit,
            alpha=alpha_for_opt,
            n_restarts_optimizer=self.n_restarts_optimizer,
            random_state=self.random_state,
            normalize_y=False,  # We handle normalization ourselves
        )

        try:
            # Suppress sklearn GP convergence warnings about parameter bounds
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=".*close to the specified.*bound.*",
                    category=UserWarning,
                    module="sklearn.gaussian_process.kernels",
                )
                warnings.filterwarnings(
                    "ignore",
                    category=ConvergenceWarning,
                )
                temp_gp.fit(self.X_train_, self.y_train_)
            
            # Extract optimized kernel
            if self.noise_variance == "optimize":
                # temp_gp.kernel_ is a Sum(base_kernel, WhiteKernel)
                # Extract the optimized base kernel and the optimized noise level
                self.kernel_ = temp_gp.kernel_.k1
                # The true optimized noise variance includes alpha_for_opt.
                # We subtract self.alpha so that when self.alpha is added in _fit_gp,
                # the total noise matches the optimized noise exactly.
                self.noise_variance_ = temp_gp.kernel_.k2.noise_level + alpha_for_opt - self.alpha
            else:
                self.kernel_ = temp_gp.kernel_

        except Exception as e:
            logging.warning(
                f"Hyperparameter optimization failed: {e}, using default parameters"
            )
            # Keep the original kernel and noise variance if optimization fails

    def _fit_implementation(self, X: np.ndarray, y: np.ndarray) -> "QuantileGP":
        """Fit Gaussian process with proper hyperparameter optimization.

        Implements robust GP fitting with:
        - Custom hyperparameter optimization with principled priors
        - Proper noise handling without post-hoc kernel modification
        - Numerical stability through Cholesky decomposition

        Args:
            X: Training features with shape (n_samples, n_features).
            y: Training targets with shape (n_samples,).

        Returns:
            Self for method chaining.
        """
        # Normalize features
        self.X_train_mean_ = np.mean(X, axis=0)
        self.X_train_std_ = np.std(X, axis=0)
        # Handle constant features
        self.X_train_std_[self.X_train_std_ < 1e-12] = 1.0
        self.X_train_ = (X - self.X_train_mean_) / self.X_train_std_

        # Normalize targets
        self.y_train_mean_ = np.mean(y)
        self.y_train_std_ = np.std(y)
        if self.y_train_std_ < 1e-12:
            self.y_train_std_ = 1.0
        self.y_train_ = (y - self.y_train_mean_) / self.y_train_std_

        # Initialize kernel with ARD support
        n_features = X.shape[1]
        self.kernel_ = self._get_kernel_object(self.kernel, n_features)

        # Set noise variance in normalized target space.
        # The kernel matrix is built on normalized targets (zero-mean, unit-variance),
        # so noise variance must be converted to the same normalized scale.
        if isinstance(self.noise_variance, (int, float)):
            self.noise_variance_ = self.noise_variance / self.y_train_std_**2
        else:
            self.noise_variance_ = 1e-6  # Default, will be optimized if needed

        # Optimize hyperparameters
        self._optimize_hyperparameters()

        # Fit the model with optimized parameters
        self._fit_gp()

        return self

    def _fit_gp(self) -> None:
        """Fit GP with current hyperparameters using robust Cholesky decomposition."""
        # Compute kernel matrix
        K = self.kernel_(self.X_train_)

        # Add noise and regularization
        K[np.diag_indices(len(self.X_train_))] += self.noise_variance_ + self.alpha

        # Robust Cholesky decomposition with progressive regularization
        regularization_levels = [0, 1e-8, 1e-6, 1e-4, 1e-3]
        
        self.effective_noise_variance_ = self.noise_variance_ + self.alpha

        for reg in regularization_levels:
            try:
                K_reg = K.copy()
                if reg > 0:
                    K_reg[np.diag_indices(len(self.X_train_))] += reg
                self.chol_factor_ = cholesky(K_reg, lower=True)
                if reg > 0:
                    logging.warning(
                        f"Added regularization {reg} for numerical stability"
                    )
                    self.effective_noise_variance_ += reg
                break
            except LinAlgError:
                if reg == regularization_levels[-1]:
                    # Final fallback: use eigendecomposition for very ill-conditioned matrices
                    logging.warning(
                        "Cholesky failed, using eigendecomposition fallback"
                    )
                    self._fit_gp_eigendecomp(K)
                    return
                continue

        # Solve for alpha = K^-1 y using Cholesky decomposition
        # Optuna's math for alpha (cov_Y_Y_inv_Y)
        self.alpha_ = solve_triangular(
            self.chol_factor_.T,
            solve_triangular(self.chol_factor_, self.y_train_, lower=True),
            lower=False,
        )

    def _fit_gp_eigendecomp(self, K: np.ndarray) -> None:
        """Fallback GP fitting using eigendecomposition for ill-conditioned matrices."""
        # Eigendecomposition of kernel matrix
        eigenvals, eigenvecs = np.linalg.eigh(K)

        # Clip negative eigenvalues and add regularization
        eigenvals = np.maximum(eigenvals, 1e-12)

        # Use pseudo-inverse for fitting
        try:
            # More stable computation of K^-1 y avoiding explicit K_inv construction
            self.alpha_ = eigenvecs @ ((eigenvecs.T @ self.y_train_) / eigenvals)
            # Store decomposition for prediction
            self.eigenvals_ = eigenvals
            self.eigenvecs_ = eigenvecs
            self.chol_factor_ = None  # Signal to use eigendecomp in prediction
        except Exception as e:
            raise RuntimeError(f"Both Cholesky and eigendecomposition failed: {e}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate quantile predictions using analytical Gaussian distribution.

        Uses the GP posterior mean and variance to compute quantiles analytically
        as q_τ(x) = μ(x) + σ(x)Φ⁻¹(τ), ensuring monotonic quantile ordering.

        Args:
            X: Features for prediction with shape (n_samples, n_features).

        Returns:
            Quantile predictions with shape (n_samples, n_quantiles).
        """
        if self.batch_size is not None and len(X) > self.batch_size:
            results = []
            for i in range(0, len(X), self.batch_size):
                batch_X = X[i : i + self.batch_size]
                batch_result = self._predict_batch(batch_X)
                results.append(batch_result)
            return np.vstack(results)
        else:
            return self._predict_batch(X)

    def _predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Compute quantiles analytically from GP posterior.

        Args:
            X: Features with shape (batch_size, n_features).

        Returns:
            Quantile predictions with shape (batch_size, n_quantiles).
        """
        # Get mean and variance from GP
        y_mean, y_var = self._predict_mean_var(X)
        y_std = np.sqrt(y_var).reshape(-1, 1)

        # Get cached inverse normal CDF values
        ppf_values = self._get_cached_ppf_values()

        # Compute quantiles analytically
        quantile_preds = y_mean.reshape(-1, 1) + y_std * ppf_values.reshape(1, -1)

        return quantile_preds

    def _predict_mean_var(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict mean and variance using Cholesky or eigendecomposition.

        Args:
            X: Features with shape (n_samples, n_features).

        Returns:
            Tuple of (y_mean, y_var) with shapes (n_samples,) each.
        """
        # Normalize test features
        X_norm = (X - self.X_train_mean_) / self.X_train_std_

        # Compute kernel between test and training points
        K_star = self.kernel_(X_norm, self.X_train_)

        if self.chol_factor_ is not None:
            # Optuna's math for prediction
            y_mean = np.dot(K_star, self.alpha_)

            # V = K_star @ inv(C)
            V = solve_triangular(
                self.chol_factor_.T,
                solve_triangular(self.chol_factor_, K_star.T, lower=True),
                lower=False,
            ).T
            
            K_star_star = self.kernel_.diag(X_norm)
            y_var = K_star_star - np.sum(K_star * V, axis=1)

        else:
            # Use eigendecomposition fallback
            y_mean = K_star @ self.alpha_

            # Compute variance using eigendecomposition
            K_star_star = self.kernel_.diag(X_norm)
            # K^{-1} K_*^T = V * Λ^{-1} * V^T * K_*^T
            K_inv_K_star = (
                self.eigenvecs_
                @ ((self.eigenvecs_.T @ K_star.T) / self.eigenvals_.reshape(-1, 1))
            )
            y_var = K_star_star - np.sum(K_star * K_inv_K_star.T, axis=1)

        # Denormalize mean
        y_mean = y_mean * self.y_train_std_ + self.y_train_mean_

        # Ensure non-negative variance before denormalization
        y_var = np.maximum(y_var, 0.0)

        # Denormalize variance (transforms from normalized to original scale)
        y_var *= self.y_train_std_**2

        # Add noise variance in original scale for total predictive variance
        # The total noise variance assumed by the model includes alpha and any regularization
        y_var += self.effective_noise_variance_ * self.y_train_std_**2

        return y_mean, y_var

    def _get_cached_ppf_values(self) -> np.ndarray:
        """Cache inverse normal CDF values for efficiency.

        Returns:
            Cached inverse normal CDF values with shape (n_quantiles,).
        """
        quantiles_key = tuple(self.quantiles)
        if quantiles_key not in self._ppf_cache:
            self._ppf_cache[quantiles_key] = np.array(
                [norm.ppf(q) for q in self.quantiles]
            )
        return self._ppf_cache[quantiles_key]

    def _get_candidate_local_distribution(self, X: np.ndarray) -> np.ndarray:
        """Generate posterior samples for Monte Carlo quantile estimation.

        This method is required by the base class but not used by this implementation
        since we use analytical quantile computation. Included for compatibility.

        Args:
            X: Features with shape (n_samples, n_features).

        Returns:
            Posterior samples with shape (n_samples, n_samples_per_point).
        """
        # Get mean and variance from GP
        y_mean, y_var = self._predict_mean_var(X)
        y_std = np.sqrt(y_var)

        # Generate samples from the GP posterior for each test point
        rng = np.random.RandomState(self.random_state)
        n_samples = 1000  # Default number of samples
        samples = np.array(
            [rng.normal(y_mean[i], y_std[i], size=n_samples) for i in range(len(X))]
        )
        return samples


class QuantileLeaf(BaseSingleFitQuantileEstimator):
    """Quantile Regression Forest via leaf-weighted empirical CDF (Meinshausen 2006).

    Each training sample receives a proximity weight relative to a test point x:

        w_i(x) = (1/B) * sum_b [ 1(X_i in L_b(x)) / |L_b(x)| ]

    where L_b(x) is the leaf reached by x in tree b and |L_b(x)| is the count of
    in-bag training samples in that leaf. Quantiles are read from the weighted
    empirical CDF F_hat(y|x) = sum_i w_i(x) * 1(Y_i <= y) using linear
    interpolation, matching numpy's default quantile convention.

    Bootstrap membership is reconstructed via sklearn's internal
    ``_generate_sample_indices`` with the exact RNG state each tree used during
    ``forest.fit()``, guaranteeing identical in-bag sets.

    At fit time a leaf-to-rank-index table is built once per tree, and weight
    rows for all training samples are pre-computed and cached (keyed by their
    leaf-ID signature). Tree leaf assignments use direct Cython ``tree_.apply``
    calls, bypassing sklearn's per-call Python validation layer.

    Args:
        n_estimators: Number of trees in the forest.
        max_depth: Maximum depth of individual trees.
        max_features: Fraction of features considered at each split.
        min_samples_split: Minimum samples required to split an internal node.
        min_samples_leaf: Minimum samples required at a leaf node.
        bootstrap: Whether to use bootstrap sampling for each tree.
        random_state: Seed for reproducible tree construction.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        max_features: float = 0.8,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        bootstrap: bool = True,
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.y_train_sorted: Optional[np.ndarray] = None
        self.forest = None
        self._leaf_table: Optional[List[Dict[int, np.ndarray]]] = None

    def _fit_implementation(self, X: np.ndarray, y: np.ndarray):
        """Fit the forest and build the per-tree leaf-to-rank-index table.

        Args:
            X: Training features, shape (n_samples, n_features).
            y: Training targets, shape (n_samples,).

        Returns:
            Self.
        """
        n_train = len(y)

        self.forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_features=self.max_features,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
        )
        self.forest.fit(X, y)

        sorter = np.argsort(y)
        self.y_train_sorted = y[sorter]
        rank_of = np.argsort(sorter)

        n_bootstrap = _get_n_samples_bootstrap(n_train, self.forest.max_samples)
        train_leaf_ids = self.forest.apply(X)
        self._tree_apply_fns = [est.tree_.apply for est in self.forest.estimators_]

        self._leaf_table = []
        for b, estimator in enumerate(self.forest.estimators_):
            bootstrap_indices = (
                _generate_sample_indices(estimator.random_state, n_train, n_bootstrap)
                if self.bootstrap
                else np.arange(n_train)
            )

            inbag_ranks = rank_of[bootstrap_indices]
            inbag_leaf_ids = train_leaf_ids[bootstrap_indices, b]

            sorted_order = np.argsort(inbag_leaf_ids)
            unique_leaves, starts = np.unique(inbag_leaf_ids[sorted_order], return_index=True)
            ends = np.append(starts[1:], len(sorted_order))

            self._leaf_table.append({
                int(leaf): np.sort(inbag_ranks[sorted_order[s:e]])
                for leaf, s, e in zip(unique_leaves, starts, ends)
            })

        self._weight_cache: Dict[bytes, np.ndarray] = {}
        train_weights = self._weights_from_leaf_ids(train_leaf_ids)
        for i in range(n_train):
            self._weight_cache[train_leaf_ids[i].tobytes()] = train_weights[i]

        return self

    def _weights_from_leaf_ids(self, leaf_ids: np.ndarray) -> np.ndarray:
        """Compute a Meinshausen (2006) weight matrix from leaf-ID assignments.

        For each tree b, test points sharing a leaf with in-bag training samples
        receive weight ``1/|L_b(x)|`` from those samples. Weights are accumulated
        in rank space (aligned to ``y_train_sorted``) across all trees, then
        normalised to sum to 1 per row.

        Args:
            leaf_ids: Integer array (n_test, n_trees) of leaf node IDs.

        Returns:
            Weight matrix (n_test, n_train_sorted), rows sum to 1.
        """
        n_test = len(leaf_ids)
        n_train = len(self.y_train_sorted)
        n_trees = len(self.forest.estimators_)

        weights = np.zeros((n_test, n_train), dtype=np.float64)

        for b in range(n_trees):
            tree_table = self._leaf_table[b]
            leaves_for_tree = leaf_ids[:, b]

            sorted_order = np.argsort(leaves_for_tree)
            unique_leaves, starts = np.unique(leaves_for_tree[sorted_order], return_index=True)
            ends = np.append(starts[1:], n_test)

            for leaf, ts, te in zip(unique_leaves, starts, ends):
                inbag_ranks = tree_table.get(int(leaf))
                if inbag_ranks is None or len(inbag_ranks) == 0:
                    continue
                test_indices = sorted_order[ts:te]
                weights[np.ix_(test_indices, inbag_ranks)] += 1.0 / len(inbag_ranks)

        weights /= n_trees
        row_sums = weights.sum(axis=1, keepdims=True)
        weights /= np.where(row_sums == 0, 1.0, row_sums)
        return weights

    def _proximity_weights(self, X: np.ndarray) -> np.ndarray:
        """Return the weight matrix for X, served from cache where possible.

        Each test point's weight row is uniquely determined by its leaf-ID
        signature across all trees. The cache is keyed by this signature
        (stable across feature-scaler changes). Misses are computed via the
        vectorised ``_weights_from_leaf_ids`` and then stored.

        Args:
            X: Test features, shape (n_test, n_features).

        Returns:
            Weight matrix (n_test, n_train_sorted), rows sum to 1.
        """
        from sklearn.tree._tree import DTYPE as _SKLEARN_DTYPE

        n_test = len(X)
        n_train = len(self.y_train_sorted)
        n_trees = len(self.forest.estimators_)

        X_f32 = np.asarray(X, dtype=_SKLEARN_DTYPE, order="C")
        leaf_ids = np.column_stack(
            [self._tree_apply_fns[b](X_f32) for b in range(n_trees)]
        )

        weights = np.empty((n_test, n_train), dtype=np.float64)
        cache = self._weight_cache
        miss_indices = []

        for i in range(n_test):
            cached = cache.get(leaf_ids[i].tobytes())
            if cached is not None:
                weights[i] = cached
            else:
                miss_indices.append(i)

        if miss_indices:
            miss_arr = np.array(miss_indices)
            miss_weights = self._weights_from_leaf_ids(leaf_ids[miss_arr])
            for local_i, global_i in enumerate(miss_indices):
                w = miss_weights[local_i]
                weights[global_i] = w
                cache[leaf_ids[global_i].tobytes()] = w

        return weights

    def _get_candidate_local_distribution(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "QuantileLeaf predicts via a weighted empirical CDF; call predict() directly."
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return quantile predictions for X via the weighted empirical CDF.

        Steps:
        1. Compute proximity weights w_i(x) for each test point.
        2. Form the weighted empirical CDF over sorted training targets.
        3. Invert the CDF at each requested quantile with linear interpolation
           (matching numpy's default 'linear' method).

        Args:
            X: Features, shape (n_samples, n_features).

        Returns:
            Quantile predictions, shape (n_samples, n_quantiles).
        """
        weights = self._proximity_weights(X)
        cdf = np.cumsum(weights, axis=1)

        quantiles_arr = np.asarray(self.quantiles)
        n_test, n_train = weights.shape

        r_hi = np.apply_along_axis(
            lambda row: np.searchsorted(row, quantiles_arr, side="left"),
            axis=1,
            arr=cdf,
        )
        r_hi = np.clip(r_hi, 0, n_train - 1)
        r_lo = np.clip(r_hi - 1, 0, n_train - 1)

        y_hi = self.y_train_sorted[r_hi]
        y_lo = self.y_train_sorted[r_lo]
        cdf_hi = cdf[np.arange(n_test)[:, None], r_hi]
        cdf_lo = cdf[np.arange(n_test)[:, None], r_lo]

        denom = cdf_hi - cdf_lo
        safe_denom = np.where(denom > 0, denom, 1.0)
        fraction = np.where(denom > 0, (quantiles_arr[None, :] - cdf_lo) / safe_denom, 0.0)

        return y_lo + fraction * (y_hi - y_lo)


class SplineQuantRegWrapper:
    """Fitted single-quantile spline-GAM model.

    Stores the intercept separately from spline/binary coefficients so the
    intercept is never subject to L1 regularization.

    Args:
        coef_: Coefficient vector for spline + binary features, shape (n_design,).
        intercept_: Scalar intercept (0.0 when ``add_intercept=False``).
        spline_transformer: Fitted ``SplineTransformer`` applied to continuous columns.
        continuous_cols: Integer indices of continuous feature columns.
        binary_cols: Integer indices of binary (passthrough) feature columns.
        add_intercept: Whether an intercept was fitted.
    """

    def __init__(
        self,
        coef_: np.ndarray,
        intercept_: float,
        spline_transformer: SplineTransformer,
        continuous_cols: np.ndarray,
        binary_cols: np.ndarray,
        add_intercept: bool,
    ):
        self.coef_ = np.asarray(coef_, dtype=np.float64)
        self.intercept_ = float(intercept_)
        self.spline_transformer = spline_transformer
        self.continuous_cols = continuous_cols
        self.binary_cols = binary_cols
        self.add_intercept = add_intercept
        self.n_features_in_: int = len(continuous_cols) + len(binary_cols)
        self.design_n_features_: int = len(self.coef_)

        _p = (
            np.concatenate([[self.intercept_], self.coef_])
            if add_intercept
            else self.coef_
        )

        class _Result:
            def __init__(self, params: np.ndarray):
                self.params = params

        self.result = _Result(_p)

    def _transform_X(self, X: np.ndarray) -> np.ndarray:
        """Build the feature matrix from raw inputs.

        Args:
            X: Raw features, shape (n_samples, n_features).

        Returns:
            Feature matrix, shape (n_samples, design_n_features_).
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        parts = []
        if len(self.continuous_cols) > 0:
            parts.append(self.spline_transformer.transform(X[:, self.continuous_cols]))
        if len(self.binary_cols) > 0:
            parts.append(X[:, self.binary_cols].astype(np.float64))
        if not parts:
            return np.zeros((len(X), 0), dtype=np.float64)
        return np.concatenate(parts, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return quantile predictions.

        Args:
            X: Features, shape (n_samples, n_features).

        Returns:
            Predictions, shape (n_samples,).
        """
        return self._transform_X(X) @ self.coef_ + self.intercept_


class SplineQuantileRegressor(BaseMultiFitQuantileEstimator):
    """Per-quantile spline GAM using B-spline expansion of continuous features.

    Fits one independent quantile regression model per requested quantile level.
    Continuous input features are expanded into B-spline basis functions using
    :class:`~sklearn.preprocessing.SplineTransformer`; binary (0/1) columns
    pass through unchanged. Binary columns are detected automatically.

    The intercept is always fitted without L1 regularization, preventing
    predictions from collapsing toward the training mean when ``alpha > 0``.

    Args:
        n_knots: Number of knots per continuous feature.
        degree: Polynomial degree of the B-splines (default 3 = cubic).
        knots: Knot placement strategy passed to ``SplineTransformer``.
            ``"quantile"`` places knots at equal quantiles of the training
            distribution; ``"uniform"`` spaces them evenly over the range.
        extrapolation: Extrapolation strategy passed to ``SplineTransformer``
            (``"linear"``, ``"constant"``, ``"continue"``, or ``"periodic"``).
        include_bias: Whether to include a bias column in the spline basis.
        binary_threshold: Tolerance for binary-column detection.
        add_intercept: Whether to fit an unregularized intercept term.
        alpha: L1 regularization strength on spline/binary coefficients.
            The intercept is always exempt from regularization.
        solver: Backend solver. ``"highs"`` uses HiGHS interior-point LP via
            ``sklearn.linear_model.QuantileRegressor``; ``"statsmodels"`` uses
            IRLS with a coordinate-descent fallback.
        max_iter: Maximum solver iterations.
        p_tol: Convergence tolerance (statsmodels IRLS only).
        monotone_rearrange: Apply monotone rearrangement to prevent quantile
            crossing (inherited from :class:`BaseMultiFitQuantileEstimator`).
        random_state: Random seed (reserved for reproducibility).

    Attributes:
        binary_cols_: Integer array of detected binary column indices.
        continuous_cols_: Integer array of detected continuous column indices.
        trained_estimators: List of :class:`SplineQuantRegWrapper`, one per quantile.
        quantiles: Quantile levels passed to ``fit``.
    """

    def __init__(
        self,
        n_knots: int = 6,
        degree: int = 3,
        knots: str = "quantile",
        extrapolation: str = "linear",
        include_bias: bool = False,
        binary_threshold: float = 1e-12,
        add_intercept: bool = True,
        alpha: float = 0.001,
        solver: str = "highs",
        max_iter: int = 1000,
        p_tol: float = 1e-6,
        monotone_rearrange: bool = True,
        random_state: Optional[int] = None,
    ):
        super().__init__()
        self.n_knots = n_knots
        self.degree = degree
        self.knots = knots
        self.extrapolation = extrapolation
        self.include_bias = include_bias
        self.binary_threshold = binary_threshold
        self.add_intercept = add_intercept
        self.alpha = alpha
        self.solver = solver
        self.max_iter = max_iter
        self.p_tol = p_tol
        self.monotone_rearrange = monotone_rearrange
        self.random_state = random_state

        self.binary_cols_: Optional[np.ndarray] = None
        self.continuous_cols_: Optional[np.ndarray] = None
        self.trained_estimators: list = []

    def fit(self, X: np.ndarray, y: np.ndarray, quantiles: List[float]):
        """Detect feature types then fit one model per quantile.

        Args:
            X: Training features, shape (n_samples, n_features).
            y: Training targets, shape (n_samples,).
            quantiles: Quantile levels strictly inside (0, 1).

        Returns:
            Self.

        Raises:
            ValueError: On invalid inputs.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X = check_array(X, dtype=np.float64, ensure_2d=True)
        y = np.asarray(y, dtype=np.float64).ravel()

        if X.shape[0] != len(y):
            raise ValueError(f"X has {X.shape[0]} rows but y has {len(y)} elements.")
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("X contains NaN or Inf values.")
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            raise ValueError("y contains NaN or Inf values.")

        quantiles_arr = np.asarray(quantiles, dtype=np.float64)
        if np.any(quantiles_arr <= 0.0) or np.any(quantiles_arr >= 1.0):
            raise ValueError("All quantiles must be strictly inside (0, 1).")

        n_features = X.shape[1]
        binary_mask = np.zeros(n_features, dtype=bool)
        for j in range(n_features):
            col = X[:, j]
            finite_vals = col[np.isfinite(col)]
            unique_vals = np.unique(finite_vals)
            binary_mask[j] = np.all(
                np.isclose(unique_vals, 0, atol=self.binary_threshold)
                | np.isclose(unique_vals, 1, atol=self.binary_threshold)
            )

        self.binary_cols_ = np.where(binary_mask)[0]
        self.continuous_cols_ = np.where(~binary_mask)[0]

        if len(self.continuous_cols_) == 0 and len(self.binary_cols_) == 0:
            raise ValueError("X has no usable columns.")

        return super().fit(X, y, list(quantiles_arr))

    def _fit_quantile_estimator(
        self, X: np.ndarray, y: np.ndarray, quantile: float
    ) -> SplineQuantRegWrapper:
        """Fit a single spline-GAM quantile model.

        Args:
            X: Training features, shape (n_samples, n_features).
            y: Training targets, shape (n_samples,).
            quantile: Quantile level in (0, 1).

        Returns:
            Fitted :class:`SplineQuantRegWrapper`.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        spline_transformer = SplineTransformer(
            n_knots=self.n_knots,
            degree=self.degree,
            knots=self.knots,
            include_bias=self.include_bias,
            extrapolation=self.extrapolation,
        )

        feat_parts: list = []
        if len(self.continuous_cols_) > 0:
            feat_parts.append(spline_transformer.fit_transform(X[:, self.continuous_cols_]))
        else:
            spline_transformer.fit(np.zeros((len(y), 1)))

        if len(self.binary_cols_) > 0:
            feat_parts.append(X[:, self.binary_cols_].astype(np.float64))

        X_features = (
            np.concatenate(feat_parts, axis=1)
            if feat_parts
            else np.zeros((len(y), 0), dtype=np.float64)
        )

        if self.solver == "highs":
            qr = SKLearnQuantileRegressor(
                quantile=quantile,
                alpha=self.alpha,
                solver="highs-ipm",
                solver_options={"maxiter": self.max_iter},
                fit_intercept=self.add_intercept,
            )
            qr.fit(X_features, y)
            coef_ = qr.coef_
            intercept_ = float(qr.intercept_) if self.add_intercept else 0.0

        elif self.solver == "statsmodels":
            if self.add_intercept:
                X_design = np.column_stack([np.ones(len(y)), X_features])
            else:
                X_design = X_features
            try:
                params = QuantReg(y, X_design).fit(
                    q=quantile, max_iter=self.max_iter, p_tol=self.p_tol
                ).params
            except np.linalg.LinAlgError:
                warnings.warn(
                    f"QuantReg SVD failed for quantile {quantile}. "
                    "Falling back to coordinate-descent solver."
                )
                params = self._coordinate_descent_qr(X_design, y, quantile)

            if self.add_intercept:
                intercept_ = float(params[0])
                coef_ = params[1:]
            else:
                intercept_ = 0.0
                coef_ = params
        else:
            raise ValueError(f"Unknown solver '{self.solver}'. Use 'highs' or 'statsmodels'.")

        return SplineQuantRegWrapper(
            coef_=coef_,
            intercept_=intercept_,
            spline_transformer=spline_transformer,
            continuous_cols=self.continuous_cols_,
            binary_cols=self.binary_cols_,
            add_intercept=self.add_intercept,
        )

    def _coordinate_descent_qr(
        self, X: np.ndarray, y: np.ndarray, quantile: float
    ) -> np.ndarray:
        """Coordinate-descent quantile regression (statsmodels fallback).

        Args:
            X: Design matrix, shape (n_samples, n_design_features).
            y: Target values, shape (n_samples,).
            quantile: Quantile level in (0, 1).

        Returns:
            Coefficient vector, shape (n_design_features,).
        """
        n_samples, n_features = X.shape
        lambda_reg = 1e-6

        try:
            beta = np.linalg.solve(X.T @ X + lambda_reg * np.eye(n_features), X.T @ y)
        except np.linalg.LinAlgError:
            beta = np.zeros(n_features)

        X_norms_sq = np.sum(X ** 2, axis=0) + lambda_reg

        for _ in range(self.max_iter):
            beta_old = beta.copy()
            for j in range(n_features):
                residual = y - X @ beta + X[:, j] * beta[j]
                r_pos = residual >= 0
                gradient = (
                    -quantile * np.sum(X[r_pos, j])
                    - (quantile - 1) * np.sum(X[~r_pos, j])
                    + lambda_reg * beta[j]
                )
                beta[j] -= gradient / X_norms_sq[j]
                if abs(beta[j]) < 1e-8:
                    beta[j] = 0.0
            if np.linalg.norm(beta - beta_old) < self.p_tol:
                break

        return beta
