import logging
import numpy as np
from typing import Optional, Tuple, List, Literal
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from ccqr_optimization.wrapping import ConformalBounds
from ccqr_optimization.utils.preprocessing import train_val_split
from ccqr_optimization.selection.estimation import (
    initialize_estimator,
    QuantileTuner,
)
from ccqr_optimization.selection.estimator_configuration import ESTIMATOR_REGISTRY
from copy import deepcopy

logger = logging.getLogger(__name__)


def set_calibration_split(n_observations: int) -> float:
    """Set to 20%, but limit to between 4 and 8 observations
    since we tend to only need at most 4 quantiles for conformal search"""
    candidate_split = 0.2
    if candidate_split * n_observations < 4:
        return 4 / n_observations
    else:
        return candidate_split


def alpha_to_quantiles(alpha: float) -> Tuple[float, float]:
    """Convert alpha level to symmetric quantile pair.

    Transforms a miscoverage level alpha into corresponding lower and upper
    quantiles for symmetric prediction intervals.

    Args:
        alpha: Miscoverage level in (0, 1). Coverage = 1 - alpha.

    Returns:
        Tuple of (lower_quantile, upper_quantile) where:
            - lower_quantile = alpha / 2
            - upper_quantile = 1 - alpha / 2

    Mathematical Details:
        For symmetric intervals with coverage 1-α:
        - Lower quantile: α/2 (captures α/2 probability in left tail)
        - Upper quantile: 1-α/2 (captures α/2 probability in right tail)
    """
    lower_quantile = alpha / 2
    upper_quantile = 1 - lower_quantile
    return lower_quantile, upper_quantile


class QuantileConformalEstimator:
    """Quantile-based conformal predictor with adaptive splitting strategies.

    Implements conformal prediction for quantile regression with adaptive splitting
    strategies (K-fold cross-validation, train-test split, or adaptive selection).
    The estimator supports both conformalized and non-conformalized modes.

    The estimator supports both conformalized and non-conformalized modes:
    - Conformalized: Conformal calibration for principled uncertainty quantification
    - Non-conformalized: Direct quantile predictions (when data is limited)

    Args:
        quantile_estimator_architecture: Architecture identifier for quantile estimator.
            Must be registered in ESTIMATOR_REGISTRY and support quantile fitting.
        alphas: List of miscoverage levels (1-alpha gives coverage probability).
            Must be in (0, 1) range.
        n_pre_conformal_trials: Minimum samples required for conformal calibration.
            Below this threshold, uses direct quantile prediction.

    Attributes:
        fold_estimators: List of fitted quantile regression models.
        nonconformity_scores: Calibration scores per alpha level.
        all_quantiles: Sorted list of all required quantiles.
        quantile_indices: Mapping from quantile values to prediction array indices.
        conformalize_predictions: Boolean flag indicating if conformal adjustment is used.
    """

    def __init__(
        self,
        quantile_estimator_architecture: str,
        alphas: List[float],
        n_pre_conformal_trials: int = 32,
        n_calibration_folds: int = 3,
        calibration_split_strategy: Literal[
            "cv", "train_test_split", "adaptive"
        ] = "adaptive",
        adaptive_threshold: int = 50,
        normalize_features: bool = True,
    ):
        self.quantile_estimator_architecture = quantile_estimator_architecture
        self.alphas = alphas
        self.updated_alphas = self.alphas.copy()
        self.n_pre_conformal_trials = n_pre_conformal_trials
        self.n_calibration_folds = n_calibration_folds
        self.calibration_split_strategy = calibration_split_strategy
        self.adaptive_threshold = adaptive_threshold
        self.normalize_features = normalize_features

        self.quantile_estimator = None
        self.nonconformity_scores = None
        self.all_quantiles = None
        self.quantile_indices = None
        self.conformalize_predictions = False
        self.last_best_params = None
        self.feature_scaler = None
        self.fold_estimators = []  # Store K-fold estimators for CV+

    def _determine_splitting_strategy(self, total_size: int) -> str:
        """Determine optimal data splitting strategy based on dataset size and configuration.

        Selects between cross-validation (CV) and train-test split approaches for quantile-based conformal
        calibration based on the configured strategy and dataset characteristics. The
        adaptive strategy automatically chooses the most appropriate method based on
        data size to balance computational efficiency with calibration stability.

        Args:
            total_size: Total number of samples in the dataset.

        Returns:
            Strategy identifier: "cv" or "train_test_split".

        Strategy Selection Logic:
            - "adaptive": Uses K-fold cross-validation for small datasets to improve
              calibration stability, and switches to train-test split for larger
              datasets to improve computational efficiency
            - "cv": Always uses K-fold cross-validation-based calibration
            - "train_test_split": Always uses single split calibration
        """
        if self.calibration_split_strategy == "adaptive":
            return "cv" if total_size < self.adaptive_threshold else "train_test_split"
        return self.calibration_split_strategy

    def _fit_non_conformal(
        self,
        X: np.ndarray,
        y: np.ndarray,
        all_quantiles: List[float],
        tuning_iterations: int,
        min_obs_for_tuning: int,
        random_state: Optional[int],
        last_best_params: Optional[dict],
    ):
        """Fit quantile estimator without conformal calibration for small datasets.

        Trains a quantile regression model directly on the provided data without
        applying conformal prediction adjustments. This mode is used when the dataset
        is too small for reliable conformal calibration (below n_pre_conformal_trials
        threshold), providing direct quantile predictions instead of conformally
        adjusted intervals.

        While this approach loses the finite-sample coverage guarantees of conformal
        prediction, it may provide more reliable predictions when calibration data
        is insufficient. The estimator assumes the quantile regression model can
        accurately capture the conditional quantiles of the target distribution.

        Args:
            X: Input features for training, shape (n_samples, n_features).
            y: Target values for training, shape (n_samples,).
            all_quantiles: Sorted list of quantile levels to estimate, in [0, 1].
            tuning_iterations: Number of hyperparameter search iterations.
            min_obs_for_tuning: Minimum samples required to trigger hyperparameter tuning.
            random_state: Random seed for reproducible model initialization.
            last_best_params: Warm-start parameters from previous hyperparameter search.

        Implementation Details:
            - Applies feature scaling if requested (fits scaler on all available data)
            - Uses hyperparameter tuning when sufficient data and iterations available
            - Falls back to default parameters for small datasets or when tuning disabled
            - Fits single quantile regression model for all required quantile levels
            - Sets conformalize_predictions flag to False for prediction behavior

        Usage Context:
            Automatically selected when dataset size < n_pre_conformal_trials, typically
            for exploratory analysis or when conformal calibration is not feasible due
            to data limitations. Users should be aware of the lack of coverage guarantees.
        """
        forced_param_configurations = []

        if last_best_params is not None:
            forced_param_configurations.append(last_best_params)

        estimator_config = ESTIMATOR_REGISTRY[self.quantile_estimator_architecture]
        default_params = deepcopy(estimator_config.default_params)
        if default_params:
            forced_param_configurations.append(default_params)

        if tuning_iterations > 1 and len(X) > min_obs_for_tuning:
            tuner = QuantileTuner(random_state=random_state, quantiles=all_quantiles)
            initialization_params = tuner.tune(
                X=X,
                y=y,
                estimator_architecture=self.quantile_estimator_architecture,
                n_searches=tuning_iterations,
                forced_param_configurations=forced_param_configurations,
            )
            self.last_best_params = initialization_params
        else:
            initialization_params = (
                forced_param_configurations[0] if forced_param_configurations else None
            )
            self.last_best_params = last_best_params

        self.quantile_estimator = initialize_estimator(
            estimator_architecture=self.quantile_estimator_architecture,
            initialization_params=initialization_params,
            random_state=random_state,
        )
        self.quantile_estimator.fit(X, y, quantiles=all_quantiles)

        # Store single estimator for compatibility with CV+ framework
        self.fold_estimators = [self.quantile_estimator]
        self.conformalize_predictions = False

    def _fit_cv_plus(
        self,
        X: np.ndarray,
        y: np.ndarray,
        all_quantiles: List[float],
        tuning_iterations: int,
        min_obs_for_tuning: int,
        random_state: Optional[int],
        last_best_params: Optional[dict],
    ):
        """Fit quantile conformal estimator using K-fold cross-validation calibration.

        For each fold, trains quantile estimator on the fold's training data and computes
        nonconformity scores on the fold's validation data. Stores all K fold estimators
        for use in prediction intervals.

        Args:
            X: Input features for training, shape (n_samples, n_features).
            y: Target values for training, shape (n_samples,).
            all_quantiles: Sorted list of quantile levels to estimate, in [0, 1].
            tuning_iterations: Number of hyperparameter search iterations per fold.
            min_obs_for_tuning: Minimum samples required to trigger hyperparameter tuning.
            random_state: Random seed for reproducible fold splits and model initialization.
            last_best_params: Warm-start parameters for quantile estimator hyperparameter search.
        """
        kfold = KFold(
            n_splits=self.n_calibration_folds, shuffle=True, random_state=random_state
        )

        # Store nonconformity scores and fold estimators for CV+
        fold_nonconformity_scores = [[] for _ in self.alphas]
        self.fold_estimators = []

        # Prepare forced parameter configurations for tuning
        forced_param_configurations = []
        if last_best_params is not None:
            forced_param_configurations.append(last_best_params)

        estimator_config = ESTIMATOR_REGISTRY[self.quantile_estimator_architecture]
        default_params = deepcopy(estimator_config.default_params)
        if default_params:
            forced_param_configurations.append(default_params)

        for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]

            # Fit quantile estimator on fold training data with tuning
            if tuning_iterations > 1 and len(X_fold_train) > min_obs_for_tuning:
                tuner = QuantileTuner(
                    random_state=random_state if random_state else None,
                    quantiles=all_quantiles,
                )
                fold_initialization_params = tuner.tune(
                    X=X_fold_train,
                    y=y_fold_train,
                    estimator_architecture=self.quantile_estimator_architecture,
                    n_searches=tuning_iterations,
                    forced_param_configurations=forced_param_configurations,
                )
            else:
                fold_initialization_params = (
                    forced_param_configurations[0]
                    if forced_param_configurations
                    else None
                )

            fold_estimator = initialize_estimator(
                estimator_architecture=self.quantile_estimator_architecture,
                initialization_params=fold_initialization_params,
                random_state=random_state if random_state else None,
            )
            fold_estimator.fit(X_fold_train, y_fold_train, quantiles=all_quantiles)

            # Store fold estimator for CV+
            self.fold_estimators.append(fold_estimator)

            # Compute nonconformity scores on validation fold
            val_prediction = fold_estimator.predict(X_fold_val)

            for i, alpha in enumerate(self.alphas):
                lower_quantile, upper_quantile = alpha_to_quantiles(alpha)
                lower_idx = self.quantile_indices[lower_quantile]
                upper_idx = self.quantile_indices[upper_quantile]

                # Symmetric nonconformity scores (CQR approach)
                lower_deviations = val_prediction[:, lower_idx] - y_fold_val
                upper_deviations = y_fold_val - val_prediction[:, upper_idx]
                fold_scores = np.maximum(lower_deviations, upper_deviations)
                fold_nonconformity_scores[i].append(fold_scores)

        # Store nonconformity scores as list of lists (one per alpha, containing fold arrays)
        self.nonconformity_scores = fold_nonconformity_scores

        # For CV+, we don't fit a final estimator on all data
        # Instead, we use the fold estimators for prediction
        self.last_best_params = last_best_params
        self.conformalize_predictions = True

    def _fit_train_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        all_quantiles: List[float],
        tuning_iterations: int,
        min_obs_for_tuning: int,
        random_state: Optional[int],
        last_best_params: Optional[dict],
    ):
        """Fit quantile conformal estimator using train-test split calibration.

        Implements the traditional split conformal prediction approach for quantile-based
        estimation using a single train-validation split. This method is computationally
        efficient for larger datasets where cross-validation becomes expensive, while
        maintaining finite-sample coverage guarantees through proper calibration.

        The input data is first split into training and validation sets. The quantile
        estimator is trained on the training set and validated on the separate validation
        set to compute nonconformity scores. Feature scaling is applied consistently
        across the split to prevent data leakage while ensuring proper normalization
        for the quantile regression model.

        Args:
            X: Input features for training, shape (n_samples, n_features).
            y: Target values for training, shape (n_samples,).
            all_quantiles: Sorted list of quantile levels to estimate, in [0, 1].
            tuning_iterations: Number of hyperparameter search iterations.
            min_obs_for_tuning: Minimum samples required to trigger hyperparameter tuning.
            random_state: Random seed for reproducible data splits and model initialization.
            last_best_params: Warm-start parameters for quantile estimator hyperparameter search.

        Implementation Details:
            - Fits feature scaler on training data only to prevent information leakage
            - Performs hyperparameter tuning on training set when data permits
            - Uses validation set exclusively for nonconformity score computation
            - Handles empty validation sets gracefully (falls back to non-conformal mode)
        """
        # Split data internally for train-test approach
        X_train, y_train, X_val, y_val = train_val_split(
            X,
            y,
            train_split=(1 - set_calibration_split(len(X))),
            normalize=False,  # Normalization already applied in fit()
            random_state=random_state,
        )

        forced_param_configurations = []

        if last_best_params is not None:
            forced_param_configurations.append(last_best_params)

        estimator_config = ESTIMATOR_REGISTRY[self.quantile_estimator_architecture]
        default_params = deepcopy(estimator_config.default_params)
        if default_params:
            forced_param_configurations.append(default_params)

        if tuning_iterations > 1 and len(X_train) > min_obs_for_tuning:
            tuner = QuantileTuner(random_state=random_state, quantiles=all_quantiles)
            initialization_params = tuner.tune(
                X=X_train,
                y=y_train,
                estimator_architecture=self.quantile_estimator_architecture,
                n_searches=tuning_iterations,
                forced_param_configurations=forced_param_configurations,
            )
            self.last_best_params = initialization_params
        else:
            initialization_params = (
                forced_param_configurations[0] if forced_param_configurations else None
            )
            self.last_best_params = last_best_params

        quantile_estimator = initialize_estimator(
            estimator_architecture=self.quantile_estimator_architecture,
            initialization_params=initialization_params,
            random_state=random_state,
        )
        quantile_estimator.fit(X_train, y_train, quantiles=all_quantiles)

        # Compute nonconformity scores on validation set if available
        if len(X_val) > 0:
            # Store single fold estimator for split conformal
            self.fold_estimators = [quantile_estimator]

            val_prediction = quantile_estimator.predict(X_val)
            fold_nonconformity_scores = [[] for _ in self.alphas]

            for i, alpha in enumerate(self.alphas):
                lower_quantile, upper_quantile = alpha_to_quantiles(alpha)
                lower_idx = self.quantile_indices[lower_quantile]
                upper_idx = self.quantile_indices[upper_quantile]

                # Symmetric nonconformity scores
                lower_deviations = val_prediction[:, lower_idx] - y_val
                upper_deviations = y_val - val_prediction[:, upper_idx]
                fold_scores = np.maximum(lower_deviations, upper_deviations)
                fold_nonconformity_scores[i].append(fold_scores)

            # Store as list of lists for consistency with CV+ structure
            self.nonconformity_scores = fold_nonconformity_scores

            self.conformalize_predictions = True
        else:
            self.conformalize_predictions = False

    def fit(
        self,
        X: np.array,
        y: np.array,
        tuning_iterations: Optional[int] = 0,
        min_obs_for_tuning: int = 50,
        random_state: Optional[int] = None,
        last_best_params: Optional[dict] = None,
    ):
        """Fit the quantile conformal estimator.

        Uses an adaptive data splitting strategy (K-fold cross-validation for small datasets,
        train-test split for larger datasets) or explicit strategy selection. Handles data
        preprocessing including feature scaling applied to the entire dataset.

        Args:
            X: Input features, shape (n_samples, n_features).
            y: Target values, shape (n_samples,).
            tuning_iterations: Hyperparameter search iterations (0 disables tuning).
            min_obs_for_tuning: Minimum samples required for hyperparameter tuning.
            random_state: Random seed for reproducible initialization.
            last_best_params: Warm-start parameters from previous fitting.
        """
        # Apply feature scaling to entire dataset if requested
        if self.normalize_features:
            self.feature_scaler = StandardScaler()
            X_scaled = self.feature_scaler.fit_transform(X)
        else:
            X_scaled = X
            self.feature_scaler = None

        all_quantiles = []
        for alpha in self.alphas:
            lower_quantile, upper_quantile = alpha_to_quantiles(alpha)
            all_quantiles.append(lower_quantile)
            all_quantiles.append(upper_quantile)
        all_quantiles = sorted(list(set(all_quantiles)))

        self.quantile_indices = {q: i for i, q in enumerate(all_quantiles)}

        total_size = len(X)
        use_conformal = total_size > self.n_pre_conformal_trials

        if use_conformal:
            strategy = self._determine_splitting_strategy(total_size)

            if strategy == "cv":
                self._fit_cv_plus(
                    X_scaled,
                    y,
                    all_quantiles,
                    tuning_iterations,
                    min_obs_for_tuning,
                    random_state,
                    last_best_params,
                )
            else:  # train_test_split
                self._fit_train_test_split(
                    X_scaled,
                    y,
                    all_quantiles,
                    tuning_iterations,
                    min_obs_for_tuning,
                    random_state,
                    last_best_params,
                )

        else:
            self._fit_non_conformal(
                X_scaled,
                y,
                all_quantiles,
                tuning_iterations,
                min_obs_for_tuning,
                random_state,
                last_best_params,
            )

    def predict_intervals(self, X: np.array) -> List[ConformalBounds]:
        """Generate conformal prediction intervals.

        Produces prediction intervals for each alpha level using the fold estimators
        and nonconformity scores computed during fitting.

        Args:
            X: Input features for prediction, shape (n_predict, n_features).

        Returns:
            List of ConformalBounds objects, one per alpha level, each containing:
                - lower_bounds: Lower interval bounds, shape (n_predict,)
                - upper_bounds: Upper interval bounds, shape (n_predict,)

        Raises:
            ValueError: If fold estimators have not been fitted.
        """
        if not self.fold_estimators:
            raise ValueError("Fold estimators must be fitted before prediction")

        # Apply same preprocessing as during training
        X_processed = X.copy()
        if self.normalize_features and self.feature_scaler is not None:
            X_processed = self.feature_scaler.transform(X_processed)

        intervals = []
        n_predict = X_processed.shape[0]

        # For CV+, we need to construct intervals using fold estimators
        for i, (alpha, alpha_adjusted) in enumerate(
            zip(self.alphas, self.updated_alphas)
        ):
            lower_quantile, upper_quantile = alpha_to_quantiles(alpha)
            lower_idx = self.quantile_indices[lower_quantile]
            upper_idx = self.quantile_indices[upper_quantile]

            if self.conformalize_predictions:
                # CV+ method: for each validation point i and corresponding fold k(i),
                # compute Q̂_{-S_{k(i)}}(x) ± R_i, then take quantiles

                # Collect all scores for this alpha level
                all_scores = []
                for fold_scores in self.nonconformity_scores[i]:
                    all_scores.extend(fold_scores)
                all_scores = np.array(all_scores)
                n_scores = len(all_scores)

                # Pre-allocate arrays for better performance
                lower_values = np.empty((n_scores, n_predict))
                upper_values = np.empty((n_scores, n_predict))

                score_idx = 0
                for fold_idx, fold_scores in enumerate(self.nonconformity_scores[i]):
                    fold_pred = self.fold_estimators[fold_idx].predict(X_processed)
                    n_fold_scores = len(fold_scores)

                    # Vectorized computation for all scores in this fold
                    fold_lower_pred = fold_pred[:, lower_idx]  # shape: (n_predict,)
                    fold_upper_pred = fold_pred[:, upper_idx]  # shape: (n_predict,)

                    # Broadcast operations
                    fold_scores_array = np.array(fold_scores).reshape(
                        -1, 1
                    )  # shape: (n_fold_scores, 1)

                    lower_values[score_idx : score_idx + n_fold_scores] = (
                        fold_lower_pred - fold_scores_array
                    )
                    upper_values[score_idx : score_idx + n_fold_scores] = (
                        fold_upper_pred + fold_scores_array
                    )

                    score_idx += n_fold_scores

                # Vectorized quantile computation
                quantile_factor = alpha_adjusted / (1 + 1 / n_scores)
                upper_quantile_factor = (1 - alpha_adjusted) / (1 + 1 / n_scores)

                lower_interval_bound = np.quantile(
                    lower_values, quantile_factor, axis=0, method="linear"
                )
                upper_interval_bound = np.quantile(
                    upper_values, upper_quantile_factor, axis=0, method="linear"
                )
            else:
                # Non-conformalized: use first fold estimator (or any single estimator)
                prediction = self.fold_estimators[0].predict(X_processed)
                lower_interval_bound = prediction[:, lower_idx]
                upper_interval_bound = prediction[:, upper_idx]

            intervals.append(
                ConformalBounds(
                    lower_bounds=lower_interval_bound, upper_bounds=upper_interval_bound
                )
            )

        return intervals

    def calculate_betas(self, X: np.array, y_true: float) -> list[float]:
        """Calculate empirical p-values (beta values) for conformity assessment.

        Computes alpha-specific empirical p-values representing the fraction of
        calibration nonconformity scores that are greater than or equal to the
        nonconformity score of a new observation.

        Args:
            X: Input features for single prediction, shape (n_features,).
            y_true: True target value for conformity assessment.

        Returns:
            List of beta values (empirical p-values), one per alpha level.
            Each beta ∈ [0, 1] represents the empirical quantile of the
            nonconformity score in the corresponding calibration distribution.
            Returns [0.5] * len(alphas) for non-conformalized mode.

        Raises:
            ValueError: If quantile estimator has not been fitted.
        """
        if self.fold_estimators == []:
            raise ValueError("Estimator must be fitted before calculating beta")

        # In non-conformalized mode, return neutral beta values since no calibration scores exist
        if not self.conformalize_predictions:
            return [0.5] * len(self.alphas)

        X_processed = X.reshape(1, -1)
        # Apply same preprocessing as during training
        if self.normalize_features and self.feature_scaler is not None:
            X_processed = self.feature_scaler.transform(X_processed)

        betas = []
        for i, alpha in enumerate(self.alphas):
            lower_quantile, upper_quantile = alpha_to_quantiles(alpha)
            lower_idx = self.quantile_indices[lower_quantile]
            upper_idx = self.quantile_indices[upper_quantile]

            # Compute average prediction across all fold estimators
            all_predictions = []
            for fold_estimator in self.fold_estimators:
                fold_pred = fold_estimator.predict(X_processed)
                all_predictions.append(fold_pred)

            avg_prediction = np.mean(all_predictions, axis=0)
            lower_bound = avg_prediction[0, lower_idx]
            upper_bound = avg_prediction[0, upper_idx]

            lower_deviation = lower_bound - y_true
            upper_deviation = y_true - upper_bound
            nonconformity = max(lower_deviation, upper_deviation)

            # Calculate beta using calibration scores from all folds for this alpha
            all_fold_scores = []
            for fold_scores in self.nonconformity_scores[i]:
                all_fold_scores.extend(fold_scores)
            beta = np.mean(np.array(all_fold_scores) >= nonconformity)

            betas.append(beta)

        return betas

    def update_alphas(self, new_alphas: List[float]):
        """Update coverage levels for the conformal estimator.

        Updates target coverage levels. Since the estimator uses the same fold
        estimators and nonconformity scores for all alpha levels, this operation
        is computationally efficient.

        Args:
            new_alphas: New miscoverage levels (1-alpha gives coverage).
                Must be in (0, 1) range.

        Important:
            If new_alphas require quantiles not computed during fit(), the estimator
            may need to be refitted. For maximum efficiency, determine the complete
            set of required alphas before calling fit().
        """
        self.updated_alphas = new_alphas.copy()
