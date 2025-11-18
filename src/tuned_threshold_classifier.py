import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import (
    get_scorer, balanced_accuracy_score, f1_score, 
    accuracy_score
)
from scipy.optimize import minimize
from copy import deepcopy


class TunedThresholdClassifierCV:
    """
    A generalized threshold tuning classifier that works for both binary and multi-class classification.
    Uses cross-validation to tune thresholds based on a specified scoring metric.
    For multi-class problems, all thresholds are tuned simultaneously.
    
    Parameters
    ----------
    estimator : object
        A scikit-learn compatible classifier that implements predict_proba.
    scoring : str or callable, default='balanced_accuracy'
        Scoring metric to optimize. Can be any sklearn scorer string or callable.
    cv : int, cross-validation generator or iterable, default=5
        Cross-validation splitting strategy.
    n_jobs : int, default=None
        Number of jobs to run in parallel during cross-validation.
    random_state : int, default=None
        Random state for reproducibility.
    """
    
    def __init__(self, estimator, scoring='balanced_accuracy', cv=5, n_jobs=None, random_state=None):
        self.estimator = estimator
        self.scoring = scoring
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.best_threshold_ = None
        self.best_thresholds_ = None  # For multi-class
        self.classes_ = None
        self.n_classes_ = None
        self.scorer_ = None
        self._score_func = None  # Direct metric function for scoring
        self._original_estimator = None  # Store original unfitted estimator
        self._cached_probas = None  # Cache probabilities for faster optimization
        self._cached_y_val = None  # Cache validation labels
        
    def _predict_with_thresholds(self, y_proba, thresholds):
        """
        Predict classes using probability thresholds.
        
        Parameters
        ----------
        y_proba : array-like of shape (n_samples, n_classes)
            Predicted probabilities for each class.
        thresholds : array-like
            Thresholds for each class. For binary, single threshold.
            For multi-class, array of thresholds.
            
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.
        """
        y_proba = np.asarray(y_proba)
        thresholds = np.asarray(thresholds)
        
        if self.n_classes_ == 2:
            # Binary classification: predict class 1 if prob >= threshold, else class 0
            y_pred = (y_proba[:, 1] >= thresholds[0]).astype(int)
            # Map to actual class labels
            y_pred = self.classes_[y_pred]
        else:
            # Multi-class: normalize probabilities by thresholds and predict class with highest normalized probability
            # This approach allows thresholds to adjust the decision boundaries for all classes simultaneously
            # Avoid division by zero by adding small epsilon
            epsilon = 1e-10
            normalized_proba = y_proba / (thresholds + epsilon)
            # Predict the class with highest normalized probability
            y_pred = np.argmax(normalized_proba, axis=1)
            # Map to actual class labels
            y_pred = self.classes_[y_pred]
        
        return y_pred
    
    def _objective_function(self, thresholds, X, y, cv_splits, cached_probas):
        """
        Objective function to minimize (negative of scoring metric).
        Uses cached probabilities if available to avoid refitting models.
        
        Parameters
        ----------
        thresholds : array-like
            Threshold values to evaluate.
        X : array-like
            Feature matrix (only used if cached_probas is None).
        y : array-like
            True labels.
        cv_splits : list
            List of (train_idx, test_idx) tuples for cross-validation.
        cached_probas : list
            Pre-computed probabilities for each CV fold.
            
        Returns
        -------
        score : float
            Negative of the cross-validated score (for minimization).
        """
        # Ensure thresholds are in valid range [0, 1]
        thresholds = np.clip(thresholds, 0.0, 1.0)
        
        scores = []
        
        # Use cached probabilities if available (much faster!)
        # Since the probability model is fixed, we just apply different thresholds
        # to the pre-computed probabilities without refitting models
        for i, (train_idx, val_idx) in enumerate(cv_splits):
            y_val_cv = y[val_idx]
            y_proba_val = cached_probas[i]  # Use cached probabilities
            
            # Predict using thresholds (only thresholds change, not probabilities)
            y_pred_val = self._predict_with_thresholds(y_proba_val, thresholds)
            
            # Calculate score
            if self._score_func is not None:
                score = self._score_func(y_val_cv, y_pred_val)
            else:
                score = accuracy_score(y_val_cv, y_pred_val)  # Fallback
            scores.append(score)
        
        # Return negative score for minimization
        return -np.mean(scores)
    
    def fit(self, X, y):
        """
        Fit the classifier and tune thresholds using cross-validation.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Training labels.
            
        Returns
        -------
        self : object
            Returns self.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Store the original unfitted estimator for cloning in CV
        self._original_estimator = clone(self.estimator)
        
        # Fit the base classifier on all data (for final predictions)
        self.estimator.fit(X, y)
        
        # Get classes and number of classes
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # Get scorer and create a mapping for direct metric calls
        self.scorer_ = get_scorer(self.scoring)
        # Map common scoring strings to metric functions for direct calls
        scoring_to_func = {
            'balanced_accuracy': balanced_accuracy_score,
            'f1_macro': lambda y_true, y_pred: f1_score(y_true, y_pred, average='macro'),
        }
        scoring_str = self.scoring if isinstance(self.scoring, str) else str(self.scoring)
        self._score_func = scoring_to_func.get(scoring_str, None)
        
        # Prepare cross-validation splits
        if isinstance(self.cv, int):
            cv = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        else:
            cv = self.cv
        
        cv_splits = list(cv.split(X, y))
        
        # Pre-compute probabilities for all CV folds (major speedup!)
        # Key insight: The probability model stays unchanged during threshold optimization.
        # Only the thresholds change, so we can fit models once per CV fold and cache
        # the probabilities. During optimization, we just apply different thresholds to
        # these cached probabilities without refitting any models.
        cached_probas = []
        for train_idx, val_idx in cv_splits:
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv = y[train_idx]
            
            # Fit classifier on training fold (done once per fold)
            try:
                estimator_copy = clone(self._original_estimator)
            except Exception:
                try:
                    estimator_copy = clone(self.estimator)
                except Exception:
                    estimator_copy = deepcopy(self._original_estimator if self._original_estimator is not None else self.estimator)
            estimator_copy.fit(X_train_cv, y_train_cv)
            
            # Cache probabilities for validation fold (these stay fixed during optimization)
            y_proba_val = estimator_copy.predict_proba(X_val_cv)
            cached_probas.append(y_proba_val)
        
        # Initialize thresholds
        if self.n_classes_ == 2:
            # Binary: optimize single threshold
            initial_threshold = np.array([0.5])
            bounds = [(0.0, 1.0)]
        else:
            # Multi-class: optimize thresholds for all classes simultaneously
            initial_threshold = np.full(self.n_classes_, 1.0 / self.n_classes_)
            bounds = [(0.0, 1.0)] * self.n_classes_
        
        # Optimize thresholds using scipy.optimize
        # First evaluate the initial threshold to ensure the objective function works
        try:
            initial_score = -self._objective_function(initial_threshold, X, y, cv_splits, cached_probas=cached_probas)
        except Exception as e:
            # If objective function fails, use default thresholds
            if self.n_classes_ == 2:
                self.best_threshold_ = 0.5
            else:
                self.best_thresholds_ = np.full(self.n_classes_, 1.0 / self.n_classes_)
            return self
        
        # Use a single, efficient optimization method
        # Nelder-Mead is good for threshold optimization and doesn't require gradients
        best_score = initial_score
        best_thresholds = initial_threshold.copy()
        
        try:
            result = minimize(
                self._objective_function,
                initial_threshold,
                args=(X, y, cv_splits, cached_probas),  # Pass cached probabilities
                method='Nelder-Mead',
                options={'maxiter': 50, 'xatol': 1e-3, 'fatol': 1e-3}  # Reduced iterations and tolerance
            )
            if result.success:
                score_result = -result.fun
                if score_result > best_score:
                    best_score = score_result
                    best_thresholds = np.clip(result.x, 0.0, 1.0)
        except Exception:
            pass
        
        # Ensure thresholds are in valid range
        best_thresholds = np.clip(best_thresholds, 0.0, 1.0)
        
        # Store best thresholds
        if self.n_classes_ == 2:
            self.best_threshold_ = float(best_thresholds[0])
        else:
            self.best_thresholds_ = best_thresholds
        
        return self
    
    def predict(self, X):
        """
        Predict class labels using tuned thresholds.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test features.
            
        Returns
        -------
        y_pred : array-like of shape (n_samples,)
            Predicted class labels.
        """
        if self.best_threshold_ is None and self.best_thresholds_ is None:
            raise ValueError("Classifier has not been fitted yet. Call fit() first.")
        
        # Convert to numpy array to avoid feature name warnings (consistent with fit)
        X = np.asarray(X)
        y_proba = self.estimator.predict_proba(X)
        
        if self.n_classes_ == 2:
            thresholds = np.array([self.best_threshold_])
        else:
            thresholds = self.best_thresholds_
        
        return self._predict_with_thresholds(y_proba, thresholds)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test features.
            
        Returns
        -------
        y_proba : array-like of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        # Convert to numpy array to avoid feature name warnings (consistent with fit)
        X = np.asarray(X)
        return self.estimator.predict_proba(X)
