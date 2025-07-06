
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class VotingModel(BaseEstimator, RegressorMixin):
    """
    A custom ensemble model that performs voting aggregation for predictions.

    Parameters:
    ----------
    estimators : list
        List of estimators (models) to be used for voting aggregation.

    Methods:
    --------
    fit(X, y=None):
        Fit the ensemble model. This method does nothing as it's not required for voting aggregation.

    predict(X):
        Perform prediction using voting aggregation on the provided features.

    predict_proba(X):
        Perform prediction with probabilities using voting aggregation on the provided features.

    """
    def __init__(self, estimators, cat_cols=None):
        """
        Initialize the VotingModel.

        Parameters:
        ----------
        estimators : list
            List of estimators (models) to be used for voting aggregation.
        cat_cols : list, optional
            List of categorical column names, by default None
        """
        super().__init__()
        self.estimators = estimators
        self.cat_cols = cat_cols
        
    def fit(self, X, y=None):
        """
        Fit the ensemble model.

        This method does nothing as it's not required for voting aggregation.

        Parameters:
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,) (default=None)
            Target values.

        Returns:
        --------
        self : object
            Returns self.
        """
        return self
    
    def predict(self, X):
        """
        Perform prediction using voting aggregation on the provided features.

        Parameters:
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Features to perform prediction on.

        Returns:
        --------
        y_pred : array-like, shape (n_samples,)
            Predicted target values.
        """
        y_preds = [estimator.predict(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)
     
    def predict_proba(self, X):
        """
        Perform prediction with probabilities using voting aggregation on the provided features.

        Parameters:
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            Features to perform prediction on.

        Returns:
        --------
        y_pred_proba : array-like, shape (n_samples, n_classes)
            Predicted probabilities.
        """
        # Assuming the first 5 estimators are LightGBM and the rest are CatBoost
        # This needs to be more robust if the order or types of models change
        lgb_preds = [estimator.predict_proba(X) for estimator in self.estimators[:5]]
        
        # CatBoost models might need categorical features as strings
        if self.cat_cols and not X[self.cat_cols].empty:
            X_cat_str = X.copy()
            X_cat_str[self.cat_cols] = X_cat_str[self.cat_cols].astype(str)
            cat_preds = [estimator.predict_proba(X_cat_str) for estimator in self.estimators[5:]]
        else:
            cat_preds = [estimator.predict_proba(X) for estimator in self.estimators[5:]]

        y_preds = lgb_preds + cat_preds
        
        return np.mean(y_preds, axis=0)


