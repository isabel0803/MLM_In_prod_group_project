from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
import numpy as np

class BaselineYieldRegressor(BaseEstimator, RegressorMixin):
    
    def __init__(self):
        self.target_mean_ = None

    def fit(self, X, y):
        # Learn the mean yield
        self.target_mean_ = np.mean(y)

        # required flag for sklearn checks
        self.is_fitted_ = True
        return self

    def predict(self, X):
        # Ensure the model has been fitted
        check_is_fitted(self, 'is_fitted_')

        # Return the mean yield for every sample
        return np.full(shape=(X.shape[0],), fill_value=self.target_mean_)