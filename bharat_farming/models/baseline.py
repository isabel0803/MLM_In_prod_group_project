import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class GroupMeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.group_means_ = {}
        self.global_mean_ = None

    def fit(self, X, y):
        X_df = pd.DataFrame(X)
        first_col = X_df.iloc[:, 0]

        self.group_means_ = y.groupby(first_col).mean().to_dict()
        self.global_mean_ = y.mean()
        return self

    def predict(self, X):
        check_is_fitted(self)
        X_df = pd.DataFrame(X)
        first_col = X_df.iloc[:, 0]

        return first_col.map(self.group_means_).fillna(self.global_mean_).values