import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Building preprocessing pipeline that applies transformations to numerical and categorical features.
def build_preprocessor(df: pd.DataFrame):

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

# Numerical features are standardized to zero mean and unit variance
# Categorical features are one-hot encoded 
    # handle_unknown="ignore" ensures robustness during inference when previously unseen categories appear in the data
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
# Return fitted preprocessor and feature lists
    return preprocessor, num_cols, cat_cols

