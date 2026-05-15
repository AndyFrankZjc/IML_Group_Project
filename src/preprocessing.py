import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold


def remove_high_missing_columns(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    threshold: float = 0.70
):
    missing_rate = X_train.isna().mean()
    keep_cols = missing_rate[missing_rate <= threshold].index.tolist()

    return X_train[keep_cols], X_val[keep_cols], X_test[keep_cols], keep_cols


def make_onehot_encoder():
    """
    Compatible with scikit-learn 1.3 and older/newer versions.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(X_train: pd.DataFrame):
    categorical_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X_train.select_dtypes(exclude=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_onehot_encoder())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols)
        ],
        remainder="drop"
    )

    return preprocessor, numeric_cols, categorical_cols


def preprocess_for_nn(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    missing_threshold: float = 0.70
):
    X_train, X_val, X_test, keep_cols = remove_high_missing_columns(
        X_train, X_val, X_test, threshold=missing_threshold
    )

    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    selector = VarianceThreshold(threshold=0.0)
    X_train_processed = selector.fit_transform(X_train_processed)
    X_val_processed = selector.transform(X_val_processed)
    X_test_processed = selector.transform(X_test_processed)

    metadata = {
        "kept_original_columns": keep_cols,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "n_features_after_variance_filter": int(X_train_processed.shape[1])
    }

    return X_train_processed, X_val_processed, X_test_processed, preprocessor, selector, metadata
