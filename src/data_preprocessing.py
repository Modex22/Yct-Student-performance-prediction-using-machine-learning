"""Data loading and the shared preprocessing pipeline (scaling + one-hot encoding)."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, DATA_PATH, FEATURE_COLUMNS, NUMERIC_FEATURES


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the student dataset from disk, generating it first if missing."""
    import os

    if not os.path.exists(path):
        from data.generate_dataset import generate_dataset

        df = generate_dataset()
        df.to_csv(path, index=False)
        return df
    # keep_default_na=False: several categorical values (e.g. "None" for
    # parental_education) collide with pandas' default NA sentinel strings
    # and would otherwise be silently read in as NaN.
    return pd.read_csv(path, keep_default_na=False, na_values=[])


def build_preprocessor() -> ColumnTransformer:
    """A ColumnTransformer that scales numeric columns and one-hot-encodes categoricals."""
    numeric_pipeline = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ]
    )


def get_features(df: pd.DataFrame) -> pd.DataFrame:
    return df[FEATURE_COLUMNS].copy()
