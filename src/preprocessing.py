"""Prepare the shared dataframe for model training and evaluation."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .data_loader import FEATURE_COLUMNS, TARGET_COLUMN


def clean_data(data):
    """Validate Iris measurements and labels without discarding valid samples.

    Exact repeated rows are retained: the source represents 150 observations,
    and identical measurements do not prove that two flower records are the
    same observation.
    """
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]).difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    cleaned = data[FEATURE_COLUMNS + [TARGET_COLUMN]].copy()
    for column in FEATURE_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    for column in FEATURE_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="raise")
    if cleaned.isna().any().any():
        missing = cleaned.isna().sum()
        raise ValueError(f"Missing values found: {missing[missing > 0].to_dict()}")
    if not np.isfinite(cleaned[FEATURE_COLUMNS].to_numpy()).all():
        raise ValueError("Features contain non-finite values.")
    invalid = (cleaned[FEATURE_COLUMNS] <= 0) | (cleaned[FEATURE_COLUMNS] > 10)
    if invalid.any().any():
        bad_columns = invalid.columns[invalid.any()].tolist()
        raise ValueError(f"Measurements outside the plausible (0, 10] cm range: {bad_columns}")
    expected_labels = {"Iris-setosa", "Iris-versicolor", "Iris-virginica"}
    unexpected = set(cleaned[TARGET_COLUMN].dropna().unique()) - expected_labels
    if unexpected:
        raise ValueError(f"Unexpected Iris species labels: {sorted(unexpected)}")
    duplicate_rows = int(cleaned.duplicated().sum())
    audit = {
        "rows_before": len(cleaned),
        "rows_after": len(cleaned),
        "missing_values": int(cleaned.isna().sum().sum()),
        "duplicate_rows_retained": duplicate_rows,
        "measurements_within_plausible_range": True,
    }
    return cleaned, audit


def split_data(data, test_size=0.2, random_state=42):
    """Split features and labels into stratified train and test partitions."""
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]).difference(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
