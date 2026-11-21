"""Load the official UCI Iris file into the project's shared interface."""

from pathlib import Path

import pandas as pd

FEATURE_COLUMNS = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
]
TARGET_COLUMN = "species"
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "uci_iris" / "iris.data"


def load_iris_data() -> pd.DataFrame:
    """Read the locally stored, headerless UCI CSV using its documented schema."""
    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"UCI Iris data not found at {DATA_PATH}. See data/uci_iris/README.md."
        )
    data = pd.read_csv(DATA_PATH, header=None, names=FEATURE_COLUMNS + [TARGET_COLUMN])
    return data
