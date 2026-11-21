"""Small, reusable exploratory summaries and plots for Iris data."""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from .data_loader import FEATURE_COLUMNS, TARGET_COLUMN


def summarize_data(data):
    """Return a concise, serializable overview of the dataset."""
    return {
        "shape": list(data.shape),
        "missing_values": {key: int(value) for key, value in data.isna().sum().items()},
        "class_counts": {
            str(key): int(value) for key, value in data[TARGET_COLUMN].value_counts().items()
        },
        "duplicate_rows": int(data.duplicated().sum()),
        "feature_summary": data[FEATURE_COLUMNS].describe().round(3).to_dict(),
        "correlation": data[FEATURE_COLUMNS].corr().round(3).to_dict(),
    }


def save_pairplot(data, output_path):
    """Save feature pair plots colored by species and close the figure."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid = sns.pairplot(data, vars=FEATURE_COLUMNS, hue=TARGET_COLUMN, corner=True)
    grid.fig.suptitle("Iris feature relationships", y=1.02)
    grid.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(grid.fig)
    return output_path
