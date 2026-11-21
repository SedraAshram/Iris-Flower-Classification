"""Reproducible EDA for the official UCI Iris data file.

Run from the project root with: python notebooks/01_exploratory_data_analysis.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "uci_iris" / "iris.data"
OUTPUT_DIR = PROJECT_ROOT / "results" / "eda"
FEATURES = [
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
]
TARGET = "species"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATA_PATH, header=None, names=FEATURES + [TARGET])
    sns.set_theme(style="whitegrid", palette="colorblind")

    print("Dataset shape:", data.shape)
    print("\nData types:\n", data.dtypes)
    print("\nMissing values:\n", data.isna().sum())
    print("\nExact duplicate rows:", int(data.duplicated().sum()))
    print("\nClass counts:\n", data[TARGET].value_counts().sort_index())
    print("\nFeature summary (cm):\n", data[FEATURES].describe().round(2))
    print("\nFeature means by species (cm):\n",
          data.groupby(TARGET, observed=True)[FEATURES].mean().round(2))
    print("\nFeature correlations:\n", data[FEATURES].corr().round(2))

    axes = data[FEATURES].hist(bins=15, figsize=(10, 7), edgecolor="white")
    plt.suptitle("Iris feature distributions", y=1.02)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_distributions.png", dpi=160, bbox_inches="tight")
    plt.close()

    long_data = data.melt(id_vars=TARGET, var_name="feature", value_name="measurement_cm")
    grid = sns.catplot(
        data=long_data, x=TARGET, y="measurement_cm", col="feature",
        kind="box", col_wrap=2, sharey=False, height=3.5, aspect=1.25,
    )
    grid.set_xticklabels(rotation=25, ha="right")
    grid.set_axis_labels("Species", "Measurement (cm)")
    grid.fig.suptitle("Feature measurements by species", y=1.03)
    grid.savefig(OUTPUT_DIR / "feature_by_species.png", dpi=160, bbox_inches="tight")
    plt.close(grid.fig)

    pair_grid = sns.pairplot(data, vars=FEATURES, hue=TARGET, corner=True, diag_kind="hist")
    pair_grid.fig.suptitle("Pairwise feature relationships by species", y=1.02)
    pair_grid.savefig(OUTPUT_DIR / "pairplot_by_species.png", dpi=160, bbox_inches="tight")
    plt.close(pair_grid.fig)

    plt.figure(figsize=(7, 5))
    sns.heatmap(data[FEATURES].corr(), annot=True, fmt=".2f", cmap="vlag", center=0,
                square=True, linewidths=0.5)
    plt.title("Correlation between Iris measurements")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_correlations.png", dpi=160, bbox_inches="tight")
    plt.close()
    print(f"\nEDA plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
