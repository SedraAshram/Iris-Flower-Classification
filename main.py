"""Run the Iris data exploration, model comparison, and evaluation workflow."""

import json
from pathlib import Path

import pandas as pd

from src.data_loader import load_iris_data
from src.eda import save_pairplot, summarize_data
from src.evaluation import evaluate_models, save_confusion_matrices
from src.models import get_models
from src.preprocessing import clean_data, split_data


def main():
    output_dir = Path("results")
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_data = load_iris_data()
    data, cleaning_audit = clean_data(raw_data)
    X_train, X_test, y_train, y_test = split_data(data)
    summary = summarize_data(data)
    summary["cleaning_audit"] = cleaning_audit
    metrics, _ = evaluate_models(get_models(), X_train, X_test, y_train, y_test)

    save_pairplot(data, output_dir / "eda" / "pairplot_by_species.png")
    save_confusion_matrices(metrics, output_dir / "confusion_matrices")
    with (output_dir / "model_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    comparison_columns = [
        "cv_accuracy_mean", "cv_accuracy_std", "cv_f1_macro_mean",
        "cv_f1_macro_std", "accuracy", "precision_macro", "recall_macro",
        "f1_macro", "precision_weighted", "recall_weighted", "f1_weighted",
    ]
    comparison = pd.DataFrame.from_dict(metrics, orient="index")[comparison_columns]
    comparison.index.name = "model"
    comparison = comparison.sort_values(
        ["cv_f1_macro_mean", "cv_accuracy_mean"], ascending=False
    )
    comparison.to_csv(output_dir / "model_comparison.csv", float_format="%.4f")
    best_model = comparison.index[0]
    summary["model_selection"] = {
        "selected_model": best_model,
        "selection_basis": "Highest mean five-fold training CV macro F1; test set was held out.",
    }
    with (output_dir / "data_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    for name, result in metrics.items():
        print(
            f"{name}: CV macro-F1={result['cv_f1_macro_mean']:.3f} "
            f"(std={result['cv_f1_macro_std']:.3f}); "
            f"test accuracy={result['accuracy']:.3f}; "
            f"test macro-F1={result['f1_macro']:.3f}"
        )
    print(f"Selected by training CV macro-F1: {best_model}")


if __name__ == "__main__":
    main()
