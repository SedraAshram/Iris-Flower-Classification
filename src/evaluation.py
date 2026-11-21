"""Cross-validate candidate classifiers and evaluate them on held-out data."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_validate


def evaluate_models(models, X_train, X_test, y_train, y_test, random_state=42):
    """Compare models with train-only CV and return test metrics and estimators."""
    results = {}
    fitted_models = {}
    labels = sorted(y_test.astype(str).unique())
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    for name, model in models.items():
        cv_scores = cross_validate(
            model,
            X_train,
            y_train,
            cv=cv,
            scoring={"accuracy": "accuracy", "f1_macro": "f1_macro"},
            return_train_score=False,
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        report = classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        )
        results[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision_macro": float(report["macro avg"]["precision"]),
            "recall_macro": float(report["macro avg"]["recall"]),
            "f1_macro": float(report["macro avg"]["f1-score"]),
            "precision_weighted": float(report["weighted avg"]["precision"]),
            "recall_weighted": float(report["weighted avg"]["recall"]),
            "f1_weighted": float(report["weighted avg"]["f1-score"]),
            "cv_accuracy_mean": float(cv_scores["test_accuracy"].mean()),
            "cv_accuracy_std": float(cv_scores["test_accuracy"].std()),
            "cv_f1_macro_mean": float(cv_scores["test_f1_macro"].mean()),
            "cv_f1_macro_std": float(cv_scores["test_f1_macro"].std()),
            "confusion_matrix": confusion_matrix(
                y_test.astype(str), predictions.astype(str), labels=labels
            ).tolist(),
            "labels": labels,
            "classification_report": report,
        }
        fitted_models[name] = model
    return results, fitted_models


def save_confusion_matrices(results, output_dir):
    """Save one labeled confusion-matrix heatmap per model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, result in results.items():
        matrix = pd.DataFrame(
            result["confusion_matrix"],
            index=result["labels"],
            columns=result["labels"],
        )
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set(title=f"{name.replace('_', ' ').title()} confusion matrix",
               xlabel="Predicted species", ylabel="True species")
        fig.tight_layout()
        fig.savefig(output_dir / f"{name}_confusion_matrix.png", dpi=160)
        plt.close(fig)
