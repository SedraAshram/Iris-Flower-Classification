import unittest

import pandas as pd

from src.data_loader import FEATURE_COLUMNS, TARGET_COLUMN, load_iris_data
from src.evaluation import evaluate_models
from src.models import get_models, train_models
from src.preprocessing import clean_data, split_data


class IrisPipelineTests(unittest.TestCase):
    def test_loader_returns_expected_schema_and_classes(self):
        data = load_iris_data()
        self.assertEqual(list(data.columns), FEATURE_COLUMNS + [TARGET_COLUMN])
        self.assertEqual(len(data), 150)
        self.assertEqual(data[TARGET_COLUMN].nunique(), 3)
        self.assertEqual(data[TARGET_COLUMN].value_counts().min(), 50)

    def test_validation_retains_repeated_observations_and_reports_them(self):
        data = load_iris_data()
        validated, audit = clean_data(data)
        self.assertEqual(len(validated), 150)
        self.assertEqual(audit["duplicate_rows_retained"], 3)
        self.assertEqual(audit["missing_values"], 0)

    def test_validation_rejects_missing_or_implausible_measurements(self):
        data = load_iris_data()
        data.loc[0, "sepal_length_cm"] = None
        with self.assertRaises(ValueError):
            clean_data(data)
        data = load_iris_data()
        data.loc[0, "sepal_length_cm"] = 11
        with self.assertRaises(ValueError):
            clean_data(data)

    def test_split_is_stratified_and_reproducible(self):
        data, _ = clean_data(load_iris_data())
        first = split_data(data)
        second = split_data(data)
        self.assertEqual(first[0].index.tolist(), second[0].index.tolist())
        self.assertEqual(first[1].index.tolist(), second[1].index.tolist())
        train_counts = first[2].value_counts().sort_index()
        test_counts = first[3].value_counts().sort_index()
        self.assertEqual(train_counts.index.tolist(), test_counts.index.tolist())
        proportions = train_counts / (train_counts + test_counts)
        self.assertTrue(((proportions - 0.8).abs() <= 0.05).all())

    def test_candidate_models_train_from_training_partition(self):
        data, _ = clean_data(load_iris_data())
        X_train, _, y_train, _ = split_data(data)
        models = get_models()
        self.assertEqual(
            set(models),
            {"logistic_regression", "knn", "decision_tree", "support_vector_machine"},
        )
        fitted = train_models(models, X_train, y_train)
        self.assertEqual(set(fitted), set(models))
        for model in fitted.values():
            self.assertEqual(set(model.classes_), set(y_train.unique()))

    def test_evaluation_returns_test_metrics_and_confusion_counts(self):
        data, _ = clean_data(load_iris_data())
        X_train, X_test, y_train, y_test = split_data(data)
        results, _ = evaluate_models(get_models(), X_train, X_test, y_train, y_test)
        required_metrics = {
            "accuracy", "precision_macro", "recall_macro", "f1_macro",
            "cv_accuracy_mean", "cv_accuracy_std", "cv_f1_macro_mean",
        }
        self.assertEqual(len(results), 4)
        for result in results.values():
            self.assertTrue(required_metrics.issubset(result))
            self.assertEqual(sum(map(sum, result["confusion_matrix"])), len(y_test))


if __name__ == "__main__":
    unittest.main()
