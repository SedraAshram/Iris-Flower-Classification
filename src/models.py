"""Candidate classifiers and training helper for Iris species prediction."""

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def get_models(random_state=42):
    """Build fresh classifiers with preprocessing contained in each pipeline."""
    def scaled_pipeline(estimator):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", estimator),
        ])

    def tree_pipeline(estimator):
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", estimator),
        ])

    return {
        "logistic_regression": scaled_pipeline(
            LogisticRegression(max_iter=1000, random_state=random_state)
        ),
        "knn": scaled_pipeline(KNeighborsClassifier(n_neighbors=5)),
        "decision_tree": tree_pipeline(
            DecisionTreeClassifier(random_state=random_state, class_weight="balanced")
        ),
        "support_vector_machine": scaled_pipeline(SVC(random_state=random_state)),
    }


def train_models(models, X_train, y_train):
    """Fit each candidate using training data only and return fitted pipelines."""
    fitted_models = {}
    for name, model in models.items():
        fitted_models[name] = model.fit(X_train, y_train)
    return fitted_models
