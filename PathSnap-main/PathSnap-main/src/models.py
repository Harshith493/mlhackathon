"""
models.py - Definitions and wrappers for the 5 machine learning models
across distinct algorithm families:
1. Logistic Regression (Linear & Probabilistic)
2. K-Nearest Neighbors (Instance-Based)
3. Decision Tree (Tree-Based)
4. Random Forest (Ensemble)
5. HistGradientBoosting (Boosting)
"""

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


def get_model(model_name, **kwargs):
    """
    Factory function returning the instantiated model pipeline.
    """
    if model_name == "logistic_regression":
        # Linear & Probabilistic
        params = {"C": 1.0, "max_iter": 500, "solver": "lbfgs", "random_state": 42}
        params.update(kwargs)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**params)),
        ])

    elif model_name == "knn":
        # Instance-Based
        params = {"n_neighbors": 15, "weights": "distance", "metric": "minkowski", "p": 2, "n_jobs": -1}
        params.update(kwargs)
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(**params)),
        ])

    elif model_name == "decision_tree":
        # Tree-Based
        params = {"criterion": "gini", "max_depth": 12, "min_samples_split": 10, "min_samples_leaf": 5, "random_state": 42}
        params.update(kwargs)
        return DecisionTreeClassifier(**params)

    elif model_name == "random_forest":
        # Ensemble
        params = {
            "n_estimators": 100,
            "max_depth": 15,
            "min_samples_split": 8,
            "min_samples_leaf": 4,
            "n_jobs": -1,
            "random_state": 42,
        }
        params.update(kwargs)
        return RandomForestClassifier(**params)

    elif model_name == "gradient_boosting":
        # Boosting
        params = {
            "learning_rate": 0.08,
            "max_iter": 150,
            "max_depth": 8,
            "min_samples_leaf": 20,
            "random_state": 42,
        }
        params.update(kwargs)
        return HistGradientBoostingClassifier(**params)

    else:
        raise ValueError(f"Unknown model name: {model_name}")


MODEL_REGISTRY = {
    "Model 1 (Logistic Regression)": {
        "key": "logistic_regression",
        "family": "Linear & Probabilistic",
        "desc": "Regularized L2 Logistic Regression with standardized features and logit link",
    },
    "Model 2 (K-Nearest Neighbors)": {
        "key": "knn",
        "family": "Instance-based",
        "desc": "Distance-weighted KNN (k=15) operating in standardized metric feature space",
    },
    "Model 3 (Decision Tree)": {
        "key": "decision_tree",
        "family": "Tree-based",
        "desc": "Depth-constrained Decision Tree with Gini impurity split criterion",
    },
    "Model 4 (Random Forest)": {
        "key": "random_forest",
        "family": "Ensemble",
        "desc": "Random Forest of 100 bootstrap trees with randomized feature subspacing",
    },
    "Model 5 (HistGradientBoosting)": {
        "key": "gradient_boosting",
        "family": "Boosting",
        "desc": "Histogram-based Gradient Boosted Trees optimizing binary cross-entropy",
    },
}

