"""
Model factory for instantiating classifiers by family name.

Decouples model creation from training logic, enabling configuration-driven
model selection and seamless addition of new model families.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Supported model families
_SUPPORTED_FAMILIES = {
    "lightgbm",
    "xgboost",
    "catboost",
    "logistic_regression",
    "decision_tree",
}


def create_model(model_family: str, params: dict[str, Any] | None = None) -> Any:
    """Instantiate a classifier given a model family name and hyperparameters.

    Parameters
    ----------
    model_family : str
        One of ``"lightgbm"``, ``"xgboost"``, ``"catboost"``,
        ``"logistic_regression"``, ``"decision_tree"``.
    params : dict | None
        Hyperparameter dictionary passed to the constructor.

    Returns
    -------
    BaseEstimator
        An unfitted scikit-learn-compatible classifier.

    Raises
    ------
    ValueError
        If *model_family* is not recognised.
    """
    if params is None:
        params = {}

    family = model_family.lower().strip()

    if family == "lightgbm":
        from lightgbm import LGBMClassifier

        defaults = {"random_state": 42, "n_jobs": -1, "verbose": -1}
        defaults.update(params)
        return LGBMClassifier(**defaults)

    if family == "xgboost":
        from xgboost import XGBClassifier

        defaults = {
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
            "tree_method": "hist",
            "eval_metric": "mlogloss",
            "objective": "multi:softprob",
            "num_class": 5,
        }
        defaults.update(params)
        return XGBClassifier(**defaults)

    if family == "catboost":
        from catboost import CatBoostClassifier

        defaults = {
            "random_seed": params.pop("random_state", 42),
            "verbose": 0,
            "allow_writing_files": False,
            "loss_function": "MultiClass",
            "bootstrap_type": "Bernoulli",
        }
        defaults.update(params)
        return CatBoostClassifier(**defaults)

    if family == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        defaults = {"max_iter": 1000, "random_state": 42, "solver": "lbfgs"}
        defaults.update(params)
        return LogisticRegression(**defaults)

    if family == "decision_tree":
        from sklearn.tree import DecisionTreeClassifier

        defaults = {"random_state": 42}
        defaults.update(params)
        return DecisionTreeClassifier(**defaults)

    raise ValueError(
        f"Unknown model family '{model_family}'. "
        f"Supported: {sorted(_SUPPORTED_FAMILIES)}"
    )
