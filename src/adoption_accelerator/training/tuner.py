"""
Optuna-based hyperparameter tuning utilities.

Provides search space definitions, study creation, trial extraction,
and study persistence for reproducible Bayesian optimization.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from adoption_accelerator.training.evaluation import (
    optimize_thresholds,
)
from adoption_accelerator.training.model_factory import create_model

logger = logging.getLogger(__name__)

# ── Early stopping defaults ──────────────────────────────────────────

EARLY_STOPPING_ROUNDS = 30
"""Default patience for early stopping (boosting families only)."""

MAX_ITERATIONS = 2000
"""Maximum boosting rounds; early stopping selects the actual count."""


# ── Search space definitions ────────────────────────────────────────


def get_search_space(model_family: str) -> dict[str, Any]:
    """Return the hyperparameter search space definition.

    Parameters
    ----------
    model_family : str
        ``"lightgbm"``, ``"xgboost"``, or ``"catboost"``.

    Returns
    -------
    dict
        Keys map parameter names to dicts with ``type``, ``low``,
        ``high``, ``log``, or ``choices``.

    Notes
    -----
    ``n_estimators`` / ``iterations`` are excluded from search spaces
    because native early stopping determines the optimal boosting
    rounds automatically.  A high cap (:data:`MAX_ITERATIONS`) is
    injected in the objective function.
    """
    family = model_family.lower().strip()

    if family == "lightgbm":
        return {
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "max_depth": {"type": "categorical", "choices": [-1, 3, 5, 7, 9]},
            "num_leaves": {"type": "int", "low": 15, "high": 255},
            "min_child_samples": {"type": "int", "low": 5, "high": 100},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.3, "high": 1.0},
            "reg_alpha": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
            "reg_lambda": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
            "min_split_gain": {"type": "float", "low": 0.0, "high": 1.0},
        }

    if family == "xgboost":
        return {
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "max_depth": {"type": "int", "low": 3, "high": 8},
            "min_child_weight": {"type": "int", "low": 1, "high": 50},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
            "colsample_bytree": {"type": "float", "low": 0.3, "high": 1.0},
            "gamma": {"type": "float", "low": 0.0, "high": 5.0},
            "reg_alpha": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
            "reg_lambda": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
        }

    if family == "catboost":
        return {
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.3, "log": True},
            "depth": {"type": "int", "low": 4, "high": 6},
            "l2_leaf_reg": {"type": "float", "low": 1.0, "high": 10.0},
            "min_data_in_leaf": {"type": "int", "low": 1, "high": 50},
            "subsample": {"type": "float", "low": 0.5, "high": 1.0},
            "colsample_bylevel": {"type": "float", "low": 0.3, "high": 1.0},
            "random_strength": {"type": "float", "low": 0.0, "high": 10.0},
        }

    raise ValueError(f"No search space defined for model family '{model_family}'.")


def _sample_params(trial: optuna.Trial, search_space: dict[str, Any]) -> dict[str, Any]:
    """Sample hyperparameters from the search space using an Optuna trial."""
    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        ptype = spec["type"]
        if ptype == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif ptype == "float":
            params[name] = trial.suggest_float(
                name, spec["low"], spec["high"], log=spec.get("log", False)
            )
        elif ptype == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unknown param type '{ptype}' for '{name}'.")
    return params


# ── Early stopping helpers ──────────────────────────────────────────


def _early_stopping_fit_kwargs(
    family: str,
    X_val: np.ndarray,
    y_val: np.ndarray,
    early_stopping_rounds: int,
) -> dict[str, Any]:
    """Build ``fit()`` keyword arguments for early stopping by model family."""
    if family == "lightgbm":
        import lightgbm as lgb

        return {
            "eval_set": [(X_val, y_val)],
            "callbacks": [
                lgb.log_evaluation(period=-1),
                lgb.early_stopping(early_stopping_rounds, verbose=False),
            ],
        }
    if family == "xgboost":
        return {
            "eval_set": [(X_val, y_val)],
            "verbose": False,
        }
    if family == "catboost":
        return {
            "eval_set": [(X_val, y_val)],
            "early_stopping_rounds": early_stopping_rounds,
        }
    return {}


def _get_best_iteration(model: Any, family: str) -> int | None:
    """Retrieve the best boosting iteration after early-stopped training."""
    if family == "lightgbm":
        return getattr(model, "best_iteration_", None)
    if family == "xgboost":
        return getattr(model, "best_iteration", None)
    if family == "catboost":
        return getattr(model, "best_iteration_", None)
    return None


# ── Study creation and execution ────────────────────────────────────


def create_optuna_study(
    model_family: str,
    X: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    n_trials: int = 50,
    timeout: int | None = None,
    seed: int = 42,
    search_space: dict[str, Any] | None = None,
    early_stopping_rounds: int = EARLY_STOPPING_ROUNDS,
    max_iterations: int = MAX_ITERATIONS,
    show_progress_bar: bool = True,
) -> optuna.Study:
    """Create and run an Optuna study for a given model family.

    Each trial samples hyperparameters, runs K-fold cross-validation
    with native early stopping and threshold optimization, and reports
    mean QWK.  Unpromising trials are pruned via fold-level
    intermediate reporting to the configured Optuna pruner.

    Parameters
    ----------
    model_family : str
        Model family identifier.
    X : ndarray
        Feature matrix.
    y : ndarray
        Target vector.
    folds : list[tuple[ndarray, ndarray]]
        Pre-computed fold index pairs.
    n_trials : int
        Number of Optuna trials.
    timeout : int | None
        Maximum seconds for the study.
    seed : int
        Random seed for the sampler.
    search_space : dict | None
        Override search space. If None, uses :func:`get_search_space`.
    early_stopping_rounds : int
        Patience for native early stopping (boosting families only).
    max_iterations : int
        Maximum boosting iterations (cap for early stopping).
    show_progress_bar : bool
        Whether to display an Optuna/tqdm progress bar.

    Returns
    -------
    optuna.Study
        Completed study object.
    """
    if search_space is None:
        search_space = get_search_space(model_family)

    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    family = model_family.lower().strip()

    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial, search_space)

        # Inject iteration cap — early stopping selects the actual count
        if family in ("lightgbm", "xgboost"):
            params["n_estimators"] = max_iterations
        if family == "xgboost":
            params["early_stopping_rounds"] = early_stopping_rounds
        if family == "catboost":
            params["iterations"] = max_iterations
            params["use_best_model"] = True

        fold_qwks: list[float] = []
        best_iterations: list[int] = []

        for fold_idx, (train_idx, val_idx) in enumerate(folds):
            X_train, X_val = X_arr[train_idx], X_arr[val_idx]
            y_train, y_val = y_arr[train_idx], y_arr[val_idx]

            try:
                model_clone = _clone_model(model_family, params)
                fit_kwargs = _early_stopping_fit_kwargs(
                    family, X_val, y_val, early_stopping_rounds
                )
                model_clone.fit(X_train, y_train, **fit_kwargs)

                # Record best iteration
                best_iter = _get_best_iteration(model_clone, family)
                if best_iter is not None:
                    best_iterations.append(best_iter)

                proba = model_clone.predict_proba(X_val)

                # Handle incomplete probability columns
                n_classes = 5
                if proba.shape[1] < n_classes:
                    full_proba = np.zeros((proba.shape[0], n_classes))
                    for i, cls in enumerate(model_clone.classes_):
                        full_proba[:, int(cls)] = proba[:, i]
                    proba = full_proba

                thresh_result = optimize_thresholds(y_val, proba)
                fold_qwks.append(thresh_result["qwk_optimized"])

            except Exception as e:
                logger.warning(
                    "Trial %d fold %d failed for %s: %s",
                    trial.number,
                    fold_idx,
                    model_family,
                    e,
                )
                return float("-inf")

            # Report intermediate metric for Optuna pruning
            running_mean = float(np.mean(fold_qwks))
            trial.report(running_mean, fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_qwk = float(np.mean(fold_qwks))
        trial.set_user_attr("fold_qwks", fold_qwks)
        trial.set_user_attr("std_qwk", float(np.std(fold_qwks)))
        if best_iterations:
            trial.set_user_attr("best_iterations", best_iterations)
            trial.set_user_attr("mean_best_iteration", int(np.mean(best_iterations)))
        return mean_qwk

    # Suppress Optuna logging during study
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_startup_trials=5),
        study_name=f"{model_family}_tuning",
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=show_progress_bar,
    )

    logger.info(
        "[%s] Optuna study complete: %d trials, best QWK=%.4f",
        model_family,
        len(study.trials),
        study.best_value,
    )
    return study


def _clone_model(model_family: str, params: dict[str, Any]) -> Any:
    """Create a fresh model instance with the given parameters."""
    return create_model(model_family, params.copy())


# ── Candidate extraction ────────────────────────────────────────────


def extract_top_candidates(
    study: optuna.Study,
    n_top: int = 3,
) -> list[dict[str, Any]]:
    """Extract the top-N trials from an Optuna study.

    Parameters
    ----------
    study : optuna.Study
        Completed study.
    n_top : int
        Number of top trials to extract.

    Returns
    -------
    list[dict]
        Each dict contains ``trial_number``, ``params``, ``mean_qwk``,
        ``std_qwk``, ``fold_qwks``.
    """
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    sorted_trials = sorted(completed, key=lambda t: t.value, reverse=True)

    candidates = []
    for trial in sorted_trials[:n_top]:
        candidates.append(
            {
                "trial_number": trial.number,
                "params": dict(trial.params),
                "mean_qwk": trial.value,
                "std_qwk": trial.user_attrs.get("std_qwk", None),
                "fold_qwks": trial.user_attrs.get("fold_qwks", None),
                "best_iterations": trial.user_attrs.get("best_iterations", None),
                "mean_best_iteration": trial.user_attrs.get(
                    "mean_best_iteration", None
                ),
            }
        )
    return candidates


# ── Study persistence ───────────────────────────────────────────────


def save_optuna_study(study: optuna.Study, path: str | Path) -> Path:
    """Serialize an Optuna study summary to JSON.

    Persists trial parameters, values, and metadata for reproducibility.

    Parameters
    ----------
    study : optuna.Study
        Completed study.
    path : str | Path
        Output JSON file path.

    Returns
    -------
    Path
        Written file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    trials_data = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        trials_data.append(
            {
                "number": trial.number,
                "value": trial.value,
                "params": dict(trial.params),
                "user_attrs": dict(trial.user_attrs),
                "duration_seconds": (
                    (trial.datetime_complete - trial.datetime_start).total_seconds()
                    if trial.datetime_complete and trial.datetime_start
                    else None
                ),
            }
        )

    payload = {
        "study_name": study.study_name,
        "direction": "maximize",
        "n_trials": len(study.trials),
        "n_complete": len(trials_data),
        "n_pruned": sum(
            1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
        ),
        "best_trial": {
            "number": study.best_trial.number,
            "value": study.best_value,
            "params": dict(study.best_params),
        },
        "trials": trials_data,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    logger.info("Optuna study saved to %s (%d trials)", path, len(trials_data))
    return path
