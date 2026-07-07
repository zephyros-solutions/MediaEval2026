"""
Shared evaluation engine for MediaEval Task 1 classification.

This module provides a unified CV/evaluation infrastructure that can be used by
any classifier. The design follows AGENT_CONTEXT.md target architecture:

- One data loading path (config.load_data())
- One CV/evaluation path (run_model_config() function)
- Dict-based model iteration (models_dict defines all models)

Usage:
    from core.evaluator import run_standalone_models, run_ensemble_models

    MODELS = {
        "svm": {
            "feature_extractor": extract_tfidf_features,
            "classifier_fn": lambda: LinearSVC(),
            "param_grid": {"C": [0.1, 1.0, 10.0]},
        },
        "lr": {
            "feature_extractor": extract_tfidf_features,
            "classifier_fn": lambda: LogisticRegression(),
        },
    }

    # Standalone (80/20 split with CV for model selection)
    results = run_standalone_models(MODELS)

    # Ensemble (train on ALL data, return artifacts for voting)
    artifacts = run_ensemble_models(MODELS)
"""

import os
import sys
import time
from typing import Dict, Tuple, Any, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from evaluation.metrics import compute_metrics


def get_train_test_split(data: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get train/test split indices and data."""
    df = data["df"]
    texts = np.array(data["texts"])
    train_idx, test_idx = config.get_train_test_indices(len(df))
    train_texts, train_labels = texts[train_idx], data["majority_labels"][train_idx]
    test_texts, test_labels = texts[test_idx], data["majority_labels"][test_idx]
    return train_texts, train_labels, test_texts, test_labels, train_idx, test_idx


def run_cv_for_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    classifier_fn: callable,
    param_grid: Optional[Dict],
    cv_folds: int = config.CV_N_FOLDS
) -> Tuple[Any, float, Dict]:
    """
    Run CV to find best hyperparameters for a classifier.

    Returns:
        (best_classifier, best_cv_f1, cv_scores_per_fold)
    """
    from sklearn.model_selection import StratifiedKFold, GridSearchCV
    from sklearn.metrics import f1_score

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.RANDOM_STATE)

    if param_grid:
        grid = GridSearchCV(
            classifier_fn(), param_grid,
            cv=skf, scoring="f1_macro", n_jobs=-1
        )
        grid.fit(X_train, y_train)
        best_clf = grid.best_estimator_
        best_f1 = float(grid.best_score_)
        cv_scores = [float(s) for s in grid.cv_results_["mean_test_score"]]
    else:
        # No hyperparameter tuning - just use the classifier as-is
        clf = classifier_fn()
        cv_scores = []
        for train_fold_idx, val_fold_idx in skf.split(X_train, y_train):
            fold_clf = classifier_fn()
            fold_clf.fit(X_train[train_fold_idx], y_train[train_fold_idx])
            y_pred = fold_clf.predict(X_train[val_fold_idx])
            fold_f1 = f1_score(y_train[val_fold_idx], y_pred, average="macro", zero_division=0)
            cv_scores.append(fold_f1)
        best_clf = classifier_fn()
        best_clf.fit(X_train, y_train)
        best_f1 = float(np.mean(cv_scores))

    return best_clf, best_f1, cv_scores


def evaluate_model(
    clf: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_texts: np.ndarray,
    test_ids: list,
    data: dict,
    test_idx: np.ndarray
) -> Tuple[list, dict]:
    """
    Evaluate a trained classifier on test set and return predictions + report.

    Returns:
        (predictions, report)
    """
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    # Build predictions in challenge format
    predictions = []
    prob_dicts = []
    for i, pid in enumerate(test_ids):
        probs = {config.CLASS_LABELS[j]: float(y_proba[i][j]) for j in range(3)}
        predictions.append({
            "id": int(pid),
            "text": test_texts[i],
            "label": config.ID_TO_LABEL[int(y_pred[i])],
            "probabilities": probs,
            "hard_prediction": int(y_pred[i]),
        })
        prob_dicts.append(probs)

    # Compute metrics
    metrics, per_class = compute_metrics(
        y_test, y_pred, prob_dicts,
        [data["ann_labels"][i] for i in test_idx],
        "evaluator_model"
    )

    report = {
        "test_metrics": {k: float(v) for k, v in metrics.items() if k not in ("per_class", "method")},
        "per_class": per_class,
        "prediction_distribution": {config.CLASS_LABELS[j]: int(np.sum(y_pred == j)) for j in range(3)},
    }
    return predictions, report


def run_model_config(
    model_name: str,
    feature_extractor: callable,
    classifier_fn: callable,
    param_grid: Optional[Dict] = None,
    use_full_data: bool = False
) -> Tuple[list, dict, Any]:
    """
    Run a single model configuration through the evaluation pipeline.

    Args:
        model_name: Name of the model for reporting
        feature_extractor: Function that takes (train_texts, test_texts or all_texts)
                          and returns (X_train, X_test or X_all)
        classifier_fn: Function that returns an untrained classifier instance
        param_grid: Hyperparameter grid for CV tuning (None = no tuning)
        use_full_data: If True, train on ALL data (for ensemble).
                      If False, use 80/20 split with CV.

    Returns:
        (predictions, report, artifacts)
        artifacts = {"classifier": clf, "feature_extractor": feature_extractor,
                     "data": data_dict}
    """
    start_time = time.time()
    print(f"\n[MODEL {model_name}] Running...")

    # Load data
    data = config.load_data()
    df = data["df"]

    if use_full_data:
        # Train on ALL data (for ensemble submission)
        print("[MODEL] Training on ALL 1333 instances...")

        # Get features for all data
        texts = np.array(data["texts"])
        X_all = feature_extractor(texts, None)

        # Train classifier
        best_clf, best_f1, _ = run_cv_for_model(X_all, data["majority_labels"], classifier_fn, param_grid)
        print(f"[MODEL] CV F1(3-class)={best_f1:.4f}")

        # Make predictions on all data
        y_pred = best_clf.predict(X_all)
        y_proba = best_clf.predict_proba(X_all)

        # Build predictions
        predictions = []
        for i in range(len(data["texts"])):
            probs = {config.CLASS_LABELS[j]: float(y_proba[i][j]) for j in range(3)}
            predictions.append({
                "id": int(df.iloc[i]["id"]),
                "text": data["texts"][i],
                "label": config.ID_TO_LABEL[int(y_pred[i])],
                "probabilities": probs,
                "hard_prediction": int(y_pred[i]),
            })

        elapsed = time.time() - start_time
        report = {
            "method": model_name,
            "mode": "ensemble",
            "cv_scores": {"macro_f1_3class": best_f1},
            "elapsed": elapsed,
        }
        artifacts = {
            "classifier": best_clf,
            "feature_extractor": feature_extractor,
            "data": data,
        }
        return predictions, report, artifacts

    else:
        # Standalone: 80/20 split with CV for model selection
        print("[MODEL] Running 5-fold CV on 80% training set...")

        train_texts, train_labels, test_texts, test_labels, train_idx, test_idx = get_train_test_split(data)
        test_ids = [int(df.iloc[i]["id"]) for i in test_idx]

        # Extract features
        X_train = feature_extractor(train_texts, None)
        X_test = feature_extractor(test_texts, None)

        # CV to find best hyperparameters
        best_clf, best_f1, cv_scores = run_cv_for_model(X_train, train_labels, classifier_fn, param_grid)
        print(f"[MODEL] Best CV F1(3-class)={best_f1:.4f}")

        # Evaluate on test set
        predictions, report = evaluate_model(
            best_clf, X_test, test_labels, test_texts, test_ids, data, test_idx
        )
        report["cv_scores"] = {"macro_f1_3class": best_f1, "per_fold": cv_scores}
        report["elapsed"] = time.time() - start_time

        artifacts = {
            "classifier": best_clf,
            "feature_extractor": feature_extractor,
            "data": data,
            "train_idx": train_idx,
            "test_idx": test_idx,
        }
        return predictions, report, artifacts


def run_standalone_models(models_dict: Dict) -> Dict[str, Tuple[list, dict]]:
    """
    Run all models in standalone mode (80/20 split with CV).

    Each model goes through the same CV/fold/evaluation pipeline for fair comparison.

    Args:
        models_dict: {name: {"feature_extractor": fn, "classifier_fn": fn, "param_grid": dict}}

    Returns:
        {name: (predictions, report)} for each model
    """
    results = {}
    print("\n" + "=" * 80)
    print("STANDALONE MODE: Running all models with 5-fold CV on 80% training set")
    print("=" * 80)

    for name, config in models_dict.items():
        try:
            predictions, report, _ = run_model_config(
                model_name=name,
                feature_extractor=config["feature_extractor"],
                classifier_fn=config["classifier_fn"],
                param_grid=config.get("param_grid"),
                use_full_data=False
            )
            results[name] = (predictions, report)
            print(f"[MODEL {name}] F1(3-class)={report['test_metrics'].get('f1_macro_3class', 'N/A'):.4f}")
        except Exception as e:
            print(f"[MODEL {name}] FAILED: {e}")
            results[name] = (None, {"error": str(e)})

    return results


def run_ensemble_models(models_dict: Dict) -> Dict[str, Tuple[list, dict, Any]]:
    """
    Run all models in ensemble mode (train on ALL 1333 instances).

    Args:
        models_dict: {name: {"feature_extractor": fn, "classifier_fn": fn, "param_grid": dict}}

    Returns:
        {name: (predictions, report, artifacts)} for each model
        artifacts can be used by submit.py to generate predictions on test set
    """
    results = {}
    print("\n" + "=" * 80)
    print("ENSEMBLE MODE: Training all models on ALL 1333 instances")
    print("=" * 80)

    for name, config in models_dict.items():
        try:
            predictions, report, artifacts = run_model_config(
                model_name=name,
                feature_extractor=config["feature_extractor"],
                classifier_fn=config["classifier_fn"],
                param_grid=config.get("param_grid"),
                use_full_data=True
            )
            results[name] = (predictions, report, artifacts)
            print(f"[MODEL {name}] Trained successfully")
        except Exception as e:
            print(f"[MODEL {name}] FAILED: {e}")
            results[name] = (None, {"error": str(e)}, None)

    return results
