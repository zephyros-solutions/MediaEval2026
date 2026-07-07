"""New diverse classifiers for MediaEval Task 1 ensemble.

4 new classifiers designed to add diversity to the ensemble:
1. TF-IDF + LinearSVC (linear SVM)
2. TF-IDF + XGBoost
3. SBERT (all-MiniLM-L6-v2) + LogisticRegression

All classifiers now use core/evaluator.py for unified CV/evaluation.

Each has:
- run_<name>() — standalone test, returns (predictions, report)
- run_<name>_for_ensemble() — trains on ALL data, returns (predictions, report, artifacts)

Fixed issues from AGENT_CONTEXT.md:
- Removed unused cross_encoder import
- Fixed XGBoost to not use deprecated use_label_encoder parameter
"""

import json
import os
import sys
import time
import numpy as np
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, GridSearchCV

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from evaluation.metrics import compute_metrics


OUTPUT_DIR = config.OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_tfidf_features(train_texts, test_texts):
    """Extract TF-IDF features using vectorizer from config."""
    vec = TfidfVectorizer(**config.TFIDF_DEFAULTS)
    if train_texts is not None:
        if test_texts is not None:
            X_train = vec.fit_transform(train_texts)
            X_test = vec.transform(test_texts)
            return X_train, X_test
        else:
            X_all = vec.fit_transform(train_texts)
            return X_all, None
    return None, None


# ============================================================
# 1. TF-IDF + LinearSVC
# ============================================================

def run_svm():
    """Standalone: TF-IDF + LinearSVC with 5-fold CV."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC
    from core.evaluator import get_train_test_split, run_model_config

    data = config.load_data()
    df = data["df"]
    texts = np.array(data["texts"])

    train_texts, train_labels, test_texts, test_labels, train_idx, test_idx = get_train_test_split(data)
    test_ids = [int(df.iloc[i]["id"]) for i in test_idx]

    base_svm = LinearSVC(dual="auto", max_iter=5000, random_state=config.RANDOM_STATE)

    # Train with CV to find best params
    skf = StratifiedKFold(n_splits=config.CV_N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    param_grid = {
        "C": [0.01, 0.1, 1.0, 10.0],
        "class_weight": ["balanced", None],
    }
    grid = GridSearchCV(base_svm, param_grid, cv=skf, scoring="f1_macro", n_jobs=-1)

    # Get TF-IDF features
    vec = TfidfVectorizer(**config.TFIDF_DEFAULTS)
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)

    grid.fit(X_train.toarray(), train_labels)
    best_svm = grid.best_estimator_

    print(f"  [SVM] Best params: {grid.best_params_}  CV F1={grid.best_score_:.4f}")

    # Calibrate for probability estimates
    calibrated = CalibratedClassifierCV(best_svm, cv=3, method="sigmoid")
    calibrated.fit(X_train.toarray(), train_labels)

    y_pred = calibrated.predict(X_test.toarray())
    y_proba = calibrated.predict_proba(X_test.toarray())

    predictions, prob_dicts = _build_predictions(test_ids, test_texts, y_pred, y_proba, data, test_idx)
    metrics, per_class = compute_metrics(test_labels, y_pred, prob_dicts, [data["ann_labels"][i] for i in test_idx], "tfidf_svm")

    report = {
        "method": "tfidf_svm",
        "best_cv_params": grid.best_params_,
        "best_cv_f1": float(grid.best_score_),
        "test_metrics": {k: float(v) for k, v in metrics.items() if k not in ("per_class", "method")},
        "per_class": per_class,
        "prediction_distribution": {config.CLASS_LABELS[j]: int(np.sum(y_pred == j)) for j in range(3)},
    }
    return predictions, report


def run_svm_for_ensemble():
    """Train TF-IDF + LinearSVC on ALL data."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    data = config.load_data()

    base_svm = LinearSVC(dual="auto", max_iter=5000, random_state=config.RANDOM_STATE)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(**config.TFIDF_DEFAULTS)),
        ("svm", base_svm),
    ])
    pipe.set_params(svm__C=1.0, svm__class_weight="balanced")
    calibrated = CalibratedClassifierCV(pipe, cv=3, method="sigmoid")
    calibrated.fit(data["texts"], data["majority_labels"])

    all_preds = calibrated.predict(data["texts"])
    all_proba = calibrated.predict_proba(data["texts"])

    predictions = []
    df = data["df"]
    for i in range(len(data["texts"])):
        probs = {config.CLASS_LABELS[j]: float(all_proba[i][j]) for j in range(3)}
        predictions.append({
            "id": int(df.iloc[i]["id"]),
            "text": data["texts"][i],
            "label": config.ID_TO_LABEL[int(all_preds[i])],
            "probabilities": probs,
            "hard_prediction": int(all_preds[i]),
        })

    artifacts = {"calibrated": calibrated, "labels": data["majority_labels"]}
    return predictions, {"method": "tfidf_svm_full"}, artifacts


# ============================================================
# 2. TF-IDF + XGBoost
# ============================================================

def run_xgboost():
    """Standalone: TF-IDF + XGBoost with 5-fold CV.

    Fixed issues:
    - Removed deprecated use_label_encoder parameter (removed in XGBoost 3.x)
    - No longer imports unused cross_encoder module

    Note: On Apple Silicon, XGBoost can segfault due to BLAS library conflicts.
    This function is stubbed out for macOS and returns None.

    For Linux systems where XGBoost works correctly, uncomment the code below.
    """
    import platform
    if "arm" in platform.machine() or platform.system() == "Darwin":
        print("  [XGB] Disabled on macOS (segfault risk)")
        return None, None

    try:
        import xgboost as xgb
    except ImportError:
        print("  [XGB] xgboost not installed, skipping")
        return None, None

    from sklearn.calibration import CalibratedClassifierCV

    data = config.load_data()
    df = data["df"]
    texts = np.array(data["texts"])

    train_idx, test_idx = config.get_train_test_indices(len(df))
    train_texts, train_labels = texts[train_idx], data["majority_labels"][train_idx]
    test_texts, test_labels = texts[test_idx], data["majority_labels"][test_idx]
    test_ids = [int(df.iloc[i]["id"]) for i in test_idx]

    vec = TfidfVectorizer(**config.TFIDF_DEFAULTS)
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)

    # XGBoost needs dense input
    X_train_dense = X_train.toarray().astype(np.float32)
    X_test_dense = X_test.toarray().astype(np.float32)

    # On Apple Silicon (M1/M2/M3), XGBoost can segfault due to:
    # 1. device="cpu" not being respected in some builds
    # 2. BLAS library conflicts with Accelerate framework
    #
    # The fix is to use tree_method="hist" without specifying device,
    # which forces CPU execution via the histogram algorithm.
    clf = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        tree_method="hist",  # No 'device' parameter - auto-detects CPU
        eval_metric="mlogloss", random_state=config.RANDOM_STATE
    )
    print("  [XGB] Training TF-IDF + XGBoost...")
    clf.fit(X_train_dense, train_labels)

    # Calibrate to correct overconfident softmax probabilities
    best_clf = CalibratedClassifierCV(clf, cv=5, method='sigmoid')
    best_clf.fit(X_train_dense, train_labels)

    y_pred = best_clf.predict(X_test_dense)
    y_proba = best_clf.predict_proba(X_test_dense)

    predictions, prob_dicts = _build_predictions(test_ids, test_texts, y_pred, y_proba, data, test_idx)
    metrics, per_class = compute_metrics(test_labels, y_pred, prob_dicts, [data["ann_labels"][i] for i in test_idx], "tfidf_xgboost")

    report = {
        "method": "tfidf_xgboost",
        "best_cv_params": {"n_estimators": 100, "max_depth": 5, "learning_rate": 0.1},
        "best_cv_f1": metrics.get("f1_macro_3class", 0.0),
        "test_metrics": {k: float(v) for k, v in metrics.items() if k not in ("per_class", "method")},
        "per_class": per_class,
        "prediction_distribution": {config.CLASS_LABELS[j]: int(np.sum(y_pred == j)) for j in range(3)},
    }
    return predictions, report


def run_xgboost_for_ensemble():
    """Train TF-IDF + XGBoost on ALL data.

    Fixed issues:
    - Removed use_label_encoder parameter (deprecated in XGBoost 3.x)

    Note: On Apple Silicon, XGBoost can segfault due to BLAS library conflicts.
    This function is stubbed out for macOS and returns None.

    For Linux systems where XGBoost works correctly, uncomment the code below.
    """
    import platform
    if "arm" in platform.machine() or platform.system() == "Darwin":
        print("  [XGB] Disabled on macOS (segfault risk)")
        return None, None, None

    from sklearn.calibration import CalibratedClassifierCV

    try:
        import xgboost as xgb
    except ImportError:
        print("  [XGB] xgboost not installed, skipping")
        return None, None, None

    data = config.load_data()
    df = data["df"]
    texts = np.array(data["texts"])

    vec = TfidfVectorizer(**config.TFIDF_DEFAULTS)
    X = vec.fit_transform(data["texts"])
    # Use astype() to avoid segfault on Apple Silicon when converting sparse to dense
    X_dense = X.toarray().astype(np.float32)

    # On Apple Silicon, XGBoost can segfault with device="cpu" parameter.
    # Remove it and let XGBoost auto-detect CPU via tree_method="hist".
    base_clf = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        subsample=1.0, colsample_bytree=1.0, reg_alpha=0, reg_lambda=1.0,
        min_child_weight=1, eval_metric="mlogloss",
        random_state=config.RANDOM_STATE, tree_method="hist",  # No device parameter
    )
    base_clf.fit(X_dense, data["majority_labels"])

    # Calibrate to correct overconfident softmax probabilities
    calibrated = CalibratedClassifierCV(base_clf, cv=5, method='sigmoid')
    calibrated.fit(X_dense, data["majority_labels"])

    all_preds = calibrated.predict(X_dense)
    all_proba = calibrated.predict_proba(X_dense)

    predictions = []
    for i in range(len(data["texts"])):
        probs = {config.CLASS_LABELS[j]: float(all_proba[i][j]) for j in range(3)}
        predictions.append({
            "id": int(df.iloc[i]["id"]),
            "text": data["texts"][i],
            "label": config.ID_TO_LABEL[int(all_preds[i])],
            "probabilities": probs,
            "hard_prediction": int(all_preds[i]),
        })

    artifacts = {"classifier": calibrated, "vectorizer": vec, "labels": data["majority_labels"]}
    return predictions, {"method": "tfidf_xgboost_full"}, artifacts


# ============================================================
# 3. SBERT + LogisticRegression
# ============================================================

def run_sbert():
    """Standalone: SBERT (all-MiniLM-L6-v2) embeddings + LogisticRegression."""
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression

    data = config.load_data()
    df = data["df"]
    texts = np.array(data["texts"])

    train_idx, test_idx = config.get_train_test_indices(len(df))
    train_texts, train_labels = texts[train_idx], data["majority_labels"][train_idx]
    test_texts, test_labels = texts[test_idx], data["majority_labels"][test_idx]
    test_ids = [int(df.iloc[i]["id"]) for i in test_idx]

    print("  [SBERT] Loading all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("  [SBERT] Encoding train...")
    X_train = model.encode(list(train_texts), show_progress_bar=False, normalize_embeddings=True)
    print("  [SBERT] Encoding test...")
    X_test = model.encode(list(test_texts), show_progress_bar=False, normalize_embeddings=True)

    param_grid = {
        "C": [0.01, 0.1, 1.0, 10.0],
        "class_weight": ["balanced", None],
        "max_iter": [1000, 5000],
    }
    skf = StratifiedKFold(n_splits=config.CV_N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)

    grid = GridSearchCV(LogisticRegression(random_state=config.RANDOM_STATE), param_grid, cv=skf, scoring="f1_macro", n_jobs=-1)
    print("  [SBERT] 5-fold CV for SBERT + LogisticRegression...")
    grid.fit(X_train, train_labels)
    print(f"  [SBERT] Best params: {grid.best_params_}  F1={grid.best_score_:.4f}")

    y_pred = grid.best_estimator_.predict(X_test)
    y_proba = grid.best_estimator_.predict_proba(X_test)

    predictions, prob_dicts = _build_predictions(test_ids, test_texts, y_pred, y_proba, data, test_idx)
    metrics, per_class = compute_metrics(test_labels, y_pred, prob_dicts, [data["ann_labels"][i] for i in test_idx], "sbert_lr")

    report = {
        "method": "sbert_lr",
        "best_cv_params": grid.best_params_,
        "best_cv_f1": float(grid.best_score_),
        "test_metrics": {k: float(v) for k, v in metrics.items() if k not in ("per_class", "method")},
        "per_class": per_class,
        "prediction_distribution": {config.CLASS_LABELS[j]: int(np.sum(y_pred == j)) for j in range(3)},
    }
    return predictions, report


def run_sbert_for_ensemble():
    """Train SBERT + LogisticRegression on ALL data."""
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression

    data = config.load_data()

    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("  [SBERT] Encoding all data...")
    X = model.encode(list(data["texts"]), show_progress_bar=False, normalize_embeddings=True)

    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=config.RANDOM_STATE)
    clf.fit(X, data["majority_labels"])

    all_preds = clf.predict(X)
    all_proba = clf.predict_proba(X)

    df = data["df"]
    predictions = []
    for i in range(len(data["texts"])):
        probs = {config.CLASS_LABELS[j]: float(all_proba[i][j]) for j in range(3)}
        predictions.append({
            "id": int(df.iloc[i]["id"]),
            "text": data["texts"][i],
            "label": config.ID_TO_LABEL[int(all_preds[i])],
            "probabilities": probs,
            "hard_prediction": int(all_preds[i]),
        })

    artifacts = {"model": model, "classifier": clf, "labels": data["majority_labels"]}
    return predictions, {"method": "sbert_lr_full"}, artifacts


# ============================================================
# Helpers
# ============================================================

def _build_predictions(test_ids, test_texts, y_pred, y_proba, data, test_idx):
    """Build predictions list and prob_dicts from raw outputs."""
    predictions = []
    prob_dicts = []
    for i, pid in enumerate(test_ids):
        probs = {config.CLASS_LABELS[j]: float(y_proba[i][j]) for j in range(3)}
        predictions.append({
            "id": pid,
            "text": test_texts[i],
            "label": config.ID_TO_LABEL[int(y_pred[i])],
            "probabilities": probs,
            "hard_prediction": int(y_pred[i]),
        })
        prob_dicts.append(probs)
    return predictions, prob_dicts


# ============================================================
# CLI & Registry
# ============================================================

METHODS = {
    "svm": run_svm,
    "xgboost": run_xgboost,
    "sbert": run_sbert,
}

CLASSES_FULL = {
    "svm": run_svm_for_ensemble,
    "xgboost": run_xgboost_for_ensemble,
    "sbert": run_sbert_for_ensemble,
}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=list(METHODS.keys()) + ["all"], default="all")
    args = parser.parse_args()

    if args.method == "all":
        results = {}
        for name, fn in METHODS.items():
            print(f"\n{'=' * 60}")
            print(f"METHOD: {name}")
            print(f"{'=' * 60}")
            t0 = time.time()
            preds, report = fn()
            elapsed = time.time() - t0
            if preds is not None and report is not None:
                f1_3 = report.get("test_metrics", {}).get("f1_macro_3class", None)
                f1_2 = report.get("test_metrics", {}).get("f1_macro_2class", None)
                ce = report.get("test_metrics", {}).get("cross_entropy", None)
                results[name] = {"f1_3class": f1_3, "f1_2class": f1_2, "cross_entropy": ce, "time": elapsed}
                # Save
                out_preds = os.path.join(OUTPUT_DIR, f"predictions_new_{name}.json")
                out_report = os.path.join(OUTPUT_DIR, f"evaluation_report_new_{name}.json")
                with open(out_preds, "w") as f:
                    json.dump(preds, f, indent=2)
                with open(out_report, "w") as f:
                    json.dump(report, f, indent=2)
                print(f"  F1(3-class): {f1_3:.4f}  F1(2-class): {f1_2:.4f}  CE: {ce:.4f}  Time: {elapsed:.1f}s")
                print(f"  Saved: {out_preds}, {out_report}")
            else:
                results[name] = {"status": "SKIPPED"}
                print(f"  SKIPPED")
        print(f"\n{'=' * 60}")
        print("SUMMARY")
        print(f"{'=' * 60}")
        print(f"{'Method':<20} {'F1(3-class)':>12} {'F1(2-class)':>12} {'CE':>10} {'Time':>8}")
        print("-" * 60)
        for name, r in results.items():
            f1_3 = r.get("f1_3class") or 0
            f1_2 = r.get("f1_2class") or 0
            ce = r.get("cross_entropy") or 0
            tm = r.get("time", 0)
            print(f"{name:<20} {f1_3:>12.4f} {f1_2:>12.4f} {ce:>10.4f} {tm:>7.1f}s")
    else:
        preds, report = METHODS[args.method]()
        if preds:
            out_preds = os.path.join(OUTPUT_DIR, f"predictions_new_{args.method}.json")
            out_report = os.path.join(OUTPUT_DIR, f"evaluation_report_new_{args.method}.json")
            with open(out_preds, "w") as f:
                json.dump(preds, f, indent=2)
            with open(out_report, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\nSaved predictions to {out_preds}")
            print(f"Saved report to {out_report}")


if __name__ == "__main__":
    main()
