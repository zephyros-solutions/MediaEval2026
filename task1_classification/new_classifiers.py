"""New diverse classifiers for MediaEval Task 1 ensemble.

4 new classifiers designed to add diversity to the ensemble:
1. TF-IDF + LinearSVC (linear SVM)
2. TF-IDF + XGBoost
3. SBERT (all-MiniLM-L6-v2) + LogisticRegression
4. Cross-encoder (reranker-MiniLM) + LogisticRegression

Each has:
- run_<name>() — standalone test, returns (predictions, report)
- run_<name>_for_ensemble() — trains on ALL data, returns (predictions, report, artifacts)
"""

import json
import os
import sys
import time
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import numpy as np

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from evaluation.metrics import compute_metrics

RANDOM_STATE = 42
OUTPUT_DIR = config.OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. TF-IDF + LinearSVC
# ============================================================

def run_svm():
    """Standalone: TF-IDF + LinearSVC with 5-fold CV.

    Uses fast LinearSVC for CV (no probability calibration during search),
    then wraps with CalibratedClassifierCV for probability estimates.
    """
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    data = config.load_data()
    df = data["df"]
    texts = np.array(data["texts"])

    split_idx = int(len(df) * config.TRAIN_VAL_SPLIT)
    train_texts, train_labels = texts[:split_idx], data["majority_labels"][:split_idx]
    test_texts, test_labels = texts[split_idx:], data["majority_labels"][split_idx:]
    test_ids = [int(df.iloc[i]["id"]) for i in range(split_idx, len(df))]

    base_svm = LinearSVC(dual="auto", max_iter=5000, random_state=RANDOM_STATE)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("svm", base_svm),
    ])
    param_grid = {
        "tfidf__max_df": [0.9, 0.95],
        "tfidf__ngram_range": [(1, 2), (1, 3)],
        "tfidf__min_df": [2, 3],
        "svm__C": [0.01, 0.1, 1.0, 10.0],
        "svm__class_weight": ["balanced", None],
    }
    skf = StratifiedKFold(n_splits=config.CV_N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    grid = GridSearchCV(pipe, param_grid, cv=skf, scoring="f1_macro", n_jobs=-1)
    print("  [SVM] 5-fold CV for TF-IDF + LinearSVC...")
    grid.fit(train_texts, train_labels)
    print(f"  [SVM] Best params: {grid.best_params_}  F1={grid.best_score_:.4f}")

    # Wrap best estimator with calibration for probability estimates
    best_pipe = grid.best_estimator_
    calibrated = CalibratedClassifierCV(best_pipe, cv=3, method="sigmoid")
    calibrated.fit(train_texts, train_labels)

    test_preds = calibrated.predict(test_texts)
    test_proba = calibrated.predict_proba(test_texts)

    predictions = []
    prob_dicts = []
    for i, pid in enumerate(test_ids):
        probs = {config.CLASS_LABELS[j]: float(test_proba[i][j]) for j in range(3)}
        predictions.append({
            "id": pid, "text": test_texts[i], "label": config.ID_TO_LABEL[int(test_preds[i])],
            "probabilities": probs, "hard_prediction": int(test_preds[i]),
        })
        prob_dicts.append(probs)

    y_true = test_labels
    y_pred = test_preds
    metrics, per_class = compute_metrics(y_true, y_pred, prob_dicts, [data["ann_labels"][i] for i in range(split_idx, len(df))], "tfidf_svm")

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
    """Train TF-IDF + LinearSVC on ALL data, return predictions + artifacts."""
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC

    data = config.load_data()
    df = data["df"]
    texts = np.array(data["texts"])

    base_svm = LinearSVC(dual="auto", max_iter=5000, random_state=RANDOM_STATE)
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
    for i in range(len(data["texts"])):
        probs = {config.CLASS_LABELS[j]: float(all_proba[i][j]) for j in range(3)}
        predictions.append({
            "id": int(df.iloc[i]["id"]), "text": data["texts"][i],
            "label": config.ID_TO_LABEL[int(all_preds[i])],
            "probabilities": probs, "hard_prediction": int(all_preds[i]),
        })

    artifacts = {"calibrated": calibrated, "labels": data["majority_labels"]}
    return predictions, {"method": "tfidf_svm_full"}, artifacts


# ============================================================
# 2. TF-IDF + XGBoost
# ============================================================

def run_xgboost():
    """Standalone: TF-IDF + XGBoost with 5-fold CV."""
    try:
        import xgboost as xgb  # noqa: F401
    except ImportError:
        print("  [XGB] xgboost not installed, skipping")
        return None, None

    data = config.load_data()
    df = data["df"]
    texts = np.array(data["texts"])

    split_idx = int(len(df) * config.TRAIN_VAL_SPLIT)
    train_texts, train_labels = texts[:split_idx], data["majority_labels"][:split_idx]
    test_texts, test_labels = texts[split_idx:], data["majority_labels"][split_idx:]
    test_ids = [int(df.iloc[i]["id"]) for i in range(split_idx, len(df))]

    vec = TfidfVectorizer(**config.TFIDF_DEFAULTS)
    X_train = vec.fit_transform(train_texts)
    X_test = vec.transform(test_texts)

    param_grid = {
        "n_estimators": [50, 100],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
        "reg_alpha": [0, 0.1],
        "reg_lambda": [0.5, 1.0],
    }
    skf = StratifiedKFold(n_splits=config.CV_N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    # XGBoost needs dense input for GridSearchCV
    X_train_dense = np.array(X_train.toarray())
    X_test_dense = np.array(X_test.toarray())

    clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=RANDOM_STATE)
    grid = GridSearchCV(clf, param_grid, cv=skf, scoring="f1_macro", n_jobs=1)
    print("  [XGB] 5-fold CV for TF-IDF + XGBoost...")
    grid.fit(X_train_dense, train_labels)
    print(f"  [XGB] Best params: {grid.best_params_}  F1={grid.best_score_:.4f}")

    y_pred = grid.best_estimator_.predict(X_test_dense)
    y_proba = grid.best_estimator_.predict_proba(X_test_dense)

    predictions, prob_dicts = __build_predictions(test_ids, test_texts, y_pred, y_proba, data, split_idx)
    metrics, per_class = compute_metrics(test_labels, y_pred, prob_dicts, [data["ann_labels"][i] for i in range(split_idx, len(df))], "tfidf_xgboost")

    report = {
        "method": "tfidf_xgboost",
        "best_cv_params": grid.best_params_,
        "best_cv_f1": float(grid.best_score_),
        "test_metrics": {k: float(v) for k, v in metrics.items() if k not in ("per_class", "method")},
        "per_class": per_class,
        "prediction_distribution": {config.CLASS_LABELS[j]: int(np.sum(y_pred == j)) for j in range(3)},
    }
    return predictions, report


def run_xgboost_for_ensemble():
    """Train TF-IDF + XGBoost on ALL data."""
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
    X_dense = X.toarray()

    clf = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        subsample=1.0, colsample_bytree=1.0, reg_alpha=0, reg_lambda=1.0,
        min_child_weight=1, use_label_encoder=False, eval_metric="mlogloss",
        random_state=RANDOM_STATE,
    )
    clf.fit(X_dense, data["majority_labels"])

    all_preds = clf.predict(X_dense)
    all_proba = clf.predict_proba(X_dense)

    predictions = []
    for i in range(len(data["texts"])):
        probs = {config.CLASS_LABELS[j]: float(all_proba[i][j]) for j in range(3)}
        predictions.append({
            "id": int(df.iloc[i]["id"]), "text": data["texts"][i],
            "label": config.ID_TO_LABEL[int(all_preds[i])],
            "probabilities": probs, "hard_prediction": int(all_preds[i]),
        })

    artifacts = {"classifier": clf, "vectorizer": vec, "labels": data["majority_labels"]}
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

    split_idx = int(len(df) * config.TRAIN_VAL_SPLIT)
    train_texts, train_labels = texts[:split_idx], data["majority_labels"][:split_idx]
    test_texts, test_labels = texts[split_idx:], data["majority_labels"][split_idx:]
    test_ids = [int(df.iloc[i]["id"]) for i in range(split_idx, len(df))]

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
    skf = StratifiedKFold(n_splits=config.CV_N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    grid = GridSearchCV(LogisticRegression(random_state=RANDOM_STATE), param_grid, cv=skf, scoring="f1_macro", n_jobs=-1)
    print("  [SBERT] 5-fold CV for SBERT + LogisticRegression...")
    grid.fit(X_train, train_labels)
    print(f"  [SBERT] Best params: {grid.best_params_}  F1={grid.best_score_:.4f}")

    y_pred = grid.best_estimator_.predict(X_test)
    y_proba = grid.best_estimator_.predict_proba(X_test)

    predictions, prob_dicts = __build_predictions(test_ids, test_texts, y_pred, y_proba, data, split_idx)
    metrics, per_class = compute_metrics(test_labels, y_pred, prob_dicts, [data["ann_labels"][i] for i in range(split_idx, len(df))], "sbert_lr")

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

    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE)
    clf.fit(X, data["majority_labels"])

    all_preds = clf.predict(X)
    all_proba = clf.predict_proba(X)

    df = data["df"]
    predictions = []
    for i in range(len(data["texts"])):
        probs = {config.CLASS_LABELS[j]: float(all_proba[i][j]) for j in range(3)}
        predictions.append({
            "id": int(df.iloc[i]["id"]), "text": data["texts"][i],
            "label": config.ID_TO_LABEL[int(all_preds[i])],
            "probabilities": probs, "hard_prediction": int(all_preds[i]),
        })

    artifacts = {"model": model, "classifier": clf, "labels": data["majority_labels"]}
    return predictions, {"method": "sbert_lr_full"}, artifacts


# ============================================================
# 4. Cross-encoder (reranker) + LogisticRegression
# ============================================================

def run_cross_encoder():
    """Standalone: Cross-encoder reranker scores + LogisticRegression."""
    from cross_encoder import CrossEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    data = config.load_data()
    df = data["df"]
    texts = np.array(data["texts"])

    split_idx = int(len(df) * config.TRAIN_VAL_SPLIT)
    train_texts, train_labels = texts[:split_idx], data["majority_labels"][:split_idx]
    test_texts, test_labels = texts[split_idx:], data["majority_labels"][split_idx:]
    test_ids = [int(df.iloc[i]["id"]) for i in range(split_idx, len(df))]

    class_descriptions = {
        0: "This tweet contains an unstated premise — a hidden assumption that supports the argument.",
        1: "This tweet contains an unstated conclusion — a logical outcome not explicitly stated.",
        2: "This tweet has a fully explicit argument with no missing premises or conclusions.",
    }

    print("  [CE] Loading cross-encoder reranker-MiniLM-v5...")
    ce = CrossEncoder("cross-encoder/reranker-MiniLM-L-6-v2")

    def encode_with_ce(texts_list, descriptions_dict):
        """Score each text against each class description, return feature matrix."""
        features = []
        for t in texts_list:
            scores = []
            for c in range(3):
                raw_scores = ce.predict([(t, descriptions_dict[c])])
                # Use raw score + unigram features as additional signal
                unigram = [float(w in set(t.lower().split())) for w in ["because", "so", "therefore", "thus", "hence", "implies", "means"]]
                scores.extend([raw_scores[0]] + unigram)
            features.append(scores)
        return np.array(features)

    # Actually score each (text, class) pair properly
    def encode_with_ce_v2(texts_list, descriptions_dict):
        pairs = []
        for t in texts_list:
            for c in range(3):
                pairs.append((t, descriptions_dict[c]))
        scores = ce.predict(pairs, show_progress_bar=False)
        # Reshape: len(texts) x 3
        scores = scores.reshape(len(texts_list), 3)
        # Add unigram features
        unigram_features = np.array([
            [float(w in set(t.lower().split())) for w in ["because", "so", "therefore", "thus", "hence", "implies", "means"]]
            for t in texts_list
        ])
        return np.hstack([scores, unigram_features])

    print("  [CE] Encoding train with reranker...")
    X_train = encode_with_ce_v2(list(train_texts), class_descriptions)
    print("  [CE] Encoding test with reranker...")
    X_test = encode_with_ce_v2(list(test_texts), class_descriptions)

    param_grid = {
        "C": [0.01, 0.1, 1.0, 10.0],
        "class_weight": ["balanced", None],
        "max_iter": [1000, 5000],
    }
    skf = StratifiedKFold(n_splits=config.CV_N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    print("  [CE] 5-fold CV for Cross-encoder + LogisticRegression...")
    grid = GridSearchCV(LogisticRegression(random_state=RANDOM_STATE), param_grid, cv=skf, scoring="f1_macro", n_jobs=-1)
    grid.fit(X_train, train_labels)
    print(f"  [CE] Best params: {grid.best_params_}  F1={grid.best_score_:.4f}")

    y_pred = grid.best_estimator_.predict(X_test)
    y_proba = grid.best_estimator_.predict_proba(X_test)

    predictions, prob_dicts = __build_predictions(test_ids, test_texts, y_pred, y_proba, data, split_idx)
    metrics, per_class = compute_metrics(test_labels, y_pred, prob_dicts, [data["ann_labels"][i] for i in range(split_idx, len(df))], "cross_encoder_lr")

    report = {
        "method": "cross_encoder_lr",
        "best_cv_params": grid.best_params_,
        "best_cv_f1": float(grid.best_score_),
        "test_metrics": {k: float(v) for k, v in metrics.items() if k not in ("per_class", "method")},
        "per_class": per_class,
        "prediction_distribution": {config.CLASS_LABELS[j]: int(np.sum(y_pred == j)) for j in range(3)},
    }
    return predictions, report


def run_cross_encoder_for_ensemble():
    """Train Cross-encoder + LogisticRegression on ALL data."""
    from cross_encoder import CrossEncoder
    from sklearn.linear_model import LogisticRegression

    data = config.load_data()
    df = data["df"]

    class_descriptions = {
        0: "This tweet contains an unstated premise — a hidden assumption that supports the argument.",
        1: "This tweet contains an unstated conclusion — a logical outcome not explicitly stated.",
        2: "This tweet has a fully explicit argument with no missing premises or conclusions.",
    }

    ce = CrossEncoder("cross-encoder/reranker-MiniLM-L-6-v2")

    def encode_all(texts_list):
        pairs = []
        for t in texts_list:
            for c in range(3):
                pairs.append((t, class_descriptions[c]))
        scores = ce.predict(pairs, show_progress_bar=False)
        scores = scores.reshape(len(texts_list), 3)
        unigram_features = np.array([
            [float(w in set(t.lower().split())) for w in ["because", "so", "therefore", "thus", "hence", "implies", "means"]]
            for t in texts_list
        ])
        return np.hstack([scores, unigram_features])

    print("  [CE] Encoding all data with reranker...")
    X = encode_all(data["texts"])

    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=RANDOM_STATE)
    clf.fit(X, data["majority_labels"])

    all_preds = clf.predict(X)
    all_proba = clf.predict_proba(X)

    predictions = []
    for i in range(len(data["texts"])):
        probs = {config.CLASS_LABELS[j]: float(all_proba[i][j]) for j in range(3)}
        predictions.append({
            "id": int(df.iloc[i]["id"]), "text": data["texts"][i],
            "label": config.ID_TO_LABEL[int(all_preds[i])],
            "probabilities": probs, "hard_prediction": int(all_preds[i]),
        })

    artifacts = {"encoder": ce, "classifier": clf, "labels": data["majority_labels"]}
    return predictions, {"method": "cross_encoder_lr_full"}, artifacts


# ============================================================
# Helpers
# ============================================================

def __build_predictions(test_ids, test_texts, y_pred, y_proba, data, split_idx):
    """Build predictions list and prob_dicts from raw outputs."""
    predictions = []
    prob_dicts = []
    for i, pid in enumerate(test_ids):
        probs = {config.CLASS_LABELS[j]: float(y_proba[i][j]) for j in range(3)}
        predictions.append({
            "id": pid, "text": test_texts[i], "label": config.ID_TO_LABEL[int(y_pred[i])],
            "probabilities": probs, "hard_prediction": int(y_pred[i]),
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
    # "cross_encoder": run_cross_encoder,  # Model unavailable (gated/network)
}

CLASSES_FULL = {
    "svm": run_svm_for_ensemble,
    "xgboost": run_xgboost_for_ensemble,
    "sbert": run_sbert_for_ensemble,
    "cross_encoder": run_cross_encoder_for_ensemble,
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
