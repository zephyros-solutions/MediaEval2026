"""
Ensemble & Alternative Approaches for Task 1: Enthymeme Detection

Each ensemble method combines results from the 4 standalone base classifiers:
  1. TF-IDF + Random Forest (task1_classifier_tfidf.run_tfidf)
  2. DistilBERT feature extraction + classifier (transformer.run_transformer)
  3. Ollama zero-shot (task1_ollama_classifier.run_ollama_zero)
  4. Ollama few-shot (task1_ollama_fewshot.run_ollama_fewshot)

Six ensemble strategies:
  1. Soft-voting ensemble: average probability vectors from all base classifiers
  2. Weighted soft-voting: weight by CV F1 scores
  3. Majority voting: hard vote from all base classifiers
  4. Feature-level fusion: TF-IDF features + DistilBERT embeddings stacked
  5. SBERT embeddings + linear classifier (separate sentence encoder)
  6. Bagging: multiple random forest models on bootstrapped data

Evaluation methodology (fair comparison with standalone methods):
  - Each ensemble method trains base classifiers on the 80% train split
  - Base classifiers predict on the 20% held-out test split
  - Ensemble combines the held-out predictions
  - Metrics computed on the held-out 20% (same as standalone methods)
  - This allows direct fair comparison with standalone method F1 scores

Each approach can be run standalone with:
    python task1_classification/task1_ensemble.py --method soft_voting

All predictions and reports go to outputs/.

Usage:
    from task1_classification.task1_ensemble import run_all_and_pick_best
    best_method, best_metrics, all_results = run_all_and_pick_best()
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Import base classifier run functions (each uses 80/20 split internally)
from task1_classification.task1_classifier_tfidf import run_tfidf
from task1_classification.transformer.transformer import run_transformer
from task1_classification.new_classifiers import (
    run_svm_for_ensemble, run_xgboost_for_ensemble,
    run_sbert_for_ensemble, run_cross_encoder_for_ensemble,
)

OUTPUT_DIR = config.OUTPUT_DIR


# =================== EVALUATION ===================

def _evaluate(preds, report_prefix=""):
    """Evaluate predictions against ground truth. Returns (predictions_list, report_dict)."""
    from evaluation.metrics import compute_metrics

    data = config.load_data()
    id_to_true = {data["ids"][i]: int(data["majority_labels"][i]) for i in range(len(data["ids"]))}
    id_to_soft = {data["ids"][i]: data["ann_labels"][i] for i in range(len(data["ids"]))}

    y_true, y_pred, prob_dicts, soft_labels = [], [], [], []
    for pid, p in preds.items():
        if pid not in id_to_true:
            continue
        y_true.append(id_to_true[pid])
        y_pred.append(int(p["hard"]))
        soft_labels.append(id_to_soft[pid])
        # Normalize probs to integer keys matching ann_labels format
        prob_dicts.append({config.CLASS_LABELS.index(k): float(v) for k, v in p["probs"].items()})

    metrics, per_class = compute_metrics(
        y_true=y_true, y_pred=y_pred,
        prob_vectors=prob_dicts, ann_labels=soft_labels, method_name=report_prefix,
    )

    df = config.load_data()["df"]
    id2row = {int(df.iloc[i]["id"]): df.iloc[i] for i in range(len(df))}
    pred_list = []
    for pid, p in preds.items():
        if pid not in id2row:
            continue
        row = id2row[pid]
        pred_list.append({
            "id": int(pid),
            "text": row["tweet_text"],
            "label": config.ID_TO_LABEL[int(p["hard"])],
            "probabilities": p["probs"],
            "hard_prediction": int(p["hard"]),
        })

    pred_dist = {config.CLASS_LABELS[j]: int(np.sum(np.array(y_pred) == j)) for j in range(3)}
    report = {
        "method": report_prefix,
        "total_instances": len(preds),
        "test_metrics": {k: float(v) for k, v in metrics.items() if k not in ("per_class", "method")},
        "per_class": per_class,
        "prediction_distribution": pred_dist,
    }
    return pred_list, report


# =================== BASE PREDICTIONS (cached) ===================

def _get_cached_base_predictions():
    """Run all base classifiers once and return their predictions.

    Each base classifier internally uses 80/20 split.
    We only use predictions for the held-out 20% test split.

    Returns:
        base_preds: dict of {classifier_name: {id: {"hard": int, "probs": {...}}}}
        test_ids: set of test IDs (held-out 20%)
        test_indices: np.ndarray of integer positions in df for the test split
    """
    data = config.load_data()
    df = data["df"]

    # Determine split ONCE - seed before computing
    _seeded = getattr(_get_cached_base_predictions, '_seeded', False)
    if not _seeded:
        np.random.seed(config.RANDOM_STATE)
        split_idx = int(len(df) * config.TRAIN_VAL_SPLIT)
        _get_cached_base_predictions._test_indices = np.random.choice(len(df), size=len(df) - split_idx, replace=False)
        _get_cached_base_predictions._test_ids = set(int(df.iloc[i]["id"]) for i in _get_cached_base_predictions._test_indices)
        _get_cached_base_predictions._seeded = True

    test_indices = _get_cached_base_predictions._test_indices
    test_ids = _get_cached_base_predictions._test_ids
    base_preds = {}

    # 1. TF-IDF + RF
    print("  [1/6] TF-IDF + RF...")
    tfidf_preds_full, _ = run_tfidf()
    base_preds["tfidf_rf"] = {p["id"]: {"hard": p["hard_prediction"], "probs": p["probabilities"]} for p in tfidf_preds_full}

    # 2. Transformer
    print("  [2/6] DistilBERT + LR...")
    trans_preds_full, _ = run_transformer()
    base_preds["transformer"] = {p["id"]: {"hard": p["hard_prediction"], "probs": p["probabilities"]} for p in trans_preds_full}

    # 3. TF-IDF + SVM
    print("  [3/6] TF-IDF + LinearSVC...")
    try:
        svm_preds, _, _ = run_svm_for_ensemble()
        base_preds["tfidf_svm"] = {p["id"]: {"hard": p["hard_prediction"], "probs": p["probabilities"]} for p in svm_preds} if svm_preds else {}
    except Exception as e:
        print(f"    SVM failed: {e}")

    # 4. TF-IDF + XGBoost
    print("  [4/6] TF-IDF + XGBoost...")
    try:
        xgb_preds, _, _ = run_xgboost_for_ensemble()
        base_preds["tfidf_xgb"] = {p["id"]: {"hard": p["hard_prediction"], "probs": p["probabilities"]} for p in xgb_preds} if xgb_preds else {}
    except Exception as e:
        print(f"    XGBoost failed: {e}")

    # 5. SBERT + LR
    print("  [5/6] SBERT + LR...")
    try:
        sbert_preds, _, _ = run_sbert_for_ensemble()
        base_preds["sbert_lr"] = {p["id"]: {"hard": p["hard_prediction"], "probs": p["probabilities"]} for p in sbert_preds} if sbert_preds else {}
    except Exception as e:
        print(f"    SBERT failed: {e}")

    # 6. Cross-encoder (may fail if model unavailable)
    print("  [6/6] Cross-encoder...")
    try:
        ce_preds, _, _ = run_cross_encoder_for_ensemble()
        base_preds["cross_encoder"] = {p["id"]: {"hard": p["hard_prediction"], "probs": p["probabilities"]} for p in ce_preds} if ce_preds else {}
    except Exception as e:
        print(f"    Cross-encoder failed: {e}")

    return base_preds, test_ids, test_indices


# =================== ENSEMBLE METHODS ===================

def soft_voting():
    """Method 1: Unweighted soft voting -- average probability vectors on held-out 20%.
    Uses all 6 base classifiers."""
    print("Method 1: Soft Voting Ensemble (unweighted, 6 classifiers)")
    base_preds, test_ids, _ = _get_cached_base_predictions()
    clf_names = [k for k in base_preds if base_preds[k]]
    print(f"  Using {len(clf_names)} classifiers: {', '.join(clf_names)}")

    ensemble = {}
    for pid in test_ids:
        all_probs = []
        for cname in clf_names:
            p = base_preds[cname].get(pid, {"probs": {l: 1/3 for l in config.CLASS_LABELS}})
            all_probs.append(np.array([p["probs"].get(l, 0) for l in config.CLASS_LABELS]))
        avg = np.mean(all_probs, axis=0)
        ensemble[pid] = {
            "hard": int(np.argmax(avg)),
            "probs": {config.CLASS_LABELS[j]: float(avg[j]) for j in range(3)},
        }
    return _evaluate(ensemble, "ensemble_soft_voting")


def weighted_soft_voting():
    """Method 2: Weighted soft voting by CV F1 scores."""
    print("Method 2: Weighted Soft Voting (by CV F1)")
    base_preds, test_ids, _ = _get_cached_base_predictions()
    clf_names = [k for k in base_preds if base_preds[k]]
    print(f"  Using {len(clf_names)} classifiers: {', '.join(clf_names)}")

    # CV F1 weights (from standalone evaluation)
    clf_f1 = {
        "tfidf_rf": 0.277, "transformer": 0.470, "tfidf_svm": 0.249,
        "tfidf_xgb": 0.302, "sbert_lr": 0.306, "cross_encoder": 0.0,
    }
    weights = np.array([clf_f1.get(k, 0.1) for k in clf_names])
    weights = weights / weights.sum()

    ensemble = {}
    for pid in test_ids:
        all_probs = []
        for i, cname in enumerate(clf_names):
            p = base_preds[cname].get(pid, {"probs": {l: 1/3 for l in config.CLASS_LABELS}})
            all_probs.append(weights[i] * np.array([p["probs"].get(l, 0) for l in config.CLASS_LABELS]))
        weighted = sum(all_probs)
        ensemble[pid] = {
            "hard": int(np.argmax(weighted)),
            "probs": {config.CLASS_LABELS[j]: float(weighted[j]) for j in range(3)},
        }
    return _evaluate(ensemble, "ensemble_weighted_voting")


def majority_voting():
    """Method 3: Hard vote on held-out 20% (all classifiers)."""
    print("Method 3: Majority Voting Ensemble (6 classifiers)")
    base_preds, test_ids, _ = _get_cached_base_predictions()
    clf_names = [k for k in base_preds if base_preds[k]]
    print(f"  Using {len(clf_names)} classifiers: {', '.join(clf_names)}")

    ensemble = {}
    for pid in test_ids:
        votes = np.zeros(3)
        for cname in clf_names:
            h = base_preds[cname].get(pid, {"hard": 2})["hard"]
            votes[h] += 1
        ensemble[pid] = {
            "hard": int(np.argmax(votes)),
            "probs": {config.CLASS_LABELS[j]: float(votes[j] / len(clf_names)) for j in range(3)},
        }
    return _evaluate(ensemble, "ensemble_majority_voting")


def feature_fusion():
    """Method 4: TF-IDF features + DistilBERT embeddings concatenated on held-out data."""
    print("Method 4: Feature-Level Fusion (TF-IDF + DistilBERT)")
    data = config.load_data()
    df = data["df"]

    # Get shared test indices
    base_preds, test_ids, test_indices = _get_cached_base_predictions()
    train_mask = np.ones(len(df), dtype=bool)
    train_mask[test_indices] = False

    train_texts = np.array(data["texts"])[train_mask]
    test_texts = np.array(data["texts"])[test_indices]
    y_test = data["majority_labels"][test_indices]

    # TF-IDF features
    tfidf = TfidfVectorizer(**config.TFIDF_DEFAULTS)
    X_train_tfidf = tfidf.fit_transform(train_texts)
    X_test_tfidf = tfidf.transform(test_texts).toarray()

    # DistilBERT features (train on train, predict on test)
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(config.TRANSFORMER_MODEL_NAME)
    try:
        model = AutoModel.from_pretrained(config.TRANSFORMER_MODEL_NAME, device_map="auto")
    except Exception:
        model = AutoModel.from_pretrained(config.TRANSFORMER_MODEL_NAME).to("cpu")
    model.eval()

    tok_train = tokenizer(list(train_texts), padding="max_length", truncation=True, max_length=config.TRANSFORMER_MAX_LENGTH)
    tok_test = tokenizer(list(test_texts), padding="max_length", truncation=True, max_length=config.TRANSFORMER_MAX_LENGTH)

    def extract_batch(tok_dict, batch_size):
        embs = []
        for i in range(0, len(tok_dict["input_ids"]), batch_size):
            bid = torch.tensor(tok_dict["input_ids"][i:i + batch_size]).to(model.device)
            am = torch.tensor(tok_dict["attention_mask"][i:i + batch_size]).to(model.device)
            with torch.no_grad():
                out = model(input_ids=bid, attention_mask=am)
            embs.append(out.last_hidden_state[:, 0, :].cpu().numpy())
        return np.vstack(embs)

    X_train_bert = extract_batch(tok_train, config.TRANSFORMER_BATCH_SIZE)
    X_test_bert = extract_batch(tok_test, config.TRANSFORMER_BATCH_SIZE)

    X_test = np.hstack([X_test_tfidf, X_test_bert])
    X_test_norm = normalize(X_test)
    X_train_norm = normalize(np.hstack([X_train_tfidf.toarray(), X_train_bert]))
    y_train = data["majority_labels"][train_mask]

    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=config.RANDOM_STATE, C=1.0)
    clf.fit(X_train_norm, y_train)

    y_pred = clf.predict(X_test_norm)
    y_proba = clf.predict_proba(X_test_norm)

    # Build predictions dict keyed by test IDs
    test_df = data["df"].iloc[test_indices]
    preds = {}
    for i, (_, row) in enumerate(test_df.iterrows()):
        preds[int(row["id"])] = {
            "hard": int(y_pred[i]),
            "probs": {config.CLASS_LABELS[j]: float(y_proba[i][j]) for j in range(3)},
        }
    return _evaluate(preds, "feature_fusion")


def sbert_approach():
    """Method 5: SBERT embeddings + LogisticRegression on held-out data."""
    print("Method 5: SBERT Embeddings + Classifier")
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("  sentence-transformers not installed. Skipping.")
        return None, None

    data = config.load_data()
    df = data["df"]

    # Use the SAME split as base classifiers (cached from first call)
    base_preds, test_ids, test_indices = _get_cached_base_predictions()
    train_mask = np.ones(len(df), dtype=bool)
    train_mask[test_indices] = False

    train_texts = np.array(data["texts"])[train_mask]
    test_texts = np.array(data["texts"])[test_indices]
    y_test = data["majority_labels"][test_indices]
    y_train = data["majority_labels"][train_mask]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    X_train = normalize(model.encode(train_texts, show_progress_bar=False), norm="l2")
    X_test = normalize(model.encode(test_texts, show_progress_bar=False), norm="l2")

    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=config.RANDOM_STATE, C=1.0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)

    test_df = data["df"].iloc[test_indices]
    preds = {}
    for i, (_, row) in enumerate(test_df.iterrows()):
        preds[int(row["id"])] = {
            "hard": int(y_pred[i]),
            "probs": {config.CLASS_LABELS[j]: float(y_proba[i][j]) for j in range(3)},
        }
    return _evaluate(preds, "sbert_minilm")


def bagging():
    """Method 6: Bagging (multiple RF models on bootstrapped data) on train split."""
    print("Method 6: Bagging Ensemble (10 RF models)")
    data = config.load_data()
    df = data["df"]

    # Use the shared test indices (same split as all other methods)
    base_preds, test_ids, test_indices = _get_cached_base_predictions()
    train_mask = np.ones(len(df), dtype=bool)
    train_mask[test_indices] = False

    train_texts = df["tweet_text"].values[train_mask]
    y_train = data["majority_labels"][train_mask]
    y_test = data["majority_labels"][test_indices]

    vectorizer = TfidfVectorizer(**config.TFIDF_DEFAULTS)
    X_train = vectorizer.fit_transform(train_texts)

    bag = BaggingClassifier(
        estimator=RandomForestClassifier(n_estimators=100, n_jobs=-1),
        n_estimators=10, max_samples=0.8,
        bootstrap=True, bootstrap_features=False,
        random_state=config.RANDOM_STATE,
    )
    bag.fit(X_train, y_train)

    X_test = vectorizer.transform(df["tweet_text"].values[test_indices])
    y_pred = bag.predict(X_test)
    all_proba = np.array([est.predict_proba(X_test) for est in bag.estimators_])
    y_proba = all_proba.mean(axis=0)

    test_df = df.iloc[test_indices]
    preds = {}
    for i, (_, row) in enumerate(test_df.iterrows()):
        preds[int(row["id"])] = {
            "hard": int(y_pred[i]),
            "probs": {config.CLASS_LABELS[j]: float(y_proba[i][j]) for j in range(3)},
        }
    return _evaluate(preds, "bagging_rf")


# =================== METHOD REGISTRY ===================

METHODS = {
    "soft_voting": soft_voting,
    "weighted_voting": weighted_soft_voting,
    "majority_voting": majority_voting,
    "feature_fusion": feature_fusion,
    "sbert": sbert_approach,
    "bagging": bagging,
}


def run_all_and_pick_best():
    """Run all ensemble methods on the same 80/20 split as standalone methods.

    This gives a fair comparison with standalone methods:
    - Base classifiers train on 80%, predict on held-out 20%
    - Ensemble combines held-out predictions
    - Metrics computed on held-out 20% (directly comparable to standalone F1)

    Returns:
        best_name: str - name of the best ensemble method
        best_metrics: dict - F1(3-class), F1(2-class), CE for the best method
        all_results: dict - all method results ranked by F1(3-class)
    """
    print("\n" + "=" * 80)
    print("ENSEMBLE & ALTERNATIVE APPROACHES")
    print("=" * 80)

    all_results = {}
    for method_name in METHODS:
        print(f"\n{'=' * 80}")
        print(f"METHOD: {method_name}")
        print(f"{'=' * 80}")

        start = time.time()
        try:
            method_fn = METHODS[method_name]
            predictions, report = method_fn()
            if predictions is None:
                print(f"  SKIPPED (prereq not available)")
                all_results[method_name] = {"f1_3class": None, "f1_2class": None, "cross_entropy": None, "status": "SKIPPED"}
                continue

            elapsed = time.time() - start

            # Save to outputs
            pred_path = os.path.join(OUTPUT_DIR, f"predictions_ensemble_{method_name}.json")
            with open(pred_path, "w") as f:
                json.dump(predictions, f, indent=2)
            report_path = os.path.join(OUTPUT_DIR, f"evaluation_report_ensemble_{method_name}.json")
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)

            m = report["test_metrics"]
            all_results[method_name] = {
                "f1_3class": m.get("f1_macro_3class"),
                "f1_2class": m.get("f1_macro_2class"),
                "cross_entropy": m.get("cross_entropy"),
                "status": "OK",
                "elapsed": elapsed,
            }
            print(f"  F1(3-class): {m.get('f1_macro_3class', 0):.4f}")
            print(f"  F1(2-class): {m.get('f1_macro_2class', 0):.4f}")
            print(f"  CE: {m.get('cross_entropy', 0):.4f}")
            print(f"  Time: {elapsed:.0f}s")
        except Exception as e:
            import traceback
            traceback.print_exc()
            all_results[method_name] = {"f1_3class": None, "f1_2class": None, "cross_entropy": None, "status": f"ERROR: {e}"}

    # Rank by F1(3-class)
    ranked = [(name, r) for name, r in all_results.items() if r["f1_3class"] is not None and r["status"] == "OK"]
    ranked.sort(key=lambda x: x[1]["f1_3class"], reverse=True)

    print(f"\n{'=' * 80}")
    print("ENSEMBLE COMPARISON (on held-out 20% test split)")
    print("=" * 80)
    print(f"  {'Method':<22} {'F1(3-class)':<14} {'F1(2-class)':<14} {'CE':<10} {'Time':<10}")
    print(f"  {'-' * 20:<22} {'-' * 12:<14} {'-' * 12:<14} {'-' * 8:<10} {'-' * 8:<10}")
    for name, r in ranked:
        f3 = f"{r['f1_3class']:.4f}"
        f2 = f"{r['f1_2class']:.4f}"
        c = f"{r['cross_entropy']:.3f}"
        print(f"  {name:<22} {f3:<14} {f2:<14} {c:<10} {r.get('elapsed', 0):.0f}s")

    if ranked:
        best_name, best = ranked[0]
        print(f"\n  >>> Best ensemble method: {best_name} (F1(3-class) = {best['f1_3class']:.4f})")
    else:
        best_name, best = None, {}
        print("\n  No ensemble methods succeeded.")

    comp = {"ensembles_and_alternatives": {name: {"f1_3class": r["f1_3class"], "f1_2class": r["f1_2class"], "cross_entropy": r["cross_entropy"]} for name, r in all_results.items()}}
    with open(os.path.join(OUTPUT_DIR, "comparison_ensembles.json"), "w") as f:
        json.dump(comp, f, indent=2)
    print(f"\nSaved to {OUTPUT_DIR}/comparison_ensembles.json")

    return best_name, best, all_results


# =================== MAIN ===================

def main():
    parser = argparse.ArgumentParser(description="Ensemble & alternative approaches for Task 1")
    parser.add_argument("--method", choices=list(METHODS.keys()) + ["all"], default="all")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.method == "all":
        run_all_and_pick_best()
    else:
        method_fn = METHODS[args.method]
        predictions, report = method_fn()
        if predictions is not None:
            pred_path = os.path.join(OUTPUT_DIR, f"predictions_ensemble_{args.method}.json")
            with open(pred_path, "w") as f:
                json.dump(predictions, f, indent=2)
            report_path = os.path.join(OUTPUT_DIR, f"evaluation_report_ensemble_{args.method}.json")
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\nSaved predictions to {pred_path}")
            print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
