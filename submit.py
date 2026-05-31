"""
Submission script for MediaEval 2026 Enthymeme Detection.

Pipeline:
1. Compute ensemble weights from 5-fold CV on 80% training set (dynamic, not hardcoded)
2. Compare ALL standalone AND ensemble methods on the SAME held-out 20% set
3. Select the best overall approach (standalone or ensemble)
4. Train the winning approach on ALL 1333 instances for final submission

Outputs:
  outputs/submit_task1_classifiers.json  - Challenge format predictions
  outputs/submit_task2_propositions.json - Challenge format propositions
"""

import argparse
import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import normalize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

warnings.filterwarnings("ignore")

OUTPUT_DIR = config.OUTPUT_DIR
TEST_CSV = config.TEST_CSV_PATH


# ============== TASK 1: CLASSIFICATION SUBMISSION ==============


def _extract_svm_proba(texts, svm_art):
    """Apply SVM classifier to texts and return probability matrix."""
    clf = svm_art["calibrated"]
    all_proba = clf.predict_proba(texts)
    if all_proba.shape[1] < 3:
        padded = np.zeros((len(texts), 3))
        for j, cls_idx in enumerate(clf.classes_):
            padded[:, cls_idx] = all_proba[:, j]
        all_proba = padded
    return all_proba


def _extract_xgb_proba(texts, xgb_art):
    """Apply XGBoost classifier to texts and return probability matrix."""
    clf = xgb_art["classifier"]
    vec = xgb_art["vectorizer"]
    X = np.array(vec.transform(texts).toarray())
    all_proba = clf.predict_proba(X)
    if all_proba.shape[1] < 3:
        padded = np.zeros((len(texts), 3))
        for j, cls_idx in enumerate(clf.classes_):
            padded[:, cls_idx] = all_proba[:, j]
        all_proba = padded
    return all_proba


def _extract_sbert_proba(texts, sbert_art):
    """Apply SBERT classifier to texts and return probability matrix."""
    from sentence_transformers import SentenceTransformer
    model = sbert_art["model"]
    clf = sbert_art["classifier"]
    X = model.encode(list(texts), show_progress_bar=False, normalize_embeddings=True)
    return clf.predict_proba(X)


def _extract_transformer_proba(texts, trans_art):
    """Apply Transformer feature extraction + classifier to texts."""
    tokenizer = trans_art["tokenizer"]
    model = trans_art["model"]
    clf = trans_art["clf"]
    embs = []
    for i in range(0, len(texts), config.TRANSFORMER_BATCH_SIZE):
        tok_batch = tokenizer(texts[i:i + config.TRANSFORMER_BATCH_SIZE], padding="max_length", truncation=True, max_length=config.TRANSFORMER_MAX_LENGTH)
        batch_ids = torch.tensor(tok_batch["input_ids"]).to(model.device)
        batch_mask = torch.tensor(tok_batch["attention_mask"]).to(model.device)
        with torch.no_grad():
            outputs = model(input_ids=batch_ids, attention_mask=batch_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embs.append(normalize(cls_emb, norm='l2'))
    X = np.vstack(embs)
    return clf.predict_proba(X)


def _extract_tfidf_proba(texts, tfidf_art):
    """Apply TF-IDF + RF classifier to texts."""
    X = tfidf_art["vectorizer"].transform(texts).toarray()
    return tfidf_art["model"].predict_proba(X)


def _apply_ensemble_proba(proba_dict, weights_dict):
    """Weighted soft voting across all classifiers.

    Args:
        proba_dict: {name: (n_samples, 3) probability matrix}
        weights_dict: {name: weight}
    Returns:
        (n_samples, 3) weighted probability matrix
    """
    names = [n for n in weights_dict if n in proba_dict]
    if not names:
        return np.ones((1, 3)) / 3

    w = np.array([weights_dict[n] for n in names])
    w = w / w.sum()
    matrices = [proba_dict[n] for n in names]
    result = sum(w[i] * matrices[i] for i in range(len(names)))
    return result


def compute_ensemble_weights():
    """Compute ensemble weights from 5-fold CV on the 80% training set.

    Each classifier is trained on the 80% training split. CV F1 scores
    determine the ensemble weights (normalized). No hardcoded constants.

    Returns:
        {name: weight} dict with normalized weights
    """
    print("\nComputing ensemble weights from 5-fold CV on training set...")

    # Import the standalone classifier functions that do proper CV
    from task1_classification.task1_classifier_tfidf import run_tfidf
    from task1_classification.transformer.transformer import run_transformer
    from task1_classification.new_classifiers import run_svm, run_xgboost, run_sbert

    cv_scores = {}

    # TF-IDF + RF
    print("  1/5 TF-IDF + RF...")
    preds, rep = run_tfidf()
    f1 = rep.get("test_metrics", {}).get("f1_macro_3class")
    cv_scores["tfidf_rf"] = f1
    print(f"    F1(3-class)={f1:.4f}")

    # Transformer
    print("  2/5 Transformer...")
    preds, rep = run_transformer()
    f1 = rep.get("test_metrics", {}).get("f1_macro_3class")
    cv_scores["transformer"] = f1
    print(f"    F1(3-class)={f1:.4f}")

    # SVM
    print("  3/5 TF-IDF + LinearSVC...")
    preds, rep = run_svm()
    f1 = rep.get("test_metrics", {}).get("f1_macro_3class")
    cv_scores["tfidf_svm"] = f1
    print(f"    F1(3-class)={f1:.4f}")

    # XGBoost
    print("  4/5 TF-IDF + XGBoost...")
    preds, rep = run_xgboost()
    f1 = rep.get("test_metrics", {}).get("f1_macro_3class")
    cv_scores["tfidf_xgb"] = f1
    print(f"    F1(3-class)={f1:.4f}")

    # SBERT
    print("  5/5 SBERT + LR...")
    preds, rep = run_sbert()
    f1 = rep.get("test_metrics", {}).get("f1_macro_3class")
    cv_scores["sbert_lr"] = f1
    print(f"    F1(3-class)={f1:.4f}")

    # Normalize weights
    scores = {k: v for k, v in cv_scores.items() if v is not None}
    if not scores:
        print("  WARNING: No valid CV scores. Using equal weights.")
        return {k: 1.0 for k in cv_scores}

    total = sum(scores.values())
    weights = {k: round(v / total, 4) for k, v in scores.items()}

    print(f"\n  Computed weights: {weights}")
    print(f"  Weight sum: {sum(weights.values()):.4f}")
    return weights


def compare_standalone_vs_ensemble():
    """Run all standalone AND ensemble methods on the SAME held-out 20% set.

    Returns the best method (standalone or ensemble) and its metrics.
    """
    import time

    # ---- Run all standalone methods ----
    print("\n" + "=" * 80)
    print("COMPARISON: Running all standalone methods on held-out 20%")
    print("=" * 80)

    from run_methods import TASK1_METHODS

    results = {}
    time_start = time.time()

    for method_key, (display, run_fn) in TASK1_METHODS.items():
        print(f"\n  --> {display}")
        start = time.time()
        try:
            predictions, report = run_fn()
            elapsed = time.time() - start

            metrics = report.get("test_metrics", report.get("final_eval", {}))
            f1_3 = metrics.get("f1_macro_3class") or metrics.get("f1_3class")
            f1_2 = metrics.get("f1_macro_2class") or metrics.get("f1_2class")
            ce = metrics.get("cross_entropy")

            results[method_key] = {
                "display": display,
                "f1_3class": f1_3, "f1_2class": f1_2, "cross_entropy": ce,
                "elapsed": elapsed, "type": "standalone",
            }
            print(f"      F1(3)={f1_3:.4f}  F1(2)={f1_2:.4f}  CE={ce:.3f}  ({elapsed:.0f}s)")
        except Exception as e:
            elapsed = time.time() - start
            print(f"      FAILED ({elapsed:.0f}s): {e}")
            results[method_key] = {
                "display": display,
                "f1_3class": None, "f1_2class": None, "cross_entropy": None,
                "elapsed": elapsed, "type": "standalone",
            }

    # ---- Run all ensemble methods ----
    print("\n" + "=" * 80)
    print("COMPARISON: Running all ensemble methods on held-out 20%")
    print("=" * 80)

    from task1_classification.task1_ensemble import METHODS as ENSEMBLE_METHODS

    for method_key in ENSEMBLE_METHODS:
        print(f"\n  --> {method_key}")
        start = time.time()
        try:
            method_fn = ENSEMBLE_METHODS[method_key]
            predictions, report = method_fn()
            if predictions is None:
                print(f"      SKIPPED (prereq not available)")
                results[method_key] = {
                    "display": method_key, "f1_3class": None, "f1_2class": None,
                    "cross_entropy": None, "elapsed": time.time() - start, "type": "ensemble",
                }
                continue

            elapsed = time.time() - start

            f1_3 = report["test_metrics"].get("f1_macro_3class")
            f1_2 = report["test_metrics"].get("f1_macro_2class")
            ce = report["test_metrics"].get("cross_entropy")

            results[method_key] = {
                "display": method_key,
                "f1_3class": f1_3, "f1_2class": f1_2, "cross_entropy": ce,
                "elapsed": elapsed, "type": "ensemble",
            }
            print(f"      F1(3)={f1_3:.4f}  F1(2)={f1_2:.4f}  CE={ce:.3f}  ({elapsed:.0f}s)")
        except Exception as e:
            elapsed = time.time() - start
            print(f"      FAILED ({elapsed:.0f}s): {e}")
            results[method_key] = {
                "display": method_key, "f1_3class": None, "f1_2class": None,
                "cross_entropy": None, "elapsed": elapsed, "type": "ensemble",
            }

    # ---- Combined ranking ----
    print("\n" + "=" * 80)
    print("COMBINED RANKING (by F1 3-class)")
    print("=" * 80)

    ranked = [(name, r) for name, r in results.items() if r["f1_3class"] is not None]
    ranked.sort(key=lambda x: x[1]["f1_3class"], reverse=True)

    if not ranked:
        print("\n  ERROR: No method succeeded.")
        return None, None, None

    print(f"\n  {'Rank':<6} {'Method':<22} {'F1(3)':<10} {'F1(2)':<10} {'CE':<10} {'Type':<10}")
    print(f"  {'-' * 4:<6} {'-' * 20:<22} {'-' * 8:<10} {'-' * 8:<10} {'-' * 8:<10} {'-' * 8:<10}")
    for i, (name, r) in enumerate(ranked, 1):
        f3 = f"{r['f1_3class']:.4f}"
        f2 = f"{r['f1_2class']:.4f}"
        c = f"{r['cross_entropy']:.3f}" if r['cross_entropy'] is not None else "N/A"
        print(f"  {i:<6} {r['display']:<22} {f3:<10} {f2:<10} {c:<10} {r['type']:<10}")

    best_name, best = ranked[0]
    print(f"\n  >>> Best method: {best_name} ({best['type']}) (F1(3-class) = {best['f1_3class']:.4f})")

    elapsed = time.time() - time_start
    print(f"  Total comparison time: {elapsed:.0f}s")

    return best_name, best, results


def _get_available_classifiers():
    """Check which classifiers are available and return their names."""
    classifiers = ["tfidf_rf", "transformer", "tfidf_svm"]
    try:
        import xgboost
        classifiers.append("tfidf_xgb")
    except ImportError:
        pass
    classifiers.append("sbert_lr")
    return classifiers


def submit_task1(weights=None):
    """Train all available classifiers on full data, generate test predictions.

    Args:
        weights: {name: weight} dict for weighted voting.
                 If None, computes weights from CV on 80% training set.

    Returns:
        (test_submission, full_submission)
    """
    print("=" * 60)
    print("TASK 1: TRAINING ALL CLASSIFIERS ON FULL DATA")
    print("=" * 60)

    df = config.load_data()["df"]
    all_texts = df["tweet_text"].tolist()

    # ---- Step 1: Compute or use provided weights ----
    if weights is None:
        weights = compute_ensemble_weights()

    # ---- Step 2: Train all classifiers on full data ----
    print("\nTraining all classifiers on ALL 1333 instances...")
    classifiers = _get_available_classifiers()
    available = {}
    all_f1 = {}

    for i, clf_name in enumerate(classifiers, 1):
        print(f"\n{i}/5 Training {clf_name} on full data...")
        try:
            if clf_name == "tfidf_rf":
                preds, rep, art = train_and_predict_tfidf_for_ensemble()
                f1 = rep.get("cv_scores", {}).get("macro_f1_3class")
                available[clf_name] = ("tfidf", art)
            elif clf_name == "transformer":
                preds, rep, art = train_and_predict_transformer_for_ensemble()
                f1 = rep.get("cv_scores", {}).get("macro_f1_3class")
                available[clf_name] = ("transformer", art)
            elif clf_name == "tfidf_svm":
                preds, rep, art = train_and_predict_svm_for_ensemble()
                f1 = rep.get("cv_scores", {}).get("macro_f1_3class")
                available[clf_name] = ("svm", art)
            elif clf_name == "tfidf_xgb":
                preds, rep, art = train_and_predict_xgboost_for_ensemble()
                if preds is None:
                    print("   [XGBoost] unavailable, skipping")
                    continue
                f1 = rep.get("cv_scores", {}).get("macro_f1_3class")
                available[clf_name] = ("xgb", art)
            elif clf_name == "sbert_lr":
                preds, rep, art = train_and_predict_sbert_for_ensemble()
                f1 = rep.get("cv_scores", {}).get("macro_f1_3class")
                available[clf_name] = ("sbert", art)
            else:
                continue
            all_f1[clf_name] = f1
            print(f"   F1(3-class)={f1:.4f}")
        except Exception as e:
            print(f"   ERROR: {e}")

    if not available:
        print("ERROR: No classifiers trained successfully.")
        return None, None

    # ---- Step 3: Generate test set predictions ----
    print("\nGenerating predictions for test set...")
    test_df = pd.read_csv(TEST_CSV)
    test_ids = test_df["id"].tolist()
    test_texts = test_df["tweet_text"].tolist()

    # Compute per-classifier predictions on test set
    proba_test = {}
    for name, (clf_type, art) in available.items():
        try:
            if clf_type == "tfidf":
                proba_test[name] = _extract_tfidf_proba(test_texts, art)
            elif clf_type == "transformer":
                proba_test[name] = _extract_transformer_proba(test_texts, art)
            elif clf_type == "svm":
                proba_test[name] = _extract_svm_proba(test_texts, art)
            elif clf_type == "xgb":
                proba_test[name] = _extract_xgb_proba(test_texts, art)
            elif clf_type == "sbert":
                proba_test[name] = _extract_sbert_proba(test_texts, art)
        except Exception as e:
            print(f"  WARNING: Could not extract probabilities for {name}: {e}")

    # ---- Step 4: Apply ensemble (weighted voting) ----
    print("Applying weighted soft voting...")
    ensemble_proba = _apply_ensemble_proba(proba_test, weights)

    test_submission = []
    pred_dist = {"premise": 0, "conclusion": 0, "none": 0}
    for i, pid in enumerate(test_ids):
        hard = int(np.argmax(ensemble_proba[i]))
        probs = {config.CLASS_LABELS[j]: float(ensemble_proba[i][j]) for j in range(3)}
        test_submission.append({
            "id": int(pid),
            "text": test_texts[i],
            "label": config.ID_TO_LABEL[hard],
            "probabilities": probs,
            "hard_prediction": hard,
        })
        pred_dist[config.ID_TO_LABEL[hard]] += 1

    # ---- Step 5: Generate full dataset predictions ----
    print("Generating predictions for full dataset...")
    proba_full = {}
    for name, (clf_type, art) in available.items():
        try:
            if clf_type == "tfidf":
                proba_full[name] = _extract_tfidf_proba(all_texts, art)
            elif clf_type == "transformer":
                proba_full[name] = _extract_transformer_proba(all_texts, art)
            elif clf_type == "svm":
                proba_full[name] = _extract_svm_proba(all_texts, art)
            elif clf_type == "xgb":
                proba_full[name] = _extract_xgb_proba(all_texts, art)
            elif clf_type == "sbert":
                proba_full[name] = _extract_sbert_proba(all_texts, art)
        except Exception as e:
            pass

    full_ensemble_proba = _apply_ensemble_proba(proba_full, weights)

    full_submission = []
    for i, (_, row) in enumerate(df.iterrows()):
        hard = int(np.argmax(full_ensemble_proba[i]))
        probs = {config.CLASS_LABELS[j]: float(full_ensemble_proba[i][j]) for j in range(3)}
        full_submission.append({
            "id": int(row["id"]),
            "text": row["tweet_text"],
            "label": config.ID_TO_LABEL[hard],
            "probabilities": probs,
            "hard_prediction": hard,
        })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "submit_task1_test.json"), "w") as f:
        json.dump(test_submission, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "submit_task1_classifiers.json"), "w") as f:
        json.dump(full_submission, f, indent=2)

    print(f"   Test predictions: {len(test_submission)} -> {OUTPUT_DIR}/submit_task1_test.json")
    print(f"   Full dataset predictions: {len(full_submission)} -> {OUTPUT_DIR}/submit_task1_classifiers.json")
    print(f"   Prediction distribution: {pred_dist}")
    print(f"   Ensemble weights: {weights}")
    print(f"   Available classifiers: {list(available.keys())}")

    return test_submission, full_submission


def _submit_standalone_on_all_data(method_name):
    """Retrain a specific standalone classifier on ALL 1333 instances and submit.

    Args:
        method_name: one of "tfidf", "transformer", "svm", "xgboost", "sbert"

    Returns:
        (test_submission, full_submission) in challenge format
    """
    from task1_classification.task1_classifier_tfidf import run_tfidf_for_ensemble
    from task1_classification.transformer.transformer import run_transformer_for_ensemble
    from task1_classification.new_classifiers import run_svm_for_ensemble, run_xgboost_for_ensemble, run_sbert_for_ensemble

    train_fn_map = {
        "tfidf": run_tfidf_for_ensemble,
        "transformer": run_transformer_for_ensemble,
        "svm": run_svm_for_ensemble,
        "xgboost": run_xgboost_for_ensemble,
        "sbert": run_sbert_for_ensemble,
    }

    extractor_map = {
        "tfidf": _extract_tfidf_proba,
        "transformer": _extract_transformer_proba,
        "svm": _extract_svm_proba,
        "xgboost": _extract_xgb_proba,
        "sbert": _extract_sbert_proba,
    }

    df = config.load_data()["df"]
    all_texts = df["tweet_text"].tolist()
    test_df = pd.read_csv(TEST_CSV)
    test_ids = test_df["id"].tolist()
    test_texts = test_df["tweet_text"].tolist()

    print(f"  Training {method_name} on ALL {len(df)} instances...")
    preds, rep, art = train_fn_map[method_name]()
    print(f"  F1(3-class)={rep.get('cv_scores', {}).get('macro_f1_3class', 'N/A'):.4f}")

    # Apply the trained model to test and full datasets
    test_proba = extractor_map[method_name](test_texts, art)
    full_proba = extractor_map[method_name](all_texts, art)

    # Generate submissions
    test_submission = []
    for i, pid in enumerate(test_ids):
        hard = int(np.argmax(test_proba[i]))
        probs = {config.CLASS_LABELS[j]: float(test_proba[i][j]) for j in range(3)}
        test_submission.append({
            "id": int(pid), "text": test_texts[i],
            "label": config.ID_TO_LABEL[hard],
            "probabilities": probs, "hard_prediction": hard,
        })

    full_submission = []
    for i, (_, row) in enumerate(df.iterrows()):
        hard = int(np.argmax(full_proba[i]))
        probs = {config.CLASS_LABELS[j]: float(full_proba[i][j]) for j in range(3)}
        full_submission.append({
            "id": int(row["id"]), "text": row["tweet_text"],
            "label": config.ID_TO_LABEL[hard],
            "probabilities": probs, "hard_prediction": hard,
        })

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "submit_task1_test.json"), "w") as f:
        json.dump(test_submission, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "submit_task1_classifiers.json"), "w") as f:
        json.dump(full_submission, f, indent=2)

    print(f"   Saved {OUTPUT_DIR}/submit_task1_test.json ({len(test_submission)} entries)")
    print(f"   Saved {OUTPUT_DIR}/submit_task1_classifiers.json ({len(full_submission)} entries)")
    return test_submission, full_submission


def train_and_predict_tfidf():
    """Import TF-IDF + RF predictions from task1_classifier_tfidf on ALL data."""
    from task1_classification.task1_classifier_tfidf import get_full_data_predictions
    return get_full_data_predictions(), "tfidf"


def train_and_predict_transformer():
    """Import DistilBERT + LR predictions from transformer.py on ALL data."""
    from task1_classification.transformer.transformer import get_full_data_predictions
    return get_full_data_predictions(), "transformer"


def train_and_predict_tfidf_for_ensemble():
    """Train TF-IDF + RF on ALL data, return (predictions, report, artifacts)."""
    from task1_classification.task1_classifier_tfidf import run_tfidf_for_ensemble
    preds, rep, art = run_tfidf_for_ensemble()
    return preds, rep, art


def train_and_predict_transformer_for_ensemble():
    """Train DistilBERT + LR on ALL data, return (predictions, report, artifacts)."""
    from task1_classification.transformer.transformer import run_transformer_for_ensemble
    preds, rep, art = run_transformer_for_ensemble()
    return preds, rep, art


def train_and_predict_svm_for_ensemble():
    """Train TF-IDF + LinearSVC on ALL data."""
    from task1_classification.new_classifiers import run_svm_for_ensemble
    preds, rep, art = run_svm_for_ensemble()
    return preds, rep, art


def train_and_predict_xgboost_for_ensemble():
    """Train TF-IDF + XGBoost on ALL data."""
    from task1_classification.new_classifiers import run_xgboost_for_ensemble
    preds, rep, art = run_xgboost_for_ensemble()
    if art is None or art.get("classifier") is None:
        return None, None, None
    return preds, rep, art


def train_and_predict_sbert_for_ensemble():
    """Train SBERT + LR on ALL data."""
    from task1_classification.new_classifiers import run_sbert_for_ensemble
    preds, rep, art = run_sbert_for_ensemble()
    return preds, rep, art


# ============== TASK 2: PROPOSITION GENERATION SUBMISSION ==============

def submit_task2_t5():
    """Generate propositions using fine-tuned T5 (+ LoRA).

    Trains from scratch if no pre-trained model exists, then generates
    propositions for test predictions. Delegates to task2_generation/t5_finetune.py.
    """
    from task2_generation.t5_finetune import train_and_generate
    return train_and_generate()


def submit_task2_ollama():
    """Generate propositions using Ollama LLM (gemma4 > qwen3.6 > mistral).

    Delegates to task2_generation/ollama_generator.py.
    """
    from task2_generation.ollama_generator import generate_with_ollama
    return generate_with_ollama()


# ============== RUN ALL & SUBMIT ==============

def run_all_and_submit():
    """Run all classifiers, all ensemble methods, compare, pick best, submit.

    Pipeline:
    1. Compute ensemble weights from CV on 80% training set
    2. Compare ALL standalone and ensemble methods on the SAME held-out 20%
    3. Select the best overall approach
    4. Train the winning approach on ALL 1333 instances for submission

    Returns:
        (best_method_name, best_metrics_dict, test_submission, full_submission)
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Step 1: Compute ensemble weights ----
    weights = compute_ensemble_weights()

    # ---- Step 2: Compare all methods on held-out 20% ----
    best_name, best, all_results = compare_standalone_vs_ensemble()
    if not best_name:
        return None, None, None, None

    # ---- Step 3: Generate submission ----
    print("\n" + "=" * 80)
    print(f"Generating submission with {best_name} ({best.get('type', 'unknown')})...")
    print("=" * 80)

    # If best is standalone, re-train it on ALL data and submit directly
    if best.get("type") == "standalone":
        print(f"\n  Best is standalone ({best_name}). Retraining on ALL 1333 instances...")
        submission, full_submission = _submit_standalone_on_all_data(best_name)
    else:
        print(f"\n  Best is ensemble ({best_name}). Training all classifiers on ALL 1333 instances...")
        submission, full_submission = submit_task1(weights=weights)

    print(f"\n  Test predictions saved to {OUTPUT_DIR}/submit_task1_test.json")
    print(f"  Full dataset predictions saved to {OUTPUT_DIR}/submit_task1_classifiers.json")

    pred_dist = {config.ID_TO_LABEL[j]: 0 for j in range(3)}
    for p in full_submission:
        pred_dist[p["label"]] = pred_dist.get(p["label"], 0) + 1
    print(f"\n  Prediction distribution: {pred_dist}")

    # ---- Task 2: Evaluate T5 vs Ollama and generate propositions ----
    print("\n" + "=" * 60)
    print("TASK 2: METHOD SELECTION (T5 vs Ollama)")
    print("=" * 60)

    # Run evaluation on validation data to decide T5 vs Ollama
    chosen = None
    try:
        from task2_generation.eval_generation import compare_and_choose
        chosen, t5_metrics, ollama_metrics = compare_and_choose()
        if chosen:
            print(f"  Chosen: {chosen.upper()} (data-driven from validation set)")
    except Exception as e:
        print(f"  Evaluation failed ({e}), defaulting to T5")

    if chosen == "ollama":
        task2_props = submit_task2_ollama()
        if not task2_props:
            print("\n  Ollama generation failed, falling back to T5...")
            task2_props = submit_task2_t5()
    elif chosen == "t5":
        task2_props = submit_task2_t5()
        if not task2_props:
            print("\n  T5 not available. Trying Ollama fallback...")
            task2_props = submit_task2_ollama()
    else:
        # Default fallback to T5
        task2_props = submit_task2_t5()
        if not task2_props:
            print("\n  T5 not available. Trying Ollama fallback...")
            task2_props = submit_task2_ollama()

    if task2_props:
        prop_path = os.path.join(OUTPUT_DIR, "submit_task2_propositions.json")
        with open(prop_path, "w") as f:
            json.dump(task2_props, f, indent=2)
        gen_count = sum(1 for p in task2_props if p.get("generated_proposition"))
        print(f"\nTask 2 submission complete ({gen_count}/{len(task2_props)} generated)")
        print(f"   Saved to {prop_path}")
    else:
        print("\nTask 2: No propositions generated.")

    print(f"\n{'=' * 80}")
    print("DONE!")
    print(f"  Best method:   {best_name} ({best.get('type', 'unknown')})")
    print(f"  F1(3-class):   {best['f1_3class']:.4f}")
    print(f"  F1(2-class):   {best['f1_2class']:.4f}")
    print(f"  Ensemble weights: {weights}")
    print(f"  Task 2 method: {chosen or 't5 (default)'}")
    print(f"{'=' * 80}")

    return best_name, best, submission, full_submission


# ============== MAIN ==============

def main():
    parser = argparse.ArgumentParser(description="MediaEval 2026 Submission Generator")
    parser.add_argument("--task1", action="store_true", help="Generate Task 1 submission only")
    parser.add_argument("--task2", action="store_true", help="Generate Task 2 submission only")
    parser.add_argument("--t5", action="store_true", help="Use T5 for Task 2")
    parser.add_argument("--ollama", action="store_true", help="Use Ollama for Task 2")
    parser.add_argument("--all", action="store_true", help="Generate all submissions")
    parser.add_argument("--run-all", action="store_true", help="Run all methods, compare, pick best, submit")
    args, _ = parser.parse_known_args()

    do_task1 = args.task1 or not args.task2 and not args.t5 and not args.ollama and not args.all and not args.run_all
    do_task2 = args.task2 or args.t5 or args.ollama or args.all
    if args.all:
        do_task1, do_task2 = True, True

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.run_all:
        best_name, best, sub, full = run_all_and_submit()
        if best_name:
            print(f"\nBest method: {best_name} (F1(3)={best['f1_3class']:.4f})")
        return

    if do_task1:
        _, full_preds = submit_task1()
        print("\nTask 1 submission complete.")

    if do_task2:
        if args.ollama:
            props = submit_task2_ollama()
        else:
            props = submit_task2_t5()
        if props:
            print("\nTask 2 submission complete.")
        else:
            print("\nTask 2: No propositions generated.")


if __name__ == "__main__":
    main()
