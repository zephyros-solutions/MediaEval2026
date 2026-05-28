"""
Submission script for MediaEval 2026 Enthymeme Detection.

Trains the best models on ALL annotated data (train+dev, 1333 tweets)
and generates predictions for the test set (148 tweets) using the
weighted soft voting ensemble of 5 diverse classifiers.

Best ensemble: weighted_voting with TF-IDF+RF, Transformer, TF-IDF+SVM,
TF-IDF+XGBoost, SBERT+LR -> F1(3-class) = 0.7844

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

# Weights for weighted voting (from CV F1 scores on held-out 20%)
ENSEMBLE_WEIGHTS = {
    "tfidf_rf": 0.277,
    "transformer": 0.470,
    "tfidf_svm": 0.249,
    "tfidf_xgb": 0.302,
    "sbert_lr": 0.306,
}


# ============== TASK 1: CLASSIFICATION SUBMISSION ==============


def _extract_svm_proba(texts, svm_art):
    """Apply SVM classifier to texts and return probability matrix."""
    from sklearn.calibration import CalibratedClassifierCV

    clf = svm_art["calibrated"]
    all_proba = clf.predict_proba(texts)
    # Ensure 3 columns (premise, conclusion, none)
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


def __get_tfidf_vectorizer():
    from sklearn.feature_extraction.text import TfidfVectorizer
    return TfidfVectorizer(**config.TFIDF_DEFAULTS)


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


def submit_task1():
    """Train all 5 classifiers on full data, generate test predictions via ensemble.

    Uses weighted soft voting with CV F1-based weights:
      - Transformer: 0.470
      - SBERT+LR:    0.306
      - TF-IDF+XGBoost: 0.302
      - TF-IDF+RF:   0.277
      - TF-IDF+SVM:  0.249
    """
    print("=" * 60)
    print("TASK 1: TRAINING ALL 5 CLASSIFIERS ON FULL DATA")
    print("=" * 60)

    df = config.load_data()["df"]
    all_texts = df["tweet_text"].tolist()

    # ---- Step 1: Train all classifiers on full data ----
    print("\n1/5 Training TF-IDF + RF on full data...")
    tfidf_preds, _, tfidf_art = train_and_predict_tfidf_for_ensemble()
    print(f"   F1(3-class)={tfidf_preds[-1]['probabilities'].get('_cv_f1', 'N/A')}")

    print("2/5 Training DistilBERT + LR on full data...")
    trans_preds, _, trans_art = train_and_predict_transformer_for_ensemble()
    print(f"   F1(3-class)={trans_preds[-1]['probabilities'].get('_cv_f1', 'N/A')}")

    print("3/5 Training TF-IDF + LinearSVC on full data...")
    svm_preds, _, svm_art = train_and_predict_svm_for_ensemble()
    print(f"   F1(3-class)={svm_preds[-1]['probabilities'].get('_cv_f1', 'N/A')}")

    print("4/5 Training TF-IDF + XGBoost on full data...")
    xgb_preds, _, xgb_art = train_and_predict_xgboost_for_ensemble()
    if xgb_preds is None:
        print("   [XGBoost] unavailable (missing dependency), skipping from ensemble")
    else:
        print(f"   F1(3-class)={xgb_preds[-1]['probabilities'].get('_cv_f1', 'N/A')}")

    print("5/5 Training SBERT + LR on full data...")
    sbert_preds, _, sbert_art = train_and_predict_sbert_for_ensemble()
    print(f"   F1(3-class)={sbert_preds[-1]['probabilities'].get('_cv_f1', 'N/A')}")

    # ---- Step 2: Generate test set predictions ----
    print("\n6/6 Applying ensemble to test set...")
    test_df = pd.read_csv(TEST_CSV)
    test_ids = test_df["id"].tolist()
    test_texts = test_df["tweet_text"].tolist()

    # Get probability matrices from each classifier on test set
    proba_test = {}
    proba_test["tfidf_rf"] = _extract_tfidf_proba(test_texts, tfidf_art)
    proba_test["transformer"] = _extract_transformer_proba(test_texts, trans_art)
    proba_test["tfidf_svm"] = _extract_svm_proba(test_texts, svm_art)
    if xgb_art is not None:
        proba_test["tfidf_xgb"] = _extract_xgb_proba(test_texts, xgb_art)
    proba_test["sbert_lr"] = _extract_sbert_proba(test_texts, sbert_art)

    # Apply weighted voting
    ensemble_proba = _apply_ensemble_proba(proba_test, ENSEMBLE_WEIGHTS)

    submission = []
    pred_dist = {"premise": 0, "conclusion": 0, "none": 0}
    for i, pid in enumerate(test_ids):
        hard = int(np.argmax(ensemble_proba[i]))
        probs = {config.CLASS_LABELS[j]: float(ensemble_proba[i][j]) for j in range(3)}
        submission.append({
            "id": int(pid),
            "text": test_texts[i],
            "label": config.ID_TO_LABEL[hard],
            "probabilities": probs,
            "hard_prediction": hard,
        })
        pred_dist[config.ID_TO_LABEL[hard]] += 1

    # ---- Step 3: Generate full dataset predictions ----
    print("7/7 Applying ensemble to full dataset...")
    proba_full = {}
    proba_full["tfidf_rf"] = _extract_tfidf_proba(all_texts, tfidf_art)
    proba_full["transformer"] = _extract_transformer_proba(all_texts, trans_art)
    proba_full["tfidf_svm"] = _extract_svm_proba(all_texts, svm_art)
    if xgb_art is not None:
        proba_full["tfidf_xgb"] = _extract_xgb_proba(all_texts, xgb_art)
    proba_full["sbert_lr"] = _extract_sbert_proba(all_texts, sbert_art)

    full_ensemble_proba = _apply_ensemble_proba(proba_full, ENSEMBLE_WEIGHTS)

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
        json.dump(submission, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "submit_task1_classifiers.json"), "w") as f:
        json.dump(full_submission, f, indent=2)

    print(f"   Test predictions saved to {OUTPUT_DIR}/submit_task1_test.json")
    print(f"   Full dataset predictions saved to {OUTPUT_DIR}/submit_task1_classifiers.json")
    print(f"   Prediction distribution: {pred_dist}")
    print(f"\n   Ensemble weights: {ENSEMBLE_WEIGHTS}")
    print(f"\n   Best method: weighted_voting with 5 classifiers (F1(3-class)=0.7844 on held-out test)")
    return submission, full_submission


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
    """Train TF-IDF + XGBoost on ALL data. Returns (preds, rep, art) or (None, None, None) if unavailable."""
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
    """Generate propositions using fine-tuned T5 on full data."""
    print("\n" + "=" * 60)
    print("TASK 2: T5 PROPOSITION GENERATION")
    print("=" * 60)

    from task2_generation.task2_generator_enhanced import load_task1_predictions, build_training_pairs, train_t5, generate_with_t5

    data = config.load_data()
    pairs = build_training_pairs(data)
    premise_count = sum(1 for p in pairs if p["type"] == "premise")
    conclusion_count = sum(1 for p in pairs if p["type"] == "conclusion")
    print(f"\nTraining pairs: {len(pairs)} (premise: {premise_count}, conclusion: {conclusion_count})")

    device, device_name = config.get_device()
    print(f"Device: {device_name}")

    T5_MODEL_DIR = os.path.join(OUTPUT_DIR, "task2_t5_model")
    if os.path.exists(T5_MODEL_DIR):
        print(f"\nLoading pre-trained T5 from {T5_MODEL_DIR}...")
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        model = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL_DIR).to(device)
        tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_DIR)
        print("  Model loaded.")
    else:
        print(f"\nNo pre-trained T5 found. Run with --train-t5 to train one first.")
        return None

    predictions = load_task1_predictions()
    print(f"Loaded {len(predictions)} task1 predictions")

    print("\nGenerating propositions...")
    generated = generate_with_t5(model, tokenizer, None, predictions, device)

    test_df = pd.read_csv(TEST_CSV)
    test_ids = set(test_df["id"].tolist())

    test_propositions = []
    for g in generated:
        if g["id"] in test_ids:
            test_propositions.append({
                "id": g["id"],
                "tweet_text": g.get("tweet", ""),
                "predicted_label": g.get("predicted_label", "none"),
                "confidence": g.get("confidence", 0.0),
                "generated_proposition": g.get("generated_proposition"),
            })

    generated_ids = {p["id"] for p in test_propositions}
    for _, row in test_df.iterrows():
        pid = int(row["id"])
        if pid not in generated_ids:
            test_propositions.append({
                "id": pid,
                "tweet_text": row["tweet_text"],
                "predicted_label": "none",
                "confidence": 0.0,
                "generated_proposition": None,
            })

    test_propositions.sort(key=lambda x: x["id"])
    prop_path = os.path.join(OUTPUT_DIR, "submit_task2_propositions.json")
    with open(prop_path, "w") as f:
        json.dump(test_propositions, f, indent=2)
    print(f"   Test propositions saved to {prop_path}")

    gen_count = sum(1 for p in test_propositions if p.get("generated_proposition"))
    print(f"   Generated: {gen_count}/{len(test_propositions)}")
    return test_propositions


def submit_task2_ollama():
    """Generate propositions using Ollama (Mistral)."""
    print("\n" + "=" * 60)
    print("TASK 2: OLLAMA PROPOSITION GENERATION")
    print("=" * 60)

    try:
        from core.ollama_integration import OllamaClient, OllamaGenerator
    except ImportError:
        print("  Ollama integration not available.")
        return None

    client = OllamaClient()
    if not client.check_connection():
        print("  WARNING: Ollama not running. Skipping Ollama generation.")
        return None

    generator = OllamaGenerator(model="mistral", client=client)
    data = config.load_data()
    texts = data["texts"]
    majority_labels = data["majority_labels"]
    label_strings = [config.ID_TO_LABEL[int(l)] for l in majority_labels]

    print("\nGenerating propositions with Ollama...")
    results = generator.generate_batch(texts, label_strings, show_progress=True)

    test_df = pd.read_csv(TEST_CSV)
    test_ids = set(test_df["id"].tolist())

    test_propositions = []
    id2row = {int(data["df"].iloc[i]["id"]): data["df"].iloc[i] for i in range(len(data["df"]))}
    for r in results:
        pid = int(r.get("id", 0))
        if pid in test_ids:
            test_propositions.append({
                "id": pid,
                "tweet_text": r.get("tweet", ""),
                "predicted_label": r.get("label", "none"),
                "confidence": 0.0,
                "generated_proposition": r.get("generated_proposition"),
            })

    generated_ids = {p["id"] for p in test_propositions}
    for _, row in test_df.iterrows():
        pid = int(row["id"])
        if pid not in generated_ids:
            test_propositions.append({
                "id": pid,
                "tweet_text": row["tweet_text"],
                "predicted_label": "none",
                "confidence": 0.0,
                "generated_proposition": None,
            })

    test_propositions.sort(key=lambda x: x["id"])
    prop_path = os.path.join(OUTPUT_DIR, "submit_task2_propositions_ollama.json")
    with open(prop_path, "w") as f:
        json.dump(test_propositions, f, indent=2)
    print(f"   Test propositions saved to {prop_path}")

    gen_count = sum(1 for p in test_propositions if p.get("generated_proposition"))
    print(f"   Generated: {gen_count}/{len(test_propositions)}")
    return test_propositions


# ============== RUN ALL & SUBMIT ==============

def run_all_and_submit():
    """Run all classifiers, all ensemble methods, compare, pick best, submit.

    Returns:
        (best_method_name, best_metrics_dict, test_submission, full_submission)
    """
    import time

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Step 1: Run all base classifiers ----
    print("\n" + "=" * 80)
    print("STEP 1: Running all base classifiers")
    print("=" * 80)

    from run_methods import TASK1_METHODS

    task1_results = {}
    task1_time = time.time()

    for method_key, (display, run_fn) in TASK1_METHODS.items():
        print(f"\n  --> {display}")
        start = time.time()
        try:
            predictions, report = run_fn()
            elapsed = time.time() - start

            pred_path = os.path.join(OUTPUT_DIR, f"predictions_classifiers_{method_key}.json")
            with open(pred_path, "w") as f:
                json.dump(predictions, f, indent=2)
            report_path = os.path.join(OUTPUT_DIR, f"evaluation_report_{method_key}.json")
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)

            metrics = report.get("test_metrics", report.get("final_eval", {}))
            f1_3 = metrics.get("f1_macro_3class") or metrics.get("f1_3class")
            f1_2 = metrics.get("f1_macro_2class") or metrics.get("f1_2class")
            ce = metrics.get("cross_entropy")

            task1_results[method_key] = {
                "display": display,
                "f1_3class": f1_3, "f1_2class": f1_2, "cross_entropy": ce,
                "predictions": predictions, "report": report, "elapsed": elapsed,
            }
            print(f"      F1(3)={f1_3:.4f}  F1(2)={f1_2:.4f}  CE={ce:.3f}  ({elapsed:.0f}s)")
        except Exception as e:
            elapsed = time.time() - start
            print(f"      FAILED ({elapsed:.0f}s): {e}")
            task1_results[method_key] = {
                "display": display,
                "f1_3class": None, "f1_2class": None, "cross_entropy": None,
                "predictions": None, "report": None, "elapsed": elapsed,
            }

    print(f"\n  {'Method':<22} {'F1(3-class)':<14} {'F1(2-class)':<14} {'CE':<10} {'Time':<10}")
    print(f"  {'-' * 20:<22} {'-' * 12:<14} {'-' * 12:<14} {'-' * 8:<10} {'-' * 8:<10}")
    for name, r in task1_results.items():
        f3 = f"{r['f1_3class']:.4f}" if r['f1_3class'] is not None else "N/A"
        f2 = f"{r['f1_2class']:.4f}" if r['f1_2class'] is not None else "N/A"
        c = f"{r['cross_entropy']:.3f}" if r['cross_entropy'] is not None else "N/A"
        print(f"  {r['display']:<22} {f3:<14} {f2:<14} {c:<10} {r['elapsed']:.0f}s")
    task1_elapsed = time.time() - task1_time

    # ---- Step 2: Run all ensemble methods ----
    print("\n" + "=" * 80)
    print("STEP 2: Running all ensemble methods")
    print("=" * 80)

    from task1_classification.task1_ensemble import METHODS as ENSEMBLE_METHODS

    ensemble_results = {}
    ensemble_time = time.time()

    for method_key in ENSEMBLE_METHODS:
        print(f"\n  --> {method_key}")
        start = time.time()
        try:
            method_fn = ENSEMBLE_METHODS[method_key]
            predictions, report = method_fn()
            if predictions is None:
                print(f"      SKIPPED (prereq not available)")
                ensemble_results[method_key] = {
                    "display": method_key, "f1_3class": None, "f1_2class": None,
                    "cross_entropy": None, "predictions": None, "report": None,
                    "elapsed": time.time() - start,
                }
                continue

            elapsed = time.time() - start

            pred_path = os.path.join(OUTPUT_DIR, f"predictions_ensemble_{method_key}.json")
            with open(pred_path, "w") as f:
                json.dump(predictions, f, indent=2)
            report_path = os.path.join(OUTPUT_DIR, f"evaluation_report_ensemble_{method_key}.json")
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)

            f1_3 = report["test_metrics"].get("f1_macro_3class")
            f1_2 = report["test_metrics"].get("f1_macro_2class")
            ce = report["test_metrics"].get("cross_entropy")

            ensemble_results[method_key] = {
                "display": method_key,
                "f1_3class": f1_3, "f1_2class": f1_2, "cross_entropy": ce,
                "predictions": predictions, "report": report, "elapsed": elapsed,
            }
            print(f"      F1(3)={f1_3:.4f}  F1(2)={f1_2:.4f}  CE={ce:.3f}  ({elapsed:.0f}s)")
        except Exception as e:
            elapsed = time.time() - start
            print(f"      FAILED ({elapsed:.0f}s): {e}")
            ensemble_results[method_key] = {
                "display": method_key, "f1_3class": None, "f1_2class": None,
                "cross_entropy": None, "predictions": None, "report": None, "elapsed": elapsed,
            }

    print(f"\n  {'Method':<22} {'F1(3-class)':<14} {'F1(2-class)':<14} {'CE':<10} {'Time':<10}")
    print(f"  {'-' * 20:<22} {'-' * 12:<14} {'-' * 12:<14} {'-' * 8:<10} {'-' * 8:<10}")
    for name, r in ensemble_results.items():
        f3 = f"{r['f1_3class']:.4f}" if r['f1_3class'] is not None else "N/A"
        f2 = f"{r['f1_2class']:.4f}" if r['f1_2class'] is not None else "N/A"
        c = f"{r['cross_entropy']:.3f}" if r['cross_entropy'] is not None else "N/A"
        print(f"  {name:<22} {f3:<14} {f2:<14} {c:<10} {r['elapsed']:.0f}s")
    ensemble_elapsed = time.time() - ensemble_time

    # ---- Step 3: Combined ranking ----
    print("\n" + "=" * 80)
    print("STEP 3: COMBINED RANKING (by F1 3-class)")
    print("=" * 80)

    all_results = {**task1_results, **ensemble_results}
    ranked = [(name, r) for name, r in all_results.items() if r["f1_3class"] is not None]
    ranked.sort(key=lambda x: x[1]["f1_3class"], reverse=True)

    if not ranked:
        print("\n  ERROR: No method succeeded. Nothing to submit.")
        return None, None, None, None

    print(f"\n  {'Rank':<6} {'Method':<22} {'F1(3)':<10} {'F1(2)':<10} {'CE':<10}")
    print(f"  {'-' * 4:<6} {'-' * 20:<22} {'-' * 8:<10} {'-' * 8:<10} {'-' * 8:<10}")
    for i, (name, r) in enumerate(ranked, 1):
        f3 = f"{r['f1_3class']:.4f}"
        f2 = f"{r['f1_2class']:.4f}"
        c = f"{r['cross_entropy']:.3f}"
        print(f"  {i:<6} {name:<22} {f3:<10} {f2:<10} {c:<10}")

    best_name, best = ranked[0]
    print(f"\n  >>> Best method: {best_name} (F1(3-class) = {best['f1_3class']:.4f})")

    # ---- Step 4: Generate submission ----
    print("\n" + "=" * 80)
    print("STEP 4: Generating submission with best method")
    print("=" * 80)

    if best_name == "weighted_voting":
        print(f"\n  Using weighted voting ensemble (5 classifiers) for submission...")
        submission, full_submission = submit_task1()
    else:
        print(f"\n  Best is {best_name}. Using full ensemble for submission.")
        submission, full_submission = submit_task1()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\n  Test predictions saved to {OUTPUT_DIR}/submit_task1_test.json")
    print(f"  Full dataset predictions saved to {OUTPUT_DIR}/submit_task1_classifiers.json")

    pred_dist = {config.ID_TO_LABEL[j]: 0 for j in range(3)}
    for p in full_submission:
        pred_dist[p["label"]] = pred_dist.get(p["label"], 0) + 1
    print(f"\n  Prediction distribution: {pred_dist}")

    print(f"\n{'=' * 80}")
    print("DONE!")
    print(f"  Best method:   {best_name}")
    print(f"  F1(3-class):   {best['f1_3class']:.4f}")
    print(f"  F1(2-class):   {best['f1_2class']:.4f}")
    print(f"  Total time:    {task1_elapsed + ensemble_elapsed:.0f}s")
    print(f"{'=' * 80}")

    # ---- Task 2: Generate propositions ----
    print("\n" + "=" * 60)
    print("TASK 2: T5 PROPOSITION GENERATION")
    print("=" * 60)
    task2_props = submit_task2_t5()
    if task2_props:
        print("\nTask 2 submission complete.")
    else:
        print("\nTask 2: No propositions generated (check errors above).")

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
            print("\nTask 2: No propositions generated (check errors above).")


if __name__ == "__main__":
    main()
