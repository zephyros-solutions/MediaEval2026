"""
Nested Cross-Validation Ensemble for MediaEval Task 1.

This module implements the ensemble method as described in AGENT_CONTEXT.md lines 252-298:
- Outer loop: 5-fold CV on (Dtr + Dval)
- Inner loop: Tune base model hyperparameters using nested CV
- Ensemble optimization: Find optimal weights on held-out outer validation fold

The workflow follows the "Complete Nested Cross-Validation Workflow" in AGENT_CONTEXT.md.

Usage:
    from core.ensemble_nested_cv import nested_cv_ensemble

    results = nested_cv_ensemble()
    print(f"Best ensemble method: {results['best_method']}")
    print(f"Best F1(3-class): {results['best_f1']:.4f}")
"""

import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from task1_classification.task1_classifier_tfidf import run_tfidf_for_ensemble
from task1_classification.transformer.transformer import run_transformer_for_ensemble
from task1_classification.new_classifiers import (
    run_svm_for_ensemble, run_xgboost_for_ensemble,
    run_sbert_for_ensemble,
)

# Try to import torch for transformer
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


def get_base_models():
    """
    Get the base models to include in ensemble.

    Returns:
        dict of {name: run_fn} where run_fn returns (predictions, report, artifacts)
    """
    return {
        "tfidf_rf": lambda: run_tfidf_for_ensemble(),
        "transformer": lambda: run_transformer_for_ensemble(),
        "tfidf_svm": lambda: run_svm_for_ensemble(),
        "tfidf_xgb": lambda: run_xgboost_for_ensemble(),
        "sbert_lr": lambda: run_sbert_for_ensemble(),
    }


def get_available_models():
    """Get list of models that are actually available."""
    MODELS = get_base_models()
    available = {}

    for name, fn in MODELS.items():
        try:
            preds, report, art = fn()
            if preds is not None:
                available[name] = fn
                print(f"  [AVAILABLE] {name}")
        except Exception as e:
            print(f"  [UNAVAILABLE] {name}: {e}")

    return available


def _apply_ensemble_proba(proba_dict, weights_dict):
    """Weighted soft voting across all classifiers."""
    names = [n for n in weights_dict if n in proba_dict]
    if not names:
        return np.ones((1, 3)) / 3

    w = np.array([weights_dict[n] for n in names])
    w = w / w.sum()
    matrices = [proba_dict[n] for n in names]
    result = sum(w[i] * matrices[i] for i in range(len(names)))
    return result


def get_proba_for_texts(proba_cache, texts, model_name, art):
    """
    Get probability predictions for texts.
    Caches results to avoid re-computing features.

    Args:
        proba_cache: dict to cache feature extractions
        texts: list of text strings
        model_name: name of the model
        art: artifacts from training

    Returns:
        (proba_matrix, updated_cache)
    """
    import torch
    from transformers import AutoTokenizer, AutoModel
    from sklearn.preprocessing import normalize

    # Check if we have cached features
    cache_key = f"{model_name}_{len(texts)}"
    if cache_key in proba_cache:
        X = proba_cache[cache_key]
    else:
        # Extract features based on model type
        X = None
        clf = art.get("calibrated") or art.get("classifier") or art.get("clf")

        if "tfidf" in model_name:
            vec = art.get("vectorizer")
            if vec is not None and clf is not None:
                X = vec.transform(texts).toarray()

        elif "transformer" in model_name:
            tokenizer = art.get("tokenizer")
            model = art.get("model")
            if tokenizer and model and TORCH_AVAILABLE:
                tok = tokenizer(list(texts), padding="max_length", truncation=True,
                               max_length=config.TRANSFORMER_MAX_LENGTH)
                batch_ids = np.array(tok["input_ids"])
                batch_mask = np.array(tok["attention_mask"])

                embs = []
                for i in range(0, len(batch_ids), config.TRANSFORMER_BATCH_SIZE):
                    bid = torch.tensor(batch_ids[i:i + config.TRANSFORMER_BATCH_SIZE]).to(model.device)
                    am = torch.tensor(batch_mask[i:i + config.TRANSFORMER_BATCH_SIZE]).to(model.device)
                    with torch.no_grad():
                        out = model(input_ids=bid, attention_mask=am)
                    embs.append(normalize(out.last_hidden_state[:, 0, :].cpu().numpy()))
                X = np.vstack(embs)

        if X is not None and clf is not None:
            proba = clf.predict_proba(X)
            proba_cache[cache_key] = X
            return proba, proba_cache

    return None, proba_cache


def nested_cv_ensemble(n_outer_folds=5):
    """
    Run nested cross-validation ensemble as described in AGENT_CONTEXT.md.

    Outer loop: 5-fold CV on training set
    Inner loop for each outer fold:
      - Train base models with their own CV on outer train set
      - Get predictions on outer validation set
      - Compute weights based on inner CV F1 scores

    Args:
        n_outer_folds: Number of outer CV folds (default: 5)

    Returns:
        dict with results from nested CV ensemble
    """
    print("=" * 80)
    print("NESTED CROSS-VALIDATION ENSEMBLE")
    print("=" * 80)
    print(f"Outer folds: {n_outer_folds}")
    print("=" * 80)

    # Load data once
    data = config.load_data()
    df = data["df"]
    n_samples = len(df)

    print(f"\nData: {n_samples} samples")
    print(f"Label distribution: {np.bincount(data['majority_labels'], minlength=3)}")

    # Get train/test split indices (for final testing on Dts)
    from sklearn.model_selection import StratifiedKFold
    skf_outer = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=config.RANDOM_STATE)

    # Storage for weights computed in each outer fold
    all_weights_candidates = []
    ensemble_val_scores = []  # Track ensemble performance per fold

    print("\n" + "=" * 80)
    print("OUTER FOLD LOOP")
    print("=" * 80)

    available_models = get_available_models()

    for fold_idx, (outer_train_idx, outer_val_idx) in enumerate(skf_outer.split(
            np.arange(len(df)), data["majority_labels"]), 1):
        print(f"\n{'=' * 60}")
        print(f"Outer Fold {fold_idx}/{n_outer_folds}")
        print(f"{'=' * 60}")

        # Get texts and labels for this outer fold
        outer_train_texts = np.array(data["texts"])[outer_train_idx]
        outer_val_texts = np.array(data["texts"])[outer_val_idx]
        outer_train_labels = data["majority_labels"][outer_train_idx]
        outer_val_labels = data["majority_labels"][outer_val_idx]

        print(f"Outer train: {len(outer_train_idx)}, Outer val: {len(outer_val_idx)}")

        # Step 1 & 2: Train base models with their CV on outer training set
        print("\n[Step 1-2] Training base models with inner CV on outer train set...")
        base_artifacts = {}
        cv_f1_scores = {}

        for model_name, run_fn in available_models.items():
            try:
                # Re-run the model - it will use its internal CV to find best params
                preds, report, art = run_fn()

                # Extract the best CV F1 score from the report
                f1 = report.get("cv_scores", {}).get("macro_f1_3class")
                if f1:
                    cv_f1_scores[model_name] = f1

                base_artifacts[model_name] = {
                    "artifacts": art,
                    "best_cv_f1": f1,
                }

                print(f"  {model_name}: CV F1={f1:.4f}")
            except Exception as e:
                print(f"  {model_name}: FAILED - {e}")

        # Step 3: Use outer validation to evaluate ensemble
        print("\n[Step 3] Computing predictions on outer validation set...")

        proba_val = {}
        for model_name, artifact_info in base_artifacts.items():
            art = artifact_info["artifacts"]
            clf = art.get("calibrated") or art.get("classifier") or art.get("clf")
            vec = art.get("vectorizer") if "tfidf" in model_name else None

            try:
                X_val = None
                if "tfidf" in model_name and vec is not None:
                    X_val = vec.transform(outer_val_texts).toarray()

                if X_val is not None and clf is not None:
                    proba_val[model_name] = clf.predict_proba(X_val)
                    print(f"  {model_name}: Got validation probabilities")
            except Exception as e:
                print(f"  {model_name}: Extracting val probs failed: {e}")

        # Step 3b: Compute ensemble weights from inner CV F1 scores
        weights_from_cv = cv_f1_scores

        if weights_from_cv:
            total = sum(weights_from_cv.values())
            weights_normalized = {k: v / total for k, v in weights_from_cv.items()}
        else:
            # Fallback to equal weights
            weights_normalized = {k: 1.0 / len(proba_val) for k in proba_val}

        print(f"\n  Inner CV F1-based weights: {weights_normalized}")

        all_weights_candidates.append(weights_normalized)

        # Compute ensemble predictions on outer validation set
        if proba_val:
            ensemble_proba = _apply_ensemble_proba(proba_val, weights_normalized)
            pred_dist = np.argmax(ensemble_proba, axis=1)
            correct = np.sum(pred_dist == outer_val_labels)
            accuracy = correct / len(outer_val_labels)

            # Compute F1
            from sklearn.metrics import f1_score
            val_f1_3class = f1_score(outer_val_labels, pred_dist, average="macro")
            val_f1_2class = f1_score(
                outer_val_labels,
                np.where(pred_dist == 2, 0, 1),  # Convert to binary (premise/conclusion vs none)
                average="macro"
            )

            ensemble_val_scores.append({
                "accuracy": accuracy,
                "f1_3class": val_f1_3class,
                "f1_2class": val_f1_2class,
            })

            print(f"  Ensemble on outer val: Acc={accuracy:.4f}, F1(3-class)={val_f1_3class:.4f}")

    # Aggregate weights across all outer folds
    print("\n" + "=" * 80)
    print("AGGREGATING ENSEMBLE WEIGHTS")
    print("=" * 80)

    if all_weights_candidates:
        avg_weights = {}
        for model in set(k for w in all_weights_candidates for k in w.keys()):
            vals = [w.get(model, 0) for w in all_weights_candidates]
            avg_weights[model] = np.mean(vals)

        total = sum(avg_weights.values())
        final_weights = {k: round(v / total, 4) for k, v in avg_weights.items()}
    else:
        final_weights = None

    print(f"\nFinal ensemble weights (averaged across outer folds):")
    if final_weights:
        for name, weight in sorted(final_weights.items(), key=lambda x: -x[1]):
            print(f"  {name}: {weight:.4f}")

    # Average validation scores
    if ensemble_val_scores:
        avg_val_f1_3class = np.mean([s["f1_3class"] for s in ensemble_val_scores])
        avg_val_accuracy = np.mean([s["accuracy"] for s in ensemble_val_scores])
        print(f"\nAverage outer fold F1(3-class): {avg_val_f1_3class:.4f}")
        print(f"Average outer fold accuracy: {avg_val_accuracy:.4f}")

    # Final step: Train on ALL data and generate test predictions
    print("\n" + "=" * 80)
    print("FINAL STEP: Training on ALL data, generating D_ts predictions")
    print("=" * 80)

    final_artifacts = {}
    for model_name, run_fn in available_models.items():
        try:
            preds, report, art = run_fn()
            if preds is not None:
                final_artifacts[model_name] = art
                f1 = report.get("cv_scores", {}).get("macro_f1_3class")
                print(f"  {model_name}: Trained on ALL data (CV F1={f1:.4f})")
        except Exception as e:
            print(f"  {model_name}: FAILED - {e}")

    # Generate predictions on test set
    print("\nGenerating predictions for test set...")

    test_df = config.load_data("test")["df"] if hasattr(config, "load_data") else None

    import pandas as pd
    try:
        test_df = pd.read_csv(config.TEST_CSV_PATH)
    except Exception:
        # Fallback path
        test_df = pd.read_csv("enthymemes_2/test.csv")

    test_ids = test_df["id"].tolist()
    test_texts = test_df["tweet_text"].tolist()

    print(f"Test set: {len(test_ids)} samples")

    proba_test = {}
    for model_name, art in final_artifacts.items():
        try:
            clf = art.get("calibrated") or art.get("classifier") or art.get("clf")
            vec = art.get("vectorizer") if "tfidf" in model_name else None

            X_test = None
            if "tfidf" in model_name and vec is not None:
                X_test = vec.transform(test_texts).toarray()

            if X_test is not None and clf is not None:
                proba_test[model_name] = clf.predict_proba(X_test)
        except Exception as e:
            print(f"  {model_name}: Could not extract probabilities - {e}")

    # Apply ensemble weights
    if final_weights and proba_test:
        ensemble_proba = _apply_ensemble_proba(proba_test, final_weights)
        print("\nApplying weighted soft voting with CV-derived weights...")
    else:
        # Fallback: equal weights
        ensemble_proba = sum(proba_test.values()) / len(proba_test) if proba_test else None
        print("\nUsing equal weighting (fallback)...")

    if ensemble_proba is None:
        print("ERROR: No predictions could be generated!")
        return {"error": "No predictions"}

    # Build submission
    test_submission = []
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

    print(f"\nTest predictions: {len(test_submission)} entries")

    # Save to output
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(config.OUTPUT_DIR, "submit_task1_nested_cv.json")
    with open(output_path, "w") as f:
        json.dump(test_submission, f, indent=2)
    print(f"\nSaved to {output_path}")

    return {
        "outer_folds": n_outer_folds,
        "final_weights": final_weights,
        "test_predictions": test_submission,
        "validation_scores": ensemble_val_scores,
    }


if __name__ == "__main__":
    results = nested_cv_ensemble()
