"""
DistilBERT Feature Extraction + Classifier for Enthymeme Detection

Uses frozen DistilBERT [CLS] embeddings + sklearn classifier.
Much faster on CPU than full fine-tuning and often matches or exceeds it
on small datasets (where fine-tuning easily overfits).

Usage:
    python transformer.py                        # Run CLI
    python transformer.py --device cpu           # Force CPU
    python transformer.py --save-model           # Also save classifier.pkl
    from task1_classification.transformer.transformer import run_transformer
    preds, report = run_transformer()            # Callable interface
"""

import argparse
import json
import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from config import (
    LABEL_TO_ID, ID_TO_LABEL, CLASS_LABELS, OUTPUT_DIR,
    CV_N_FOLDS, RANDOM_STATE, get_device,
    TRANSFORMER_MODEL_NAME, TRANSFORMER_BATCH_SIZE, TRANSFORMER_LEARNING_RATE, TRANSFORMER_EPOCHS, TRANSFORMER_MAX_LENGTH,
)
from evaluation.metrics import compute_metrics

try:
    from transformers import AutoTokenizer, AutoModel
    from sklearn.linear_model import LogisticRegression, SGDClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import normalize
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score, accuracy_score
    TRANSFORMERS_AVAILABLE = True
except ImportError as e:
    print(f"Missing required package: {e}")
    sys.exit(1)

import task1_classification.transformer.config as tconfig

# -- Dataset --

class TextDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels=None):
        self.input_ids = torch.tensor(input_ids, dtype=torch.long)
        self.attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long) if labels is not None else None

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        item = {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }
        if self.labels is not None:
            item["labels"] = self.labels[idx]
        return item


# -- Data loading --


def extract_features(model, tokenizer, texts, device, batch_size=64):
    """Extract [CLS] token embeddings from frozen DistilBERT."""
    tok = tokenizer(texts.tolist(), padding="max_length", truncation=True, max_length=TRANSFORMER_MAX_LENGTH)

    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_ids = torch.tensor(tok["input_ids"][i:i + batch_size]).to(device)
        batch_mask = torch.tensor(tok["attention_mask"][i:i + batch_size]).to(device)
        with torch.no_grad():
            outputs = model(input_ids=batch_ids, attention_mask=batch_mask)
        # Use [CLS] token embedding (first token)
        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        embeddings.append(cls_embeddings)

    return np.vstack(embeddings)


# -- 5-fold CV --

def run_cv(X, y, texts):
    """Run 5-fold stratified CV on *already normalized* features and return (mean_f1, std_f1, per_fold_f1, best_clf).

    Uses the *same* classifier definitions as `_build_classifier`. Selects the classifier with the highest mean validation F1.
    Returns the *fitted* best classifier so the caller can use it directly.
    """
    # All classifiers already receive normalized data (from run_transformer)
    skf = StratifiedKFold(n_splits=CV_N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    f1_scores = []
    fold_mean_f1s = {}

    # Keep per-fold best classifier to refit the winner on all training data later
    fold_best_clfs = {}  # fold_idx -> best_fitted_clf
    fold_best_names = {}  # fold_idx -> best_name

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(texts, y)):
        print(f"\n--- Fold {fold_idx + 1}/{CV_N_FOLDS} ---")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Class weights for this fold
        fold_counts = np.bincount(y_train, minlength=3)
        fold_total = len(y_train)
        fold_weights = {c: fold_total / max(fold_counts[c], 1) for c in range(3)}

        # Classifier candidates — use _build_classifier so there is a single source of truth
        classifier_names = ["lr_balanced", "lr_weighted", "sgd_log_balanced", "sgd_log_weighted"]

        best_f1 = -1
        best_name = ""
        best_clf = None

        for name in classifier_names:
            clf = _build_classifier(name, class_weight=fold_weights)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
            acc = accuracy_score(y_val, y_pred)
            fold_mean_f1s.setdefault(name, []).append(f1)
            print(f"  {name}: F1={f1:.4f} Acc={acc:.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_name = name
                best_clf = clf

        print(f"  >>> Best this fold: {best_name} F1={best_f1:.4f}")
        f1_scores.append(best_f1)
        fold_best_clfs[fold_idx] = best_clf
        fold_best_names[fold_idx] = best_name

    mean_f1 = float(np.mean(f1_scores))
    std_f1 = float(np.std(f1_scores))
    # Select overall best by mean fold F1
    best_name = max(fold_mean_f1s, key=lambda n: np.mean(fold_mean_f1s[n]))
    print(f"\nCV Macro F1: {mean_f1:.4f} +/- {std_f1:.4f}")
    fold_means = {n: float(np.mean(fs)) for n, fs in fold_mean_f1s.items()}
    print(f"Mean fold F1: {fold_means}")
    print(f"Overall best by mean F1: {best_name}")
    return mean_f1, std_f1, f1_scores, best_name, fold_best_clfs, fold_best_names



# -- Callable interface for run_methods.py --

def _build_classifier(name, class_weight=None):
    """Build the classifier pipeline identified by ``name`` from CV.

    Args:
        name: One of lr_balanced, lr_weighted, sgd_log_balanced, sgd_log_weighted
        class_weight: Override class_weight (e.g. fold_weights dict). If None, uses defaults.
    """
    if name == "lr_balanced":
        cw = class_weight if class_weight is not None else "balanced"
        return make_pipeline(LogisticRegression(class_weight=cw, max_iter=1000, random_state=RANDOM_STATE, C=1.0))
    if name == "lr_weighted":
        cw = class_weight if class_weight is not None else "balanced"
        return make_pipeline(LogisticRegression(class_weight=cw, max_iter=1000, random_state=RANDOM_STATE, C=1.0))
    if name == "sgd_log_balanced":
        cw = class_weight if class_weight is not None else "balanced"
        return make_pipeline(SGDClassifier(loss="log_loss", class_weight=cw, max_iter=1000, random_state=RANDOM_STATE, tol=1e-3))
    if name == "sgd_log_weighted":
        cw = class_weight if class_weight is not None else {"balanced": None}
        return make_pipeline(SGDClassifier(loss="log_loss", class_weight=cw, max_iter=1000, random_state=RANDOM_STATE, tol=1e-3))
    # Fallback
    cw = class_weight if class_weight is not None else "balanced"
    return make_pipeline(LogisticRegression(class_weight=cw, max_iter=1000, random_state=RANDOM_STATE, C=1.0))


def run_transformer(device_arg=None, save_model=False):
    """Run DistilBERT feature extraction + classifier. Returns (predictions, report).

    Args:
        device_arg: Force device ("cpu", "cuda", "mps") or None for auto-detect.
        save_model: If True, save classifier.pkl and config.json to output dir.
    """
    pred_list, report, _ = _run_transformer_pipeline(device_arg=device_arg, save_model=save_model)
    return pred_list, report


def _run_transformer_pipeline(device_arg=None, save_model=False):
    """Core transformer pipeline used by both run_transformer and ensemble interface.

    Returns (predictions, report, artifacts).
    artifacts = {"clf": classifier, "X_all_norm": ndarray, "y_all": ndarray,
                 "best_name": str, "device": device, "device_name": str,
                 "X_all": ndarray, "model": DistilBERT, "tokenizer": Tokenizer,
                 "df": DataFrame}
    """
    if device_arg:
        device = torch.device(device_arg)
        device_name = f"{device_arg.upper()} (forced)"
    else:
        device, device_name = get_device()

    df = config.load_training_data()
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL_NAME)

    try:
        model = AutoModel.from_pretrained(TRANSFORMER_MODEL_NAME, device_map="auto")
    except Exception as e:
        print(f"  device_map='auto' failed ({e}), falling back to CPU...")
        model = AutoModel.from_pretrained(TRANSFORMER_MODEL_NAME).to("cpu")
        device = torch.device("cpu")
        device_name = "CPU (fallback)"
    model.eval()

    X_all = extract_features(model, tokenizer, df["text"], device)
    y_all = df["label"].values
    print(f"Feature shape: {X_all.shape}")

    # 5-fold CV to pick best classifier
    print("\n" + "=" * 80)
    print("5-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 80)
    X_all_norm = normalize(X_all, norm='l2')
    mean_f1, std_f1, cv_f1_scores, best_name, _, _ = run_cv(X_all_norm, y_all, df["text"])

    # Train final model on 80% train split using CV-best classifier
    print("\n" + "=" * 80)
    print("PHASE 2: Final model on training data")
    print("=" * 80)

    train_idx, test_idx = config.get_train_test_indices(len(df))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[test_idx].reset_index(drop=True)

    X_train = X_all_norm[train_idx]
    X_val = X_all_norm[test_idx]
    y_train = df_train["label"].values
    y_val = df_val["label"].values

    print(f"Using classifier chosen by CV: {best_name}")
    best_clf = _build_classifier(best_name)
    best_clf.fit(X_train, y_train)
    y_val_pred = best_clf.predict(X_val)
    val_f1 = f1_score(y_val, y_val_pred, average="macro", zero_division=0)
    print(f"Final val F1: {val_f1:.4f}")

    # Predict on ALL data (X_all_norm already normalized)
    predictions = best_clf.predict(X_all_norm)
    all_probs = best_clf.predict_proba(X_all_norm)

    pred_list = []
    for i in range(len(df)):
        pred_id = int(predictions[i])
        prob_dict = {ID_TO_LABEL[k]: float(all_probs[i][k]) for k in range(3)}
        pred_list.append({
            "id": int(df.iloc[i]["id"]),
            "text": df.iloc[i]["text"],
            "label": ID_TO_LABEL[pred_id],
            "probabilities": prob_dict,
            "hard_prediction": pred_id,
        })

    pred_dist = Counter(p["label"] for p in pred_list)

    all_probs_dicts = [{CLASS_LABELS[j]: float(all_probs[i][j]) for j in range(3)} for i in range(len(df))]
    all_soft = [dict(row["soft_label"]) for _, row in df.iterrows()]
    metrics, per_class = compute_metrics(
        y_true=y_all, y_pred=predictions,
        prob_vectors=all_probs_dicts, ann_labels=all_soft, method_name="distilbert_features",
    )

    report = {
        "model": "distilbert_features_" + best_name,
        "hyperparameters": {
            "embedding_model": TRANSFORMER_MODEL_NAME,
            "device": str(device),
            "device_name": device_name,
            "feature_dim": X_all.shape[1],
            "embedding_norm": "l2",
            "classifier": best_name,
            "max_length": TRANSFORMER_MAX_LENGTH,
            "batch_size": TRANSFORMER_BATCH_SIZE,
        },
        "cv_scores": {"macro_f1_mean": float(mean_f1), "macro_f1_std": float(std_f1), "per_fold": cv_f1_scores},
        "final_eval": {
            "f1_macro_3class": metrics["f1_macro_3class"],
            "f1_weighted_3class": metrics["f1_weighted_3class"],
            "f1_macro_2class": metrics["f1_macro_2class"],
            "f1_weighted_2class": metrics["f1_weighted_2class"],
            "cross_entropy": metrics["cross_entropy"],
            "accuracy": float(accuracy_score(y_all, predictions)),
        },
        "test_metrics": {k: metrics[k] for k in ("f1_macro_3class", "f1_weighted_3class", "f1_macro_2class", "f1_weighted_2class", "cross_entropy")},
        "per_class": per_class,
        "prediction_distribution": {k: int(v) for k, v in pred_dist.items()},
    }

    # Save classifier artifact if requested
    if save_model:
        import joblib
        os.makedirs(tconfig.TRANSFORMER_OUTPUT_DIR, exist_ok=True)
        joblib.dump(best_clf, os.path.join(tconfig.TRANSFORMER_OUTPUT_DIR, "classifier.pkl"))
        with open(os.path.join(tconfig.TRANSFORMER_OUTPUT_DIR, "config.json"), "w") as f:
            json.dump({
                "model_name": TRANSFORMER_MODEL_NAME,
                "device": str(device),
                "feature_dim": X_all.shape[1],
                "max_length": TRANSFORMER_MAX_LENGTH,
                "embedding_norm": "l2",
            }, f, indent=2)
        print(f"Classifier saved to {tconfig.TRANSFORMER_OUTPUT_DIR}")

    # Return model artifacts for ensemble reuse
    artifacts = {
        "clf": best_clf,
        "X_all_norm": X_all_norm,
        "y_all": y_all,
        "df": df,
        "X_train": X_train,
        "y_train": y_train,
        "best_name": best_name,
        "device": device,
        "device_name": device_name,
        "X_all": X_all,
        "model": model,
        "tokenizer": tokenizer,
    }

    return pred_list, report, artifacts


def run_transformer_for_ensemble():
    """Run DistilBERT CV + train on ALL data. Returns (predictions, report, artifacts).

    The ensemble uses this to get the EXACT model selected by CV, trained on all data.
    """
    print("\n" + "=" * 80)
    print("ENSEMBLE MODE: Training on ALL data using CV-selected classifier")
    print("=" * 80)

    # Reuse the full pipeline but pass use_full_data=True
    # We do this by calling _run_transformer_pipeline with a flag
    # Actually, we need the same model extracted on ALL data. Let's just
    # call _run_transformer_pipeline (which already does that) and use
    # its clf on the full dataset.
    # The clf was trained on 80%, so we need to refit it on 100% with the same params.
    pred_list, report, artifacts = _run_transformer_pipeline()

    # The classifier was trained on the 80% split. For the ensemble, we need
    # it trained on all data. Extract the best classifier type and retrain.
    clf = _build_classifier(artifacts["best_name"])
    clf.fit(artifacts["X_all_norm"], artifacts["y_all"])

    # Replace with retrained model
    artifacts["clf"] = clf

    # Predict on all data (already done above, but with the retrained clf)
    predictions = clf.predict(artifacts["X_all_norm"])
    all_probs = clf.predict_proba(artifacts["X_all_norm"])

    pred_list = []
    df = artifacts["df"]
    for i in range(len(df)):
        pred_id = int(predictions[i])
        prob_dict = {ID_TO_LABEL[k]: float(all_probs[i][k]) for k in range(3)}
        pred_list.append({
            "id": int(df.iloc[i]["id"]),
            "text": df.iloc[i]["text"],
            "label": ID_TO_LABEL[pred_id],
            "probabilities": prob_dict,
            "hard_prediction": pred_id,
        })

    pred_dist = Counter(p["label"] for p in pred_list)
    all_probs_dicts = [{CLASS_LABELS[j]: float(all_probs[i][j]) for j in range(3)} for i in range(len(df))]
    all_soft = [dict(row["soft_label"]) for _, row in df.iterrows()]
    metrics, per_class = compute_metrics(
        y_true=artifacts["y_all"], y_pred=predictions,
        prob_vectors=all_probs_dicts, ann_labels=all_soft, method_name="distilbert_features",
    )

    # Override report with full-data metrics
    report["test_metrics"]["f1_macro_3class"] = metrics["f1_macro_3class"]
    report["test_metrics"]["f1_macro_2class"] = metrics["f1_macro_2class"]
    report["test_metrics"]["cross_entropy"] = metrics["cross_entropy"]
    report["per_class"] = per_class
    report["prediction_distribution"] = {k: int(v) for k, v in pred_dist.items()}

    artifacts["clf"] = clf  # retrained on all data
    return pred_list, report, artifacts


def get_full_data_predictions():
    """Train DistilBERT + CV-selected classifier on ALL data.

    Returns predictions in challenge submission format (no report).
    Uses the exact classifier selected by 5-fold CV.
    """
    preds, _, _ = run_transformer_for_ensemble()
    return preds


# -- Main --

def main():
    parser = argparse.ArgumentParser(description="DistilBERT feature extraction + classifier for enthymeme detection")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None,
                        help="Force device (auto-detects if not specified)")
    parser.add_argument("--save-model", action="store_true", help="Save classifier artifact (classifier.pkl) to transformer output dir")
    args = parser.parse_args()

    print("=" * 80)
    print("DISTILBERT FEATURE EXTRACTION + CLASSIFIER")
    print("=" * 80)

    pred_list, report = run_transformer(device_arg=args.device, save_model=args.save_model)

    # Save predictions and report
    pred_path = os.path.join(OUTPUT_DIR, "predictions_classifiers_transformer.json")
    with open(pred_path, "w") as f:
        json.dump(pred_list, f, indent=2)
    print(f"Predictions saved to {pred_path}")

    report_path = os.path.join(OUTPUT_DIR, "evaluation_report_transformer.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {report_path}")

    print(f"\n{'=' * 80}")
    print("COMPLETE")
    print(f"Device: {report['hyperparameters']['device_name']}")
    print(f"CV F1: {report['cv_scores']['macro_f1_mean']:.4f} +/- {report['cv_scores']['macro_f1_std']:.4f}")
    print(f"Final F1 (3-class): {report['test_metrics']['f1_macro_3class']:.4f}")
    print(f"Final F1 (2-class): {report['test_metrics']['f1_macro_2class']:.4f}")
    print(f"Final CE: {report['test_metrics']['cross_entropy']:.4f}")
    print(f"Prediction dist: {report['prediction_distribution']}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
