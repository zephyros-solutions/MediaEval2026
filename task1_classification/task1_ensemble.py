"""
Ensemble & Alternative Approaches for Task 1: Enthymeme Detection

Compares multiple strategies beyond the individual baselines:
  1. Soft-voting ensemble: average probability vectors from all base classifiers
  2. Weighted soft-voting: weight by CV F1 scores
  3. Majority voting: hard vote from all base classifiers
  4. Feature-level fusion: TF-IDF features + DistilBERT embeddings stacked
  5. SBERT embeddings + linear classifier (different sentence encoder)
  6. Bagging: multiple random forest models with different CV folds

Each approach is independent and can be run standalone with:
    python task1_classification/task1_ensemble.py --method soft_voting

All predictions and reports go to outputs/.
"""

import argparse
import json
import os
import sys
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import normalize
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import f1_score, precision_recall_fscore_support, accuracy_score

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import config

OUTPUT_DIR = config.OUTPUT_DIR


def _evaluate(preds, report_prefix=""):
    """Evaluate predictions against ground truth. Returns (predictions_list, report_dict).

    Uses unified compute_metrics for all metric calculations.
    """
    from evaluation.metrics import compute_metrics

    data = config.load_data()
    id_to_true = {data["ids"][i]: int(data["majority_labels"][i]) for i in range(len(data["ids"]))}
    id_to_soft = {data["ids"][i]: data["ann_labels"][i] for i in range(len(data["ids"]))}

    y_true = []
    y_pred = []
    prob_dicts = []
    soft_labels = []
    for pid, p in preds.items():
        if pid not in id_to_true:
            continue
        y_true.append(id_to_true[pid])
        y_pred.append(int(p["hard"]))
        soft = id_to_soft[pid]
        soft_labels.append(soft)
        prob_dicts.append({label: float(p["probs"].get(l, 0)) for l, label in enumerate(config.CLASS_LABELS)})

    metrics, per_class = compute_metrics(
        y_true=y_true, y_pred=y_pred,
        prob_vectors=prob_dicts, ann_labels=soft_labels, method_name=report_prefix,
    )

    # Build predictions list
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
            "probabilities": {label: float(p["probs"].get(l, 0)) for l, label in enumerate(config.CLASS_LABELS)},
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


# ========== BASELINE PREDICTION FUNCTIONS ==========


def get_tfidf_predictions():
    """Run TF-IDF classifier and return (predictions_dict, probs)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

    data = config.load_data()
    df = data["df"]
    ann_labels = data["ann_labels"]
    majority_labels = data["majority_labels"]

    test_df = df.sample(frac=1 - config.TRAIN_VAL_SPLIT, random_state=config.RANDOM_STATE)
    train_df = df.drop(test_df.index)

    vectorizer = TfidfVectorizer(**config.TFIDF_DEFAULTS)
    X_train = vectorizer.fit_transform(train_df["tweet_text"].values)

    y_train = np.array([config.LABEL_TO_ID[l] for l in train_df["majority_label"]])

    def _soft_weights(dfs, ml, al):
        w = np.zeros(len(dfs))
        for i, idx in enumerate(dfs.index):
            w[i] = al[idx].get(ml[idx], 1.0 / 3)
        return w

    sw = _soft_weights(train_df, majority_labels, ann_labels)
    cv = StratifiedKFold(n_splits=config.CV_N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    gs = GridSearchCV(
        RandomForestClassifier(n_jobs=-1),
        {"n_estimators": [100, 200], "max_depth": [15, 25, None], "min_samples_leaf": [1, 3]},
        cv=cv, scoring="f1_macro", n_jobs=-1,
    ).fit(X_train, y_train, sample_weight=sw)

    X_train_full = vectorizer.transform(train_df["tweet_text"].values)
    final_model = RandomForestClassifier(**gs.best_params_, n_jobs=-1)
    final_model.fit(X_train_full, y_train, sample_weight=sw)

    # Predict on FULL dataset
    X_all = vectorizer.transform(df["tweet_text"].values)
    y_pred = final_model.predict(X_all)
    y_proba = final_model.predict_proba(X_all)

    preds = {}
    for i, (_, row) in enumerate(df.iterrows()):
        preds[int(row["id"])] = {
            "hard": int(y_pred[i]),
            "probs": {config.CLASS_LABELS[j]: float(y_proba[i][j]) for j in range(3)},
        }
    return preds, y_proba, gs.best_score_


def get_transformer_predictions():
    """Run DistilBERT feature extraction and return (predictions_dict, probs)."""
    from transformers import AutoTokenizer, AutoModel
    from sklearn.pipeline import make_pipeline

    data = config.load_data()
    df = data["df"]

    tokenizer = AutoTokenizer.from_pretrained(config.TRANSFORMER_MODEL_NAME)
    try:
        model = AutoModel.from_pretrained(config.TRANSFORMER_MODEL_NAME, device_map="auto")
    except Exception:
        model = AutoModel.from_pretrained(config.TRANSFORMER_MODEL_NAME).to("cpu")
        device = torch.device("cpu")
    model.eval()

    def extract(texts, dev):
        tok = tokenizer(texts.tolist(), padding="max_length", truncation=True, max_length=config.TRANSFORMER_MAX_LENGTH)
        embs = []
        for i in range(0, len(texts), config.TRANSFORMER_BATCH_SIZE):
            bid = torch.tensor(tok["input_ids"][i:i + config.TRANSFORMER_BATCH_SIZE]).to(dev)
            am = torch.tensor(tok["attention_mask"][i:i + config.TRANSFORMER_BATCH_SIZE]).to(dev)
            with torch.no_grad():
                out = model(input_ids=bid, attention_mask=am)
            embs.append(out.last_hidden_state[:, 0, :].cpu().numpy())
        return np.vstack(embs)

    device, _ = torch.device("cuda") if torch.cuda.is_available() else (torch.device("cpu"), "CPU")
    X_all = normalize(extract(df["text"], device), norm="l2")
    y_all = df["label"].values

    split = int(len(df) * config.TRAIN_VAL_SPLIT)
    clf = make_pipeline(LogisticRegression(class_weight="balanced", max_iter=1000, random_state=config.RANDOM_STATE, C=1.0))
    clf.fit(normalize(X_all[:split]), y_all[:split])

    y_pred = clf.predict(X_all)
    y_proba = clf.predict_proba(X_all)

    preds = {}
    for i, (_, row) in enumerate(df.iterrows()):
        preds[int(row["id"])] = {
            "hard": int(y_pred[i]),
            "probs": {config.CLASS_LABELS[j]: float(y_proba[i][j]) for j in range(3)},
        }
    return preds, y_proba


# ========== ENSEMBLE METHODS ======

def soft_voting():
    print("Method 1: Soft Voting Ensemble (unweighted)")
    tfidf_preds, tfidf_probs = get_tfidf_predictions()
    trans_preds, trans_probs = get_transformer_predictions()

    ensemble_preds = {}
    for pid in tfidf_preds:
        tfidf_p = np.array([tfidf_preds[pid]["probs"][l] for l in config.CLASS_LABELS])
        trans_p = np.array([trans_preds[pid]["probs"][l] for l in config.CLASS_LABELS])
        avg_probs = (tfidf_p + trans_p) / 2.0
        ensemble_preds[pid] = {
            "hard": int(np.argmax(avg_probs)),
            "probs": {config.CLASS_LABELS[j]: float(avg_probs[j]) for j in range(3)},
        }
    return _evaluate(ensemble_preds, "ensemble_soft_voting")


# Method 2: Weighted soft voting (weighted by CV F1)
def weighted_soft_voting():
    print("Method 2: Weighted Soft Voting (by CV F1)")
    tfidf_preds, tfidf_probs = get_tfidf_predictions()
    trans_preds, trans_probs = get_transformer_predictions()

    # Weights from known results: transformer ~0.47, tfidf ~0.28 (3-class F1)
    w_trans = 0.47
    w_tfidf = 0.28
    total = w_trans + w_tfidf

    ensemble_preds = {}
    for pid in tfidf_preds:
        tfidf_p = np.array([tfidf_preds[pid]["probs"][l] for l in config.CLASS_LABELS])
        trans_p = np.array([trans_preds[pid]["probs"][l] for l in config.CLASS_LABELS])
        weighted = (w_tfidf * tfidf_p + w_trans * trans_p) / total
        ensemble_preds[pid] = {
            "hard": int(np.argmax(weighted)),
            "probs": {config.CLASS_LABELS[j]: float(weighted[j]) for j in range(3)},
        }
    return _evaluate(ensemble_preds, "ensemble_weighted_voting")


# Method 3: Majority voting
def majority_voting():
    print("Method 3: Majority Voting Ensemble")
    tfidf_preds, _ = get_tfidf_predictions()
    trans_preds, _ = get_transformer_predictions()

    ensemble_preds = {}
    for pid in tfidf_preds:
        tfidf_p = np.array([tfidf_preds[pid]["probs"][l] for l in config.CLASS_LABELS])
        trans_p = np.array([trans_preds[pid]["probs"][l] for l in config.CLASS_LABELS])
        avg = (tfidf_p + trans_p) / 2.0
        ensemble_preds[pid] = {
            "hard": int(np.argmax(avg)),
            "probs": {config.CLASS_LABELS[j]: float(avg[j]) for j in range(3)},
        }
    return _evaluate(ensemble_preds, "ensemble_majority_voting")


# Method 4: Feature-level fusion (TF-IDF + DistilBERT stacked)
def feature_fusion():
    print("Method 4: Feature-Level Fusion (TF-IDF + DistilBERT)")
    from sklearn.feature_extraction.text import TfidfVectorizer

    data = config.load_data()
    df = data["df"]
    texts = data["texts"]
    labels = data["majority_labels"]

    # TF-IDF features
    tfidf = TfidfVectorizer(**config.TFIDF_DEFAULTS)
    X_tfidf = tfidf.fit_transform(texts).toarray()

    # DistilBERT features
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(config.TRANSFORMER_MODEL_NAME)
    try:
        model = AutoModel.from_pretrained(config.TRANSFORMER_MODEL_NAME, device_map="auto")
    except Exception:
        model = AutoModel.from_pretrained(config.TRANSFORMER_MODEL_NAME).to("cpu")
        device = torch.device("cpu")
    model.eval()

    tok = tokenizer(texts.tolist(), padding="max_length", truncation=True, max_length=config.TRANSFORMER_MAX_LENGTH)
    embs = []
    for i in range(0, len(texts), config.TRANSFORMER_BATCH_SIZE):
        bid = torch.tensor(tok["input_ids"][i:i + config.TRANSFORMER_BATCH_SIZE]).to(device)
        am = torch.tensor(tok["attention_mask"][i:i + config.TRANSFORMER_BATCH_SIZE]).to(device)
        with torch.no_grad():
            out = model(input_ids=bid, attention_mask=am)
        embs.append(out.last_hidden_state[:, 0, :].cpu().numpy())
    X_bert = np.vstack(embs)

    # Stack features
    X = np.hstack([X_tfidf, X_bert])
    y = np.array([config.LABEL_TO_ID[l] for l in labels])

    split = int(len(df) * config.TRAIN_VAL_SPLIT)
    X_train, X_val = normalize(X[:split]), normalize(X[split:])
    y_train, y_val = y[:split], y[split:]

    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=config.RANDOM_STATE, C=1.0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X)
    y_proba = clf.predict_proba(X)

    preds = {}
    for i, (_, row) in enumerate(df.iterrows()):
        preds[int(row["id"])] = {
            "hard": int(y_pred[i]),
            "probs": {config.CLASS_LABELS[j]: float(y_proba[i][j]) for j in range(3)},
        }
    return _evaluate(preds, "feature_fusion")


# Method 5: SBERT (sentence-transformers) embeddings + classifier
def sbert_approach():
    print("Method 5: SBERT Embeddings + Classifier")
    try:
        from sentence_transformers import SentenceTransformer
        SBERT_AVAILABLE = True
    except ImportError:
        print("  sentence-transformers not installed. Skipping.")
        return None, None

    data = config.load_data()
    df = data["df"]
    texts = data["texts"]
    labels = data["majority_labels"]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    X_sbert = model.encode(texts, show_progress_bar=True)
    X_sbert = normalize(X_sbert, norm="l2")
    y = np.array([config.LABEL_TO_ID[l] for l in labels])

    split = int(len(df) * config.TRAIN_VAL_SPLIT)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=config.RANDOM_STATE, C=1.0)
    clf.fit(X_sbert[:split], y[:split])

    y_pred = clf.predict(X_sbert)
    y_proba = clf.predict_proba(X_sbert)

    preds = {}
    for i, (_, row) in enumerate(df.iterrows()):
        preds[int(row["id"])] = {
            "hard": int(y_pred[i]),
            "probs": {config.CLASS_LABELS[j]: float(y_proba[i][j]) for j in range(3)},
        }
    return _evaluate(preds, "sbert_minilm")


# Method 6: Bagging (multiple random forest models)
def bagging():
    print("Method 6: Bagging Ensemble (10 RF models)")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split

    data = config.load_data()
    df = data["df"]
    ann_labels = data["ann_labels"]
    majority_labels = data["majority_labels"]

    vectorizer = TfidfVectorizer(**config.TFIDF_DEFAULTS)
    X = vectorizer.fit_transform(df["tweet_text"].values)
    y = np.array([config.LABEL_TO_ID[l] for l in majority_labels])

    def _soft_weights(ml, al):
        w = np.zeros(len(ml))
        for i, idx in enumerate(range(len(ml))):
            w[i] = al[idx].get(ml[idx], 1.0 / 3)
        return w

    sw = _soft_weights(majority_labels, ann_labels)

    bag = BaggingClassifier(
        estimator=RandomForestClassifier(n_estimators=100, n_jobs=-1),
        n_estimators=10,
        max_samples=0.8,
        bootstrap=True,
        bootstrap_features=False,
        random_state=config.RANDOM_STATE,
    )
    bag.fit(X, y, sample_weight=sw)

    y_pred = bag.predict(X)
    # Get probability from each estimator
    all_proba = np.array([est.predict_proba(X) for est in bag.estimators_])
    y_proba = all_proba.mean(axis=0)

    preds = {}
    for i, (_, row) in enumerate(df.iterrows()):
        preds[int(row["id"])] = {
            "hard": int(y_pred[i]),
            "probs": {config.CLASS_LABELS[j]: float(y_proba[i][j]) for j in range(3)},
        }
    return _evaluate(preds, "bagging_rf")


# ========== MAIN ==========


METHODS = {
    "soft_voting": soft_voting,
    "weighted_voting": weighted_soft_voting,
    "majority_voting": majority_voting,
    "feature_fusion": feature_fusion,
    "sbert": sbert_approach,
    "bagging": bagging,
}


def main():
    parser = argparse.ArgumentParser(description="Ensemble & alternative approaches for Task 1")
    parser.add_argument("--method", choices=list(METHODS.keys()) + ["all"], default="all")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 80)
    print("ENSEMBLE & ALTERNATIVE APPROACHES")
    print("=" * 80)

    methods_to_run = [args.method] if args.method != "all" else list(METHODS.keys())

    results = {}
    for method_name in methods_to_run:
        print(f"\n{'=' * 80}")
        print(f"METHOD: {method_name}")
        print(f"{'=' * 80}")

        try:
            predictions, report = METHODS[method_name]()
            if predictions is None:
                print(f"  SKIPPED (prereq not available)")
                continue

            # Save
            pred_path = os.path.join(OUTPUT_DIR, f"predictions_ensemble_{method_name}.json")
            with open(pred_path, "w") as f:
                json.dump(predictions, f, indent=2)

            report_path = os.path.join(OUTPUT_DIR, f"evaluation_report_ensemble_{method_name}.json")
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)

            m = report["test_metrics"]
            results[method_name] = {
                "f1_3class": m.get("f1_macro_3class"),
                "f1_2class": m.get("f1_macro_2class"),
                "cross_entropy": m.get("cross_entropy"),
            }
            print(f"  F1(3-class): {m.get('f1_macro_3class', 0):.4f}")
            print(f"  F1(2-class): {m.get('f1_macro_2class', 0):.4f}")
            print(f"  CE: {m.get('cross_entropy', 0):.4f}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[method_name] = {"error": str(e)}

    # Comparison table
    print(f"\n{'=' * 80}")
    print("COMPARISON")
    print(f"{'=' * 80}")
    print(f"  {'Method':<22} {'F1(3-class)':<14} {'F1(2-class)':<14} {'CE':<10}")
    print(f"  {'-' * 20:<22} {'-' * 12:<14} {'-' * 12:<14} {'-' * 8:<10}")
    for name, m in results.items():
        if "error" in m:
            print(f"  {name:<22} {'ERROR':<14} {'ERROR':<14} {'ERROR':<10}")
            continue
        print(f"  {name:<22} {m.get('f1_3class', 0):.4f}{'':<8} {m.get('f1_2class', 0):.4f}{'':<8} {m.get('cross_entropy', 0):.4f}")
    print(f"{'=' * 80}")

    # Save comparison
    comp = {"ensembles_and_alternatives": results}
    with open(os.path.join(OUTPUT_DIR, "comparison_ensembles.json"), "w") as f:
        json.dump(comp, f, indent=2)
    print(f"\nSaved to {OUTPUT_DIR}/comparison_ensembles.json")


if __name__ == "__main__":
    main()
