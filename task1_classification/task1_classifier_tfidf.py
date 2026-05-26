"""
Task 1: TF-IDF + Random Forest Classification with Soft Labels

Approach:
  1. Load annotations from shared config
  2. 5-fold stratified CV for model selection
  3. Final model on full train+val with soft-label sample weights
  4. Probability vectors for challenge submission

Output:
  outputs/predictions_classifiers.json   (challenge format with probabilities)
  outputs/evaluation_report.json        (CV + test metrics + cross-entropy)

Callable interface:
  from task1_classification.task1_classifier_tfidf import run_tfidf
  predictions, report = run_tfidf()
"""

import sys
import os
import pickle
import numpy as np
import pandas as pd
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from evaluation.metrics import compute_metrics
import config

warnings.filterwarnings("ignore")


def _make_soft_weights(df_subset, majority_labels_full, ann_labels_list):
    """Return scalar sample weights from soft label probability at true class."""
    indices = df_subset.index
    weights = np.zeros(len(indices))
    for i, idx in enumerate(indices):
        soft = ann_labels_list[idx]
        true_class = majority_labels_full[idx]
        weights[i] = soft.get(true_class, 1.0 / 3)
    return weights


def _run_pipeline():
    """Run the full TF-IDF pipeline. Returns (predictions, report)."""
    data = config.load_data()
    df = data["df"]
    ann_labels = data["ann_labels"]
    majority_labels = data["majority_labels"]

    print(f"Dataset size: {len(df)}")
    print(f"Label distribution:\n{pd.Series(majority_labels).value_counts().sort_index().to_string(index=False)}\n")

    test_df = df.sample(frac=1 - config.TRAIN_VAL_SPLIT, random_state=config.RANDOM_STATE)
    train_df = df.drop(test_df.index)
    print(f"Data splits: train={len(train_df)}, test={len(test_df)}\n")

    vectorizer = TfidfVectorizer(**config.TFIDF_DEFAULTS)
    X_train = vectorizer.fit_transform(train_df["tweet_text"].values)
    X_test = vectorizer.transform(test_df["tweet_text"].values)
    y_train = np.array([config.LABEL_TO_ID[l] for l in train_df["majority_label"]])
    y_test = np.array([config.LABEL_TO_ID[l] for l in test_df["majority_label"]])
    print(f"Features: {X_train.shape}\n")

    sw = _make_soft_weights(train_df, majority_labels, ann_labels)

    print("Phase 1: 5-fold CV for hyperparameter tuning...")
    param_grid = config.RF_DEFAULTS
    cv = StratifiedKFold(n_splits=config.CV_N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    grid_search = GridSearchCV(
        RandomForestClassifier(n_jobs=-1), param_grid, cv=cv, scoring="f1_macro", n_jobs=-1,
    ).fit(X_train, y_train, sample_weight=sw)
    print(f"  Best params: {grid_search.best_params_}")
    print(f"  Best CV macro F1: {grid_search.best_score_:.4f}")

    # Phase 2: Train on full training set
    X_train_full = vectorizer.transform(train_df["tweet_text"].values)
    y_train_full = np.array([config.LABEL_TO_ID[l] for l in train_df["majority_label"]])
    print(f"\nPhase 2: Training final model on {len(train_df)} samples...")
    final_model = RandomForestClassifier(**grid_search.best_params_, n_jobs=-1)
    final_model.fit(X_train_full, y_train_full, sample_weight=_make_soft_weights(train_df, majority_labels, ann_labels))

    # Evaluate
    y_pred = final_model.predict(X_test)
    y_proba = final_model.predict_proba(X_test)
    prob_dicts = [{config.CLASS_LABELS[j]: float(y_proba[i][j]) for j in range(3)} for i in range(len(y_test))]
    # Get soft labels for test set
    test_soft = []
    for row_idx in test_df.index:
        test_soft.append(ann_labels[row_idx])
    metrics, per_class = compute_metrics(
        y_true=y_test, y_pred=y_pred,
        prob_vectors=prob_dicts, ann_labels=test_soft, method_name="tfidf",
    )

    # Save model
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../outputs/fine_tuned_model/task1_model_tfidf.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"model": final_model, "vectorizer": vectorizer}, f)

    # Build predictions
    predictions = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        pred_id = int(y_pred[i])
        predictions.append({
            "id": int(row["id"]),
            "text": row["tweet_text"],
            "label": config.ID_TO_LABEL[pred_id],
            "probabilities": {label: float(y_proba[i][j]) for j, label in enumerate(config.CLASS_LABELS)},
            "hard_prediction": pred_id,
        })

    report = {
        "model": "tfidf_random_forest",
        "hyperparameters": grid_search.best_params_,
        "cv_scores": {"macro_f1_3class": float(grid_search.best_score_), "per_fold": [float(s) for s in grid_search.cv_results_["mean_test_score"]]},
        "test_metrics": {k: float(v) for k, v in metrics.items() if k not in ("per_class", "method")},
        "per_class": per_class,
        "prediction_distribution": {config.CLASS_LABELS[j]: int(np.sum(y_pred == j)) for j in range(3)},
    }

    print(f"\nF1(3-class): {metrics['f1_macro_3class']:.4f}  F1(2-class): {metrics['f1_macro_2class']:.4f}  CE: {metrics['cross_entropy']:.4f}")
    return predictions, report


def run_tfidf():
    """Run TF-IDF classifier. Returns (predictions, report)."""
    return _run_pipeline()


if __name__ == "__main__":
    run_tfidf()
