"""
Pipeline orchestrators for MediaEval Task 1 classification.

This module provides thin wrappers around core/evaluator.py to support:
- Standalone mode: CV on 80% training set, evaluate on 20% test set
- Ensemble mode: Train on ALL data for submission

Usage:
    from core.pipeline import run_standalone, run_ensemble

    # Run all registered models in standalone mode
    results = run_standalone()

    # Train all models for ensemble submission
    artifacts = run_ensemble()
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import numpy as np
from core.evaluator import run_model_config


def get_classifier_models():
    """
    Define all classifier models using the new unified architecture.

    Returns:
        {name: {"feature_extractor": fn, "classifier_fn": fn, "param_grid": dict}}

    To add a new model:
        1. Create feature extractor function (takes texts, returns features)
        2. Create classifier factory (returns untrained sklearn classifier)
        3. Add to this dict
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import LogisticRegression, SGDClassifier, RidgeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import ComplementNB
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import RandomForestClassifier
    try:
        import xgboost as xgb
        XGB_AVAILABLE = True
    except ImportError:
        XGB_AVAILABLE = False

    # Feature extractors
    def extract_tfidf(train_texts, test_texts):
        """Extract TF-IDF features."""
        vec = TfidfVectorizer(**config.TFIDF_DEFAULTS)
        if train_texts is not None and test_texts is not None:
            X_train = vec.fit_transform(train_texts)
            X_test = vec.transform(test_texts)
            return X_train, X_test
        elif train_texts is not None:
            X_all = vec.fit_transform(train_texts)
            return X_all, None
        return None, None

    def extract_tfidf_trained(vec, texts):
        """Use pre-fitted vectorizer for new texts (for test set predictions)."""
        return vec.transform(texts)

    # Classifier factories
    def make_lr():
        return LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, random_state=config.RANDOM_STATE)

    def make_svm():
        return LinearSVC(dual="auto", max_iter=5000, random_state=config.RANDOM_STATE)

    def make_rf():
        return RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=config.RANDOM_STATE)

    # XGBoost disabled due to macOS ARM64 segfault issues (BLAS library conflicts)
    # Uncomment this block on Linux systems where XGBoost works correctly:
    """
    if XGB_AVAILABLE:
        def make_xgb():
            return xgb.XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                tree_method="hist", eval_metric="mlogloss", random_state=config.RANDOM_STATE
            )
    else:
        make_xgb = None
    """
    make_xgb = None  # Disabled for macOS compatibility

    def make_sgd():
        return SGDClassifier(loss="log_loss", class_weight="balanced", max_iter=1000, random_state=config.RANDOM_STATE, tol=1e-3)

    def make_ridge():
        return RidgeClassifier(class_weight="balanced", random_state=config.RANDOM_STATE)

    def make_knn():
        return KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

    def make_complement_nb():
        # ComplementNB doesn't use the TF-IDF vectorizer directly in param_grid
        return ComplementNB()

    # Model definitions (name -> config)
    MODELS = {
        "tfidf_lr": {
            "feature_extractor": extract_tfidf,
            "classifier_fn": make_lr,
            "param_grid": {"C": [0.1, 1.0, 10.0]},
            "train_full_vectorizer": True,  # Fit vectorizer on all data for ensemble
        },
        "tfidf_svm": {
            "feature_extractor": extract_tfidf,
            "classifier_fn": make_svm,
            "param_grid": {"C": [0.01, 0.1, 1.0, 10.0]},
            "train_full_vectorizer": True,
        },
        "tfidf_ridge": {
            "feature_extractor": extract_tfidf,
            "classifier_fn": make_ridge,
            "param_grid": {"alpha": [0.1, 1.0, 10.0]},
            "train_full_vectorizer": True,
        },
        "tfidf_knn": {
            "feature_extractor": extract_tfidf,
            "classifier_fn": make_knn,
            "param_grid": {"n_neighbors": [3, 5, 7]},
            "train_full_vectorizer": True,
        },
        "tfidf_complement_nb": {
            "feature_extractor": extract_tfidf,
            "classifier_fn": make_complement_nb,
            "param_grid": None,  # ComplementNB has few hyperparameters
            "train_full_vectorizer": True,
        },
        "tfidf_rf": {
            "feature_extractor": extract_tfidf,
            "classifier_fn": make_rf,
            "param_grid": {"n_estimators": [50, 100], "max_depth": [None, 10]},
            "train_full_vectorizer": True,
        },
    }

    if XGB_AVAILABLE:
        MODELS["tfidf_xgb"] = {
            "feature_extractor": extract_tfidf,
            "classifier_fn": make_xgb,
            "param_grid": None,  # Fixed params to avoid segfault
            "train_full_vectorizer": True,
        }

    return MODELS


def get_transformer_model():
    """
    Define the transformer model (DistilBERT features + classifier).

    Returns:
        {"name": str, "feature_extractor": fn, "classifier_fn": fn}
    """
    import torch
    from transformers import AutoTokenizer, AutoModel
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import normalize

    device, _ = config.get_device()

    def extract_transformer_features(texts, _):
        """Extract DistilBERT [CLS] embeddings."""
        tokenizer = AutoTokenizer.from_pretrained(config.TRANSFORMER_MODEL_NAME)
        model = AutoModel.from_pretrained(config.TRANSFORMER_MODEL_NAME, device_map="auto")
        model.eval()

        all_embeddings = []
        for i in range(0, len(texts), config.TRANSFORMER_BATCH_SIZE):
            batch_texts = texts[i:i + config.TRANSFORMER_BATCH_SIZE]
            tok = tokenizer(batch_texts, padding="max_length", truncation=True, max_length=config.TRANSFORMER_MAX_LENGTH)
            batch_ids = torch.tensor(tok["input_ids"]).to(device)
            batch_mask = torch.tensor(tok["attention_mask"]).to(device)
            with torch.no_grad():
                outputs = model(input_ids=batch_ids, attention_mask=batch_mask)
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(normalize(cls_emb, norm='l2'))

        return np.vstack(all_embeddings)

    def make_transformer_clf():
        return LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, random_state=config.RANDOM_STATE)

    return {
        "transformer": {
            "feature_extractor": extract_transformer_features,
            "classifier_fn": make_transformer_clf,
            "param_grid": {"C": [0.1, 1.0, 10.0]},
            "requires_gpu": True,
        }
    }


def run_standalone():
    """
    Run all models in standalone mode (CV on 80% training set).

    Returns:
        {name: (predictions, report)} for each model
    """
    from core.evaluator import run_standalone_models

    # Get all classifier models
    MODELS = get_classifier_models()

    # Add transformer if GPU available
    device, _ = config.get_device()
    if "cuda" in str(device) or "mps" in str(device):
        MODELS.update(get_transformer_model())

    results = run_standalone_models(MODELS)
    return results


def run_ensemble():
    """
    Train all models on ALL 1333 instances for ensemble submission.

    Returns:
        {name: (predictions, report, artifacts)} for each model
    """
    from core.evaluator import run_ensemble_models

    MODELS = get_classifier_models()
    MODELS.update(get_transformer_model())

    results = run_ensemble_models(MODELS)
    return results


if __name__ == "__main__":
    print("=" * 80)
    print("RUNNING ALL MODELS IN STANDALONE MODE")
    print("=" * 80)
    results = run_standalone()

    print("\n" + "=" * 80)
    print("STANDALONE RESULTS")
    print("=" * 80)
    for name, (preds, report) in results.items():
        if preds is not None:
            f1_3 = report.get("test_metrics", {}).get("f1_macro_3class", "N/A")
            print(f"{name}: F1(3-class)={f1_3}")
