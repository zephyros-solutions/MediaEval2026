"""
Submission script for MediaEval 2026 Enthymeme Detection.

Trains the best models on ALL annotated data (train+dev, 1333 tweets)
and generates predictions for the test set (148 tweets).

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

warnings.filterwarnings("ignore")

OUTPUT_DIR = config.OUTPUT_DIR
TEST_CSV = "/Users/SB/LocalProjects/DataSets/Medieeval/2026/enthymemes_2/test_v2.csv"


# ============================================================
# TASK 1: CLASSIFICATION SUBMISSION
# ============================================================

def train_and_predict_tfidf():
    """Train TF-IDF + RF on ALL data. Return predictions dict {id: {hard, probs}}."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.ensemble import RandomForestClassifier

    data = config.load_data()
    df = data["df"]
    ann_labels = data["ann_labels"]
    majority_labels = data["majority_labels"]

    texts = df["tweet_text"].values
    y = np.array([config.LABEL_TO_ID[l] for l in majority_labels])

    vectorizer = TfidfVectorizer(**config.TFIDF_DEFAULTS)
    X = vectorizer.fit_transform(texts)

    # Soft label sample weights
    weights = np.zeros(len(y))
    for i in range(len(y)):
        true_class = majority_labels[i]
        soft = ann_labels[i]
        weights[i] = soft.get(true_class, 1.0 / 3)

    model = RandomForestClassifier(**{"n_estimators": 200, "max_depth": 25, "min_samples_leaf": 1}, n_jobs=-1)
    model.fit(X, y, sample_weight=weights)

    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    preds = {}
    for i, (_, row) in enumerate(df.iterrows()):
        preds[int(row["id"])] = {
            "hard": int(y_pred[i]),
            "probs": {config.CLASS_LABELS[j]: float(y_proba[i][j]) for j in range(3)},
        }
    return preds, "tfidf"


def train_and_predict_transformer():
    """Train DistilBERT feature extraction + LR on ALL data."""
    from transformers import AutoTokenizer, AutoModel
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import normalize

    data = config.load_data()
    df = data["df"]

    tokenizer = AutoTokenizer.from_pretrained(config.TRANSFORMER_MODEL_NAME)
    try:
        model = AutoModel.from_pretrained(config.TRANSFORMER_MODEL_NAME, device_map="auto")
    except Exception:
        model = AutoModel.from_pretrained(config.TRANSFORMER_MODEL_NAME).to("cpu")
        device = torch.device("cpu")
    model.eval()

    texts = df["text"].tolist()
    y = df["label"].values

    # Extract features
    tok = tokenizer(texts, padding="max_length", truncation=True, max_length=config.TRANSFORMER_MAX_LENGTH)
    embs = []
    for i in range(0, len(texts), config.TRANSFORMER_BATCH_SIZE):
        bid = torch.tensor(tok["input_ids"][i:i + config.TRANSFORMER_BATCH_SIZE]).to(model.device)
        am = torch.tensor(tok["attention_mask"][i:i + config.TRANSFORMER_BATCH_SIZE]).to(model.device)
        with torch.no_grad():
            out = model(input_ids=bid, attention_mask=am)
        embs.append(out.last_hidden_state[:, 0, :].cpu().numpy())
    X = normalize(np.vstack(embs), norm="l2")

    clf = make_pipeline(LogisticRegression(class_weight="balanced", max_iter=1000, random_state=config.RANDOM_STATE, C=1.0))
    clf.fit(X, y)

    y_pred = clf.predict(X)
    y_proba = clf.predict_proba(X)

    preds = {}
    for i, (_, row) in enumerate(df.iterrows()):
        preds[int(row["id"])] = {
            "hard": int(y_pred[i]),
            "probs": {config.CLASS_LABELS[j]: float(y_proba[i][j]) for j in range(3)},
        }
    return preds, "transformer"


def soft_voting_submission(tfidf_preds, trans_preds, w_tfidf=0.28, w_trans=0.47):
    """Combine TF-IDF and Transformer predictions via weighted soft voting."""
    total = w_tfidf + w_trans
    ensemble = {}
    for pid in tfidf_preds:
        tp = np.array([tfidf_preds[pid]["probs"][l] for l in config.CLASS_LABELS])
        xp = np.array([trans_preds[pid]["probs"][l] for l in config.CLASS_LABELS])
        weighted = (w_tfidf * tp + w_trans * xp) / total
        ensemble[pid] = {
            "hard": int(np.argmax(weighted)),
            "probs": {config.CLASS_LABELS[j]: float(weighted[j]) for j in range(3)},
        }
    return ensemble


def submit_task1():
    """Train best models on full data, generate test predictions, save submission."""
    print("=" * 60)
    print("TASK 1: TRAINING ON FULL DATA & GENERATING SUBMISSION")
    print("=" * 60)

    # Train both models on full data
    print("\n1. Training TF-IDF + RF on full data...")
    tfidf_preds, _ = train_and_predict_tfidf()
    print(f"   Trained on {len(tfidf_preds)} instances")

    print("\n2. Training DistilBERT + LR on full data...")
    import torch
    trans_preds, _ = train_and_predict_transformer()
    print(f"   Trained on {len(trans_preds)} instances")

    # Ensemble via weighted voting
    print("\n3. Combining via weighted soft voting...")
    ensemble = soft_voting_submission(tfidf_preds, trans_preds)

    # Load test set
    print("\n4. Generating predictions for test set...")
    test_df = config.load_data(csv_path=TEST_CSV) if os.path.exists(TEST_CSV) else None
    if test_df is None:
        test_df = pd.read_csv(TEST_CSV)
        test_ids = test_df["id"].tolist()
        test_texts = test_df["tweet_text"].tolist()
    else:
        test_ids = test_df["ids"]
        test_texts = test_df["texts"]

    # Build submission format
    id2row = {int(test_df.iloc[i]["id"]): test_df.iloc[i] for i in range(len(test_df))} if not isinstance(test_df, dict) else {}

    submission = []
    pred_dist = {"premise": 0, "conclusion": 0, "none": 0}
    for i, pid in enumerate(test_ids):
        pid_int = int(pid)
        hard = ensemble.get(pid_int, {}).get("hard", 2)
        probs = ensemble.get(pid_int, {}).get("probs", {l: 1/3 for l in config.CLASS_LABELS})
        submission.append({
            "id": pid_int,
            "text": test_texts[i] if not isinstance(test_df, dict) else "",
            "label": config.ID_TO_LABEL[hard],
            "probabilities": {l: float(probs.get(l, 0)) for l in config.CLASS_LABELS},
            "hard_prediction": hard,
        })
        pred_dist[config.ID_TO_LABEL[hard]] += 1

    # Also add training predictions (full dataset submission)
    full_submission = []
    df = config.load_data()["df"]
    id2row = {int(df.iloc[i]["id"]): df.iloc[i] for i in range(len(df))}
    for i, (_, row) in enumerate(df.iterrows()):
        pid = int(row["id"])
        hard = ensemble.get(pid, {}).get("hard", 2)
        probs = ensemble.get(pid, {}).get("probs", {l: 1/3 for l in config.CLASS_LABELS})
        full_submission.append({
            "id": pid,
            "text": row["tweet_text"],
            "label": config.ID_TO_LABEL[hard],
            "probabilities": {l: float(probs.get(l, 0)) for l in config.CLASS_LABELS},
            "hard_prediction": hard,
        })

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Test set predictions
    test_pred_path = os.path.join(OUTPUT_DIR, "submit_task1_test.json")
    with open(test_pred_path, "w") as f:
        json.dump(submission, f, indent=2)
    print(f"   Test predictions saved to {test_pred_path}")

    # Full dataset predictions (for evaluation)
    full_pred_path = os.path.join(OUTPUT_DIR, "submit_task1_classifiers.json")
    with open(full_pred_path, "w") as f:
        json.dump(full_submission, f, indent=2)
    print(f"   Full dataset predictions saved to {full_pred_path}")

    print(f"\n   Prediction distribution: {pred_dist}")
    return submission, full_submission


# ============================================================
# TASK 2: PROPOSITION GENERATION SUBMISSION
# ============================================================

def submit_task2_t5():
    """Generate propositions using fine-tuned T5 on full data."""
    print("\n" + "=" * 60)
    print("TASK 2: T5 PROPOSITION GENERATION")
    print("=" * 60)

    from task2_generation.task2_generator_enhanced import load_task1_predictions, build_training_pairs, train_t5, generate_with_t5

    # Load data
    data = config.load_data()
    pairs = build_training_pairs(data)
    premise_count = sum(1 for p in pairs if p["type"] == "premise")
    conclusion_count = sum(1 for p in pairs if p["type"] == "conclusion")
    print(f"\nTraining pairs: {len(pairs)} (premise: {premise_count}, conclusion: {conclusion_count})")

    device, device_name = config.get_device()
    print(f"Device: {device_name}")

    # Check for pre-trained model
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

    # Load task1 predictions for labels
    predictions = load_task1_predictions()
    print(f"Loaded {len(predictions)} task1 predictions")

    # Generate
    print("\nGenerating propositions...")
    generated = generate_with_t5(model, tokenizer, None, predictions, device)

    # Filter to test set only
    test_ids = set()
    test_df = pd.read_csv(TEST_CSV)
    test_ids.update(test_df["id"].tolist())

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

    # Fill in missing test instances
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

    # Sort by id
    test_propositions.sort(key=lambda x: x["id"])

    # Save
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

    # Build prompts from full training data (use majority labels)
    data = config.load_data()
    df = data["df"]
    texts = data["texts"]
    majority_labels = data["majority_labels"]
    label_strings = [config.ID_TO_LABEL[int(l)] for l in majority_labels]

    print("\nGenerating propositions with Ollama...")
    results = generator.generate_batch(texts, label_strings, show_progress=True)

    # Load test IDs
    test_df = pd.read_csv(TEST_CSV)
    test_ids = set(test_df["id"].tolist())

    # Map results to test instances
    id2row = {int(df.iloc[i]["id"]): df.iloc[i] for i in range(len(df))}
    test_propositions = []
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

    # Fill missing
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


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="MediaEval 2026 Submission Generator")
    parser.add_argument("--task1", action="store_true", help="Generate Task 1 submission only")
    parser.add_argument("--task2", action="store_true", help="Generate Task 2 submission only")
    parser.add_argument("--t5", action="store_true", help="Use T5 for Task 2")
    parser.add_argument("--ollama", action="store_true", help="Use Ollama for Task 2")
    parser.add_argument("--all", action="store_true", help="Generate all submissions")
    args, _ = parser.parse_known_args()

    do_task1 = args.task1 or not args.task2 and not args.t5 and not args.ollama and not args.all
    do_task2 = args.task2 or args.t5 or args.ollama or args.all
    if args.all:
        do_task1, do_task2 = True, True

    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
