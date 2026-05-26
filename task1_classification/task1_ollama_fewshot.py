"""
Task 1: Ollama (Mistral) Few-Shot Classification

Approach:
  1. Load annotations from shared config
  2. Balanced stratified sampling: FEWSHOT_EXAMPLES_PER_CLASS per class
  3. Multi-shot averaging (FEWSHOT_NUM_ROUNDS rounds, different examples each)
  4. Dirichlet-smoothed probability vectors
  5. Cross-entropy against soft gold labels

Output:
  outputs/predictions_classifiers.json   (challenge format with probabilities)
  outputs/evaluation_report.json        (2-class F1 + cross-entropy)

Callable interface:
  from task1_classification.task1_ollama_fewshot import run_ollama_fewshot
  predictions, report = run_ollama_fewshot()
"""

import sys
import numpy as np
import os
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from evaluation.metrics import compute_metrics
import config
from core.ollama_integration import OllamaClient, OllamaClassifier


def _run_pipeline():
    """Run the full Ollama few-shot pipeline. Returns (predictions, report)."""
    print("=" * 80)
    print("TASK 1: OLLAMA FEW-SHOT CLASSIFICATION (Mistral)")
    print("=" * 80)

    client = OllamaClient()
    if not client.check_connection():
        raise RuntimeError("Ollama not running")

    data = config.load_data()
    df = data["df"]
    texts = data["texts"]
    ann_labels = data["ann_labels"]
    true_labels = data["majority_labels"]

    print(f"Loaded {len(texts)} instances")

    def get_examples(df, per_class):
        examples = {}
        for label in config.CLASS_LABELS:
            label_df = df[df["majority_label"] == label]
            if len(label_df) >= per_class:
                sss = StratifiedShuffleSplit(n_splits=1, test_size=per_class, random_state=config.RANDOM_STATE)
                for _, test_idx in sss.split(label_df, label_df["majority_label"]):
                    examples[label] = label_df.iloc[test_idx]["tweet_text"].tolist()
            else:
                examples[label] = label_df["tweet_text"].tolist()
        return examples

    def build_prompt(examples, tweet):
        lines = ["Classify the following tweet. Here are examples:\n\n"]
        for label in config.CLASS_LABELS:
            if label == "none": cat = "NONE (no implicit argument):"
            elif label == "premise": cat = "PREMISE (unstated assumption):"
            else: cat = "CONCLUSION (unstated inference):"
            lines.append(f"\n{cat}")
            for ex in examples[label]:
                lines.append(f'  - "{ex}"')
        lines.append(f'\n\nNow classify this tweet:\nTweet: "{tweet}"\nAnswer with ONE WORD ONLY: none, premise, or conclusion')
        return "\n".join(lines)

    # Multi-shot averaging
    all_round_preds = [[] for _ in range(config.FEWSHOT_NUM_ROUNDS)]
    for r in range(config.FEWSHOT_NUM_ROUNDS):
        round_ex = get_examples(df, config.FEWSHOT_EXAMPLES_PER_CLASS)
        print(f"  Round {r + 1}/{config.FEWSHOT_NUM_ROUNDS}...")
        for tweet in texts:
            response = client.generate("mistral", build_prompt(round_ex, tweet), temperature=config.FEWSHOT_TEMPERATURE, num_predict=5)
            if not response:
                all_round_preds[r].append(config.LABEL_TO_ID["none"])
                continue
            rl = response.lower().strip()
            for key in config.LABEL_TO_ID:
                if rl.startswith(key):
                    all_round_preds[r].append(config.LABEL_TO_ID[key])
                    break
            else:
                all_round_preds[r].append(config.LABEL_TO_ID["none"])

    prob_vectors = []
    for i in range(len(texts)):
        counts = np.zeros(3)
        for r in range(config.FEWSHOT_NUM_ROUNDS):
            counts[all_round_preds[r][i]] += 1
        prob_vectors.append(counts / config.FEWSHOT_NUM_ROUNDS)
    hard_preds = [int(np.argmax(p)) for p in prob_vectors]

    true_labels = np.asarray(true_labels).ravel()
    hard_preds = np.asarray([int(x) for x in hard_preds]).ravel()
    prob_dicts = [{config.CLASS_LABELS[j]: float(prob_vectors[i][j]) for j in range(3)} for i in range(len(texts))]
    metrics, _ = compute_metrics(
        y_true=true_labels, y_pred=hard_preds,
        prob_vectors=prob_dicts, ann_labels=ann_labels, method_name="ollama_fewshot",
    )
    dist = np.bincount(hard_preds, minlength=3)

    predictions = []
    for i in range(len(texts)):
        predictions.append({
            "id": int(data["ids"][i]),
            "text": texts[i],
            "label": config.ID_TO_LABEL[int(hard_preds[i])],
            "probabilities": prob_dicts[i],
            "hard_prediction": int(hard_preds[i]),
        })

    report = {
        "model": "mistral_7b_few_shot",
        "hyperparameters": {"examples_per_class": config.FEWSHOT_EXAMPLES_PER_CLASS, "num_rounds": config.FEWSHOT_NUM_ROUNDS, "temperature": config.FEWSHOT_TEMPERATURE},
        "test_metrics": {k: float(v) for k, v in metrics.items() if k not in ("per_class", "method")},
        "per_class": metrics["per_class"],
        "prediction_distribution": {config.CLASS_LABELS[j]: int(dist[j]) for j in range(3)},
    }

    print(f"F1(3-class): {metrics['f1_macro_3class']:.4f}  F1(2-class): {metrics['f1_macro_2class']:.4f}  CE: {metrics['cross_entropy']:.4f}")
    return predictions, report


def run_ollama_fewshot():
    """Run Ollama few-shot classifier. Returns (predictions, report)."""
    return _run_pipeline()


if __name__ == "__main__":
    run_ollama_fewshot()
