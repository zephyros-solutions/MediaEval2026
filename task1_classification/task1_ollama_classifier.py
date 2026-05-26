"""
Task 1: Ollama (Mistral) Zero-Shot Classification

Approach:
  1. Load annotations from shared config
  2. Classify with Mistral (no examples)
  3. Dirichlet-smoothed probability vectors
  4. Cross-entropy against soft gold labels

Output:
  outputs/predictions_classifiers.json   (challenge format with probabilities)
  outputs/evaluation_report.json        (2-class F1 + cross-entropy)

Callable interface:
  from task1_classification.task1_ollama_classifier import run_ollama_zero
  predictions, report = run_ollama_zero()
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluation.metrics import compute_metrics
import config
from core.ollama_integration import OllamaClient, OllamaClassifier


def _run_pipeline():
    """Run the full Ollama zero-shot pipeline. Returns (predictions, report)."""
    print("=" * 80)
    print("TASK 1: OLLAMA ZERO-SHOT CLASSIFICATION (Mistral)")
    print("=" * 80)

    client = OllamaClient()
    if not client.check_connection():
        raise RuntimeError("Ollama not running. Run: ollama serve")

    data = config.load_data()
    texts = data["texts"]
    ann_labels = data["ann_labels"]
    true_labels = data["majority_labels"]

    print(f"Loaded {len(texts)} instances")

    classifier = OllamaClassifier(model="mistral", client=client)
    results = classifier.classify_batch(texts, show_progress=True)

    hard_preds = results["hard_predictions"]

    def dirichlet_smooth(p, conf=0.85, prior=0.3):
        probs = np.ones(3) * (1.0 / 3.0) * prior
        probs[p] += conf
        return probs / probs.sum()
    prob_vectors = [dirichlet_smooth(p) for p in hard_preds]

    true_labels = np.asarray(true_labels).ravel()
    hard_preds = np.asarray(hard_preds).ravel()
    prob_dicts = [{config.CLASS_LABELS[j]: float(prob_vectors[i][j]) for j in range(3)} for i in range(len(texts))]
    metrics, _ = compute_metrics(
        y_true=true_labels, y_pred=hard_preds,
        prob_vectors=prob_dicts, ann_labels=ann_labels, method_name="ollama_zero",
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
        "model": "mistral_7b_zero_shot",
        "hyperparameters": {"temperature": config.FEWSHOT_TEMPERATURE},
        "test_metrics": {k: float(v) for k, v in metrics.items() if k != "per_class" and k != "method"},
        "per_class": metrics["per_class"],
        "prediction_distribution": {config.CLASS_LABELS[j]: int(dist[j]) for j in range(3)},
    }

    print(f"F1(3-class): {metrics['f1_macro_3class']:.4f}  F1(2-class): {metrics['f1_macro_2class']:.4f}  CE: {metrics['cross_entropy']:.4f}")
    return predictions, report


def run_ollama_zero():
    """Run Ollama zero-shot classifier. Returns (predictions, report)."""
    return _run_pipeline()


if __name__ == "__main__":
    run_ollama_zero()
