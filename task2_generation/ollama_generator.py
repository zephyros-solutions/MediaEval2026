"""Task 2 method: Ollama-based proposition generation.

Uses Ollama LLM (gemma4 > qwen3.6 > mistral) to generate implicit
premises and conclusions for test predictions.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
import pandas as pd

OUTPUT_DIR = config.OUTPUT_DIR
TEST_CSV = config.TEST_CSV_PATH


def _select_generation_model(client):
    """Select best available generation model from Ollama.

    Preference: gemma4 > qwen3.6 > mistral
    """
    available_models = client.get_available_models()
    print(f"   Available Ollama models: {', '.join(available_models)}")

    for preferred in ["gemma4", "qwen3.6", "mistral"]:
        if preferred in available_models:
            return preferred
    return None


def generate_with_ollama():
    """Generate propositions using Ollama LLM for test predictions.

    Uses task1 ensemble predictions to determine which tweets need generation.
    Only premise/conclusion predictions get propositions generated.

    Returns:
        list of proposition dicts or None if Ollama unavailable
    """
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

    gen_model = _select_generation_model(client)
    if not gen_model:
        print("  ERROR: No compatible generation model found in Ollama.")
        return None
    print(f"  Using generation model: {gen_model}")

    # Load task1 predictions to get labels for test instances
    test_df = pd.read_csv(TEST_CSV)
    test_ids = set(test_df["id"].tolist())

    task1_pred_path = os.path.join(OUTPUT_DIR, "submit_task1_test.json")
    id2label = {}
    if os.path.exists(task1_pred_path):
        with open(task1_pred_path) as f:
            task1_preds = json.load(f)
        id2label = {p["id"]: p["label"] for p in task1_preds}

    # Build test text/label lists in CSV order
    test_texts = []
    test_pids = []
    test_labels = []
    for _, row in test_df.iterrows():
        pid = int(row["id"])
        test_texts.append(row["tweet_text"])
        test_pids.append(pid)
        test_labels.append(id2label.get(pid, "none"))

    print(f"\nGenerating propositions with Ollama ({gen_model}) for {len(test_texts)} test instances...")
    generator = OllamaGenerator(model=gen_model, client=client)
    results = generator.generate_batch(test_texts, test_labels, show_progress=True)

    # Map index-based results to test IDs
    id2result = {int(r.get("id", i)): r for i, r in enumerate(results)}
    test_propositions = []
    for i, pid in enumerate(test_pids):
        r = id2result.get(i, id2result.get(pid, None))
        if r:
            test_propositions.append({
                "id": pid,
                "tweet_text": r.get("tweet", test_texts[i]),
                "predicted_label": r.get("label", test_labels[i]),
                "confidence": 0.0,
                "generated_proposition": r.get("generated_proposition"),
            })
        else:
            test_propositions.append({
                "id": pid,
                "tweet_text": test_texts[i],
                "predicted_label": test_labels[i],
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


if __name__ == "__main__":
    generate_with_ollama()
