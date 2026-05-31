"""Task 2 evaluation: compare T5 vs Ollama generation quality on validation data.

Builds (source, target) pairs from the annotation CSV using majority_label.
80% for training the models, 20% for evaluation.
For each 20% instance with a ground-truth implicit text, generates propositions
with both methods and computes:
- Lexical overlap (precision/recall/F1 via token intersection)
- Semantic similarity via sentence embeddings (all-MiniLM-L6-v2)
- Coverage (proportion of instances that produced output)
- Average length
"""

import json
import os
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


OUTPUT_DIR = config.OUTPUT_DIR
TRAIN_VAL_SPLIT = config.TRAIN_VAL_SPLIT
RANDOM_STATE = config.RANDOM_STATE


def _get_implicit_text(row):
    """Get the first non-empty implicit text from annotation columns."""
    for i in range(1, 6):
        val = row.get(f"ann{i}_implicit")
        if pd.notna(val) and str(val).strip():
            return str(val).strip()
    return None


def _build_val_set(data):
    """Build validation set from the 20% held-out split.

    Returns (train_pairs, val_instances). Each val instance has:
    id, tweet_text, ground_truth, type (premise/conclusion)
    """
    n = len(data["df"])
    rng = np.random.RandomState(RANDOM_STATE)
    indices = np.arange(n)
    rng.shuffle(indices)
    split = int(n * TRAIN_VAL_SPLIT)
    train_indices = sorted(indices[:split])
    val_indices = sorted(indices[split:])

    train_pairs = []
    for i in train_indices:
        row = data["df"].iloc[i]
        label = row["majority_label"]
        if label not in ("premise", "conclusion"):
            continue
        implicit_text = _get_implicit_text(row)
        if not implicit_text:
            continue
        task_type = "premise" if label == "premise" else "conclusion"
        train_pairs.append({
            "id": int(row["id"]),
            "source": f"Generate implicit {task_type} for: {row['tweet_text']}",
            "target": implicit_text,
            "type": task_type,
            "tweet_text": row["tweet_text"],
        })

    val = []
    for i in val_indices:
        row = data["df"].iloc[i]
        label = row["majority_label"]
        if label not in ("premise", "conclusion"):
            continue
        implicit_text = _get_implicit_text(row)
        if not implicit_text:
            continue
        val.append({
            "id": int(row["id"]),
            "tweet_text": row["tweet_text"],
            "ground_truth": implicit_text,
            "type": label,
        })

    return train_pairs, val


def _compute_lexical(generated, ground_truth):
    """Compute token-level precision/recall/F1 between generated and ground truth."""
    gen_tokens = set(generated.lower().split())
    gt_tokens = set(ground_truth.lower().split())
    if not gen_tokens:
        return 0.0, 0.0, 0.0
    intersection = len(gen_tokens & gt_tokens)
    precision = intersection / len(gen_tokens)
    recall = intersection / len(gt_tokens)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _compute_semantic_sim(generated, ground_truth):
    """Compute semantic similarity via sentence embeddings (all-MiniLM-L6-v2)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    gen_emb = model.encode([generated], show_progress_bar=False, normalize_embeddings=True)
    gt_emb = model.encode([ground_truth], show_progress_bar=False, normalize_embeddings=True)
    return float(np.dot(gen_emb[0], gt_emb[0]))


def _avg_length(text):
    return len(text.split()) if text else 0


def _compute_metrics(generated_fn, val):
    """Run generation function on validation set and compute metrics.

    Args:
        generated_fn: function(tweet_text, task_type) -> str | None
        val: validation instances list
    Returns:
        metrics dict or None if no val instances
    """
    if not val:
        return None

    val_types = {}
    for v in val:
        val_types[v["type"]] = val_types.get(v["type"], 0) + 1

    results = {"precision": [], "recall": [], "f1": [], "semantic_sim": [], "lengths": [], "coverage": []}
    count = 0

    for v in val:
        generated = generated_fn(v["tweet_text"], v["type"])
        if generated:
            p, r, f = _compute_lexical(generated, v["ground_truth"])
            s = _compute_semantic_sim(generated, v["ground_truth"])
            results["precision"].append(p)
            results["recall"].append(r)
            results["f1"].append(f)
            results["semantic_sim"].append(s)
            results["lengths"].append(_avg_length(generated))
            results["coverage"].append(1)
        else:
            results["coverage"].append(0)
        count += 1

    gen_count = sum(results["coverage"])
    total = len(val)
    return {
        "total_instances": total,
        "generated": gen_count,
        "coverage": gen_count / total if total else 0,
        "bleu_precision": float(np.mean(results["precision"])) if results["precision"] else 0,
        "bleu_recall": float(np.mean(results["recall"])) if results["recall"] else 0,
        "bleu_f1": float(np.mean(results["f1"])) if results["f1"] else 0,
        "semantic_similarity": float(np.mean(results["semantic_sim"])) if results["semantic_sim"] else 0,
        "avg_length": float(np.mean(results["lengths"])) if results["lengths"] else 0,
    }


def evaluate_t5():
    """Evaluate T5 generation on validation data.

    Trains T5 from scratch on the 80% training split, then evaluates on 20%.
    Returns metrics dict or None on failure.
    """
    data = config.load_data()
    pairs, val = _build_val_set(data)
    print(f"\n  Training pairs: {len(pairs)}")
    print(f"  Validation instances: {len(val)}")
    if not val:
        return None

    # Load or train T5 model
    t5_model_dir = os.path.join(OUTPUT_DIR, "task2_t5_model")
    t5_pairs_file = os.path.join(OUTPUT_DIR, "task2_t5_training_pairs.json")

    # Build training pairs if needed (reuse the same logic as t5_finetune.py)
    if not os.path.exists(t5_pairs_file):
        from random import Random
        rng = Random(42)
        n = len(data["df"])
        indices = list(range(n))
        rng.shuffle(indices)
        split = int(n * 0.8)
        train_idx = sorted(indices[:split])
        pairs = []
        for i in train_idx:
            row = data["df"].iloc[i]
            label = row["majority_label"]
            if label not in ("premise", "conclusion"):
                continue
            it = _get_implicit_text(row)
            if not it:
                continue
            task_type = "premise" if label == "premise" else "conclusion"
            pairs.append({
                "source": f"Generate implicit {task_type} for: {row['tweet_text']}",
                "target": it,
                "type": task_type,
            })
        with open(t5_pairs_file, "w") as f:
            json.dump(pairs, f)
        print(f"  Built and cached {len(pairs)} training pairs")

    # Train T5 if no model exists
    if not os.path.isdir(os.path.join(t5_model_dir, "peft_adapter")):
        print("  Training T5 from scratch (this may take a few minutes)...")
        _train_t5(data, t5_model_dir, t5_pairs_file)

    # Load model for generation
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from peft import PeftModel
    import torch

    base = AutoModelForSeq2SeqLM.from_pretrained(config.T5_MODEL_NAME)
    adapter_path = os.path.join(t5_model_dir, "peft_adapter")
    model = PeftModel.from_pretrained(base, adapter_path).to("cpu")
    tokenizer = AutoTokenizer.from_pretrained(config.T5_MODEL_NAME)

    def t5_gen(tweet_text, task_type):
        prompt = f"Generate implicit {task_type} for: {tweet_text}"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, max_length=128, early_stopping=True)
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated if generated and generated.strip() else None

    metrics = _compute_metrics(t5_gen, val)
    if metrics:
        _print_metrics("T5", metrics)
    return metrics


def evaluate_ollama():
    """Evaluate Ollama generation on validation data.

    Returns metrics dict or None on failure.
    """
    try:
        from core.ollama_integration import OllamaClient, OllamaGenerator
    except ImportError:
        print("  Ollama integration not available.")
        return None

    client = OllamaClient()
    if not client.check_connection():
        print("  Ollama server not running.")
        return None

    available = client.get_available_models()
    print(f"  Available Ollama models: {', '.join(available)}")

    gen_model = "gemma4"
    if gen_model not in available and "qwen3.6" in available:
        gen_model = "qwen3.6"
    elif gen_model not in available and "mistral" in available:
        gen_model = "mistral"
    elif not available:
        print("  No models available.")
        return None

    generator = OllamaGenerator(model=gen_model, client=client)

    # Get validation set
    data = config.load_data()
    _, val = _build_val_set(data)
    print(f"\n  Validation instances: {len(val)}")
    if not val:
        return None

    def ollama_gen(tweet_text, task_type):
        result = generator.generate(tweet_text, task_type)
        return result.get("generated_proposition") if result and result.get("generated_proposition") else None

    metrics = _compute_metrics(ollama_gen, val)
    if metrics:
        _print_metrics("Ollama", metrics)
    return metrics


def compare_and_choose(metric="bleu_f1"):
    """Evaluate both T5 and Ollama on validation data and return the best method.

    Args:
        metric: which metric to use for choosing.
                'bleu_f1' = lexical overlap F1 (primary for this challenge)
                'semantic_similarity' = embedding cosine similarity

    Returns:
        (chosen_method, t5_metrics, ollama_metrics) or (None, None, None)
    """
    print("=" * 60)
    print("TASK 2: GENERATION METHOD COMPARISON")
    print("=" * 60)

    print("\n--- Evaluating T5 ---")
    t5_metrics = None
    try:
        t5_metrics = evaluate_t5()
    except Exception as e:
        print(f"  T5 evaluation failed: {e}")

    print("\n--- Evaluating Ollama ---")
    ollama_metrics = None
    try:
        ollama_metrics = evaluate_ollama()
    except Exception as e:
        print(f"  Ollama evaluation failed: {e}")

    # Compare
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)

    if not t5_metrics and not ollama_metrics:
        print("Both methods failed. Cannot choose.")
        return None, None, None

    if not t5_metrics:
        print("\n>>> Best method: Ollama (T5 unavailable)")
        return "ollama", None, ollama_metrics
    if not ollama_metrics:
        print("\n>>> Best method: T5 (Ollama unavailable)")
        return "t5", t5_metrics, None

    _print_comparison(t5_metrics, ollama_metrics)

    t5_score = t5_metrics.get(metric, 0)
    ollama_score = ollama_metrics.get(metric, 0)

    if t5_score > ollama_score:
        print(f"\n>>> Best method: T5 ({metric}: {t5_score:.4f} vs {ollama_score:.4f})")
        return "t5", t5_metrics, ollama_metrics
    elif ollama_score > t5_score:
        print(f"\n>>> Best method: Ollama ({metric}: {ollama_score:.4f} vs {t5_score:.4f})")
        return "ollama", t5_metrics, ollama_metrics
    else:
        print(f"\n>>> Tie ({metric}: {t5_score:.4f}). Defaulting to T5.")
        return "t5", t5_metrics, ollama_metrics


def _train_t5(data, model_dir, pairs_file):
    """Train flan-t5-base + LoRA from scratch (borrowed from t5_finetune.py)."""
    import json
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    from torch.utils.data import DataLoader, TensorDataset, RandomSampler
    from torch.optim import AdamW
    from sklearn.model_selection import train_test_split as _train_test_split
    from tqdm import tqdm

    with open(pairs_file) as f:
        pairs = json.load(f)

    print(f"  Loaded {len(pairs)} training pairs")

    base = AutoModelForSeq2SeqLM.from_pretrained(config.T5_MODEL_NAME)
    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.1,
        target_modules=["q", "k", "v", "o"],
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(base, lora_config)
    model.print_trainable_parameters()
    model = model.to("cpu")

    tokenizer = AutoTokenizer.from_pretrained(config.T5_MODEL_NAME)

    train_pairs, val_pairs = _train_test_split(pairs, test_size=0.2, random_state=config.RANDOM_STATE)
    print(f"  Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    max_len = config.T5_MAX_LENGTH
    batch_size = config.T5_BATCH_SIZE
    num_epochs = config.T5_NUM_EPOCHS

    def tokenize(pairs):
        src = tokenizer([p["source"] for p in pairs], max_length=max_len, padding="max_length", truncation=True, return_tensors="pt")
        tgt = tokenizer([p["target"] for p in pairs], max_length=max_len, padding="max_length", truncation=True, return_tensors="pt")
        return src, tgt

    ts, tt = tokenize(train_pairs)
    train_ds = TensorDataset(ts["input_ids"], ts["attention_mask"], tt["input_ids"], tt["attention_mask"])
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=RandomSampler(train_ds))

    vs, vt = tokenize(val_pairs)
    val_ds = TensorDataset(vs["input_ids"], vs["attention_mask"], vt["input_ids"], vt["attention_mask"])
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    optimizer = AdamW(model.parameters(), lr=config.T5_LEARNING_RATE)
    best_loss = float("inf")

    print(f"  Fine-tuning {config.T5_MODEL_NAME} for {num_epochs} epochs on CPU...\n")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch[0]
            attention_mask = batch[1]
            labels = batch[2]
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train = total_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch[0]
                attention_mask = batch[1]
                labels = batch[2]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
        avg_val = val_loss / len(val_loader)
        print(f"    train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")
        if avg_val < best_loss:
            best_loss = avg_val

    model.save_pretrained(os.path.join(model_dir, "peft_adapter"))
    tokenizer.save_pretrained(model_dir)
    base.config.save_pretrained(model_dir)
    print(f"  Model saved to {model_dir}/peft_adapter/")


def _print_metrics(name, m):
    print(f"\n  [{name}]")
    print(f"    Coverage:        {m['coverage']:.1%} ({m['generated']}/{m['total_instances']})")
    print(f"    BLEU precision:  {m['bleu_precision']:.4f}")
    print(f"    BLEU recall:     {m['bleu_recall']:.4f}")
    print(f"    BLEU F1:         {m['bleu_f1']:.4f}")
    print(f"    Semantic sim:    {m['semantic_similarity']:.4f}")
    print(f"    Avg length:      {m['avg_length']:.1f}")


def _print_comparison(t5, ollama):
    print(f"\n  {'Metric':<18} {'T5':>10} {'Ollama':>10} {'Winner':>10}")
    print(f"  {'-' * 16:<18} {'-' * 10:>10} {'-' * 10:>10} {'-' * 10:>10}")
    for name, t5_val, ol_val in [
        ("Coverage", t5["coverage"], ollama["coverage"]),
        ("BLEU precision", t5["bleu_precision"], ollama["bleu_precision"]),
        ("BLEU recall", t5["bleu_recall"], ollama["bleu_recall"]),
        ("BLEU F1", t5["bleu_f1"], ollama["bleu_f1"]),
        ("Semantic sim", t5["semantic_similarity"], ollama["semantic_similarity"]),
        ("Avg length", t5["avg_length"], ollama["avg_length"]),
    ]:
        winner = "T5" if t5_val > ol_val else ("Ollama" if ol_val > t5_val else "Tie")
        print(f"  {name:<18} {t5_val:>10.4f} {ol_val:>10.4f} {winner:>10}")


# Create output directories once at module load time
os.makedirs(config.OUTPUT_DIR, exist_ok=True)


if __name__ == "__main__":
    chosen, t5_m, ollama_m = compare_and_choose()
    if chosen:
        print(f"\n{'=' * 60}")
        print(f"RECOMMENDATION: Use {chosen.upper()} for Task 2 submission")
        print(f"{'=' * 60}")
