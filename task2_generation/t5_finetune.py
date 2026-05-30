"""Task 2 method: T5 + LoRA fine-tuned proposition generation.

Trains google/flan-t5-base with LoRA adapter on training implicit texts,
then generates propositions for test predictions.

Usage (from project root):
    python task2_generation/t5_finetune.py          # Train + generate
    python task2_generation/t5_finetune.py --train   # Train only
    python task2_generation/t5_finetune.py --gen     # Generate with existing model
"""

import argparse
import json
import os
import sys
import hashlib
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
import pandas as pd
import numpy as np
import torch


# ============== CONFIG ==============

T5_MODEL_DIR = os.path.join(config.OUTPUT_DIR, "task2_t5_model")
T5_TRAIN_PAIRS_FILE = os.path.join(config.OUTPUT_DIR, "task2_t5_training_pairs.json")
OUTPUT_FILE = os.path.join(config.OUTPUT_DIR, "submit_task2_propositions.json")


# ============== HELPERS ==============


def build_training_pairs(data, random_state=42):
    """Build (source, target) training pairs from annotation CSV data.

    Uses majority_label to determine which implicit text to use.
    80/20 random split (seed=42) to exclude test instances from training.
    """
    n = len(data["df"])
    rng = np.random.RandomState(random_state)
    indices = np.arange(n)
    rng.shuffle(indices)
    split = int(n * 0.8)
    train_indices = sorted(indices[:split])

    print(f"  Using {len(train_indices)}/{n} instances for training (80% split, seed={random_state})")

    pairs = []
    for i in train_indices:
        row = data["df"].iloc[i]
        label = row["majority_label"]
        if label not in ("premise", "conclusion"):
            continue

        implicit_cols = [c for c in data["df"].columns if c.endswith("_implicit")]
        implicit_text = None
        for col in implicit_cols:
            val = row.get(col)
            if pd.notna(val) and str(val).strip():
                implicit_text = str(val).strip()
                break

        if not implicit_text:
            continue

        task_type = "premise" if label == "premise" else "conclusion"
        pairs.append({
            "source": f"Generate implicit {task_type} for: {row['tweet_text']}",
            "target": implicit_text,
            "type": task_type,
        })
    return pairs


def _template_generate(tweet, task_type):
    """Template fallback when T5 produces empty output."""
    tweet_lower = tweet.lower()
    words = tweet_lower.split()
    mid_idx = max(len(words) // 3, 2)
    fill_start = max(mid_idx - 3, 0)
    fill = " ".join(words[fill_start:fill_start + 6])

    if task_type == "premise":
        templates = [
            f"This tweet assumes that {fill.capitalize()}.",
            f"The argument assumes {fill.capitalize()}.",
        ]
    else:
        templates = [
            f"Therefore, the implicit conclusion is that {fill.capitalize()}.",
            f"This reasoning leads to the conclusion that {fill.capitalize()}.",
        ]
    idx = int(hashlib.md5(tweet.encode()).hexdigest()[:4], 16) % len(templates)
    return templates[idx]


def generate_with_t5(model, tokenizer, test_preds, device):
    """Generate propositions using T5 model."""
    print("\nGenerating propositions...")

    results = []
    for pred in test_preds:
        pid = pred["id"]
        tweet = pred["text"]
        label = pred.get("label", "none")
        probs = pred.get("probabilities", {})
        confidence = float(max(probs.values())) if probs else 0.0

        if label == "none":
            results.append({
                "id": pid,
                "tweet_text": tweet,
                "predicted_label": label,
                "confidence": confidence,
                "generated_proposition": None,
            })
            continue

        task_type = "premise" if label == "premise" else "conclusion"
        prompt = f"Generate implicit {task_type} for: {tweet}"

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                max_length=config.T5_GENERATION_MAX_LENGTH,
                num_beams=config.T5_NUM_BEAMS,
                early_stopping=True,
                temperature=config.T5_TEMPERATURE,
                top_p=config.T5_TOP_P,
            )
        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if not generated or generated.strip() == "":
            generated = _template_generate(tweet, task_type)

        results.append({
            "id": pid,
            "tweet_text": tweet,
            "predicted_label": label,
            "confidence": confidence,
            "generated_proposition": generated,
        })
    return results


# ============== MAIN METHOD ==============


def train_and_generate():
    """Train T5 + LoRA from scratch, then generate propositions."""
    print("=" * 60)
    print("TASK 2: T5 + LoRA PROPOSITION GENERATION")
    print("=" * 60)

    # Load and prepare training data
    data = config.load_data()
    pairs = build_training_pairs(data)
    premise_count = sum(1 for p in pairs if p["type"] == "premise")
    conclusion_count = sum(1 for p in pairs if p["type"] == "conclusion")
    print(f"\nTraining pairs: {len(pairs)} (premise: {premise_count}, conclusion: {conclusion_count})")

    # Load test predictions
    test_pred_path = os.path.join(config.OUTPUT_DIR, "submit_task1_test.json")
    if not os.path.exists(test_pred_path):
        print(f"ERROR: {test_pred_path} not found. Run Task 1 first.")
        return None
    with open(test_pred_path) as f:
        test_preds = json.load(f)
    print(f"Loaded {len(test_preds)} test predictions")

    device, device_name = config.get_device()
    print(f"Device: {device_name}")

    # Train or load model
    if os.path.exists(T5_MODEL_DIR):
        print(f"\nLoading pre-trained T5 from {T5_MODEL_DIR}...")
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from peft import PeftModel
        adapter_path = os.path.join(T5_MODEL_DIR, "peft_adapter")
        if os.path.isdir(adapter_path) and os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
            base_model = AutoModelForSeq2SeqLM.from_pretrained(config.T5_MODEL_NAME)
            model = PeftModel.from_pretrained(base_model, adapter_path).to(device)
            tokenizer = AutoTokenizer.from_pretrained(config.T5_MODEL_NAME)
            print("  Model loaded with LoRA adapter.")
        else:
            model = AutoModelForSeq2SeqLM.from_pretrained(T5_MODEL_DIR).to(device)
            tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_DIR)
            print("  Model loaded (full model, no LoRA adapter).")
    else:
        model, tokenizer = _train_from_scratch(data, device)

    # Generate
    generated = generate_with_t5(model, tokenizer, test_preds, device)

    # Save
    output_path = OUTPUT_FILE
    with open(output_path, "w") as f:
        json.dump(generated, f, indent=2)

    gen_count = sum(1 for g in generated if g.get("generated_proposition"))
    print(f"\nPropositions saved to {output_path}")
    print(f"Generated: {gen_count}/{len(generated)}")
    return generated


def _train_from_scratch(data, device):
    """Train flan-t5-base + LoRA from scratch."""
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, TaskType
    from torch.utils.data import DataLoader, TensorDataset, RandomSampler
    from torch.optim import AdamW
    from sklearn.model_selection import train_test_split as _train_test_split
    from tqdm import tqdm

    T5_TRAIN_PAIRS_FILE = os.path.join(config.OUTPUT_DIR, "task2_t5_training_pairs.json")
    if os.path.exists(T5_TRAIN_PAIRS_FILE):
        with open(T5_TRAIN_PAIRS_FILE) as f:
            pairs = json.load(f)
        print(f"  Loaded {len(pairs)} training pairs from cache")
    else:
        pairs = build_training_pairs(data)
        with open(T5_TRAIN_PAIRS_FILE, "w") as f:
            json.dump(pairs, f)
        print(f"  Built and cached {len(pairs)} training pairs")

    print(f"\n  Loading {config.T5_MODEL_NAME}...")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(config.T5_MODEL_NAME)

    lora_config = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.1,
        target_modules=["q", "k", "v", "o"],
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(config.T5_MODEL_NAME)

    train_pairs, val_pairs = _train_test_split(pairs, test_size=0.2, random_state=config.RANDOM_STATE)
    print(f"  Train: {len(train_pairs)}, Val: {len(val_pairs)}")

    def tokenize_pair(pairs, tokenizer, max_len):
        sources = [p["source"] for p in pairs]
        targets = [p["target"] for p in pairs]
        source_enc = tokenizer(sources, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt")
        target_enc = tokenizer(targets, max_length=max_len, padding="max_length", truncation=True, return_tensors="pt")
        return source_enc, target_enc

    train_source, train_target = tokenize_pair(train_pairs, tokenizer, config.T5_MAX_LENGTH)
    train_dataset = TensorDataset(
        train_source["input_ids"], train_source["attention_mask"],
        train_target["input_ids"], train_target["attention_mask"],
    )
    train_loader = DataLoader(train_dataset, batch_size=config.T5_BATCH_SIZE, sampler=RandomSampler(train_dataset))

    val_source, val_target = tokenize_pair(val_pairs, tokenizer, config.T5_MAX_LENGTH)
    val_dataset = TensorDataset(
        val_source["input_ids"], val_source["attention_mask"],
        val_target["input_ids"], val_target["attention_mask"],
    )
    val_loader = DataLoader(val_dataset, batch_size=config.T5_BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=config.T5_LEARNING_RATE)
    best_loss = float("inf")

    print(f"\n  Fine-tuning {config.T5_MODEL_NAME} for {config.T5_NUM_EPOCHS} epochs on {device.type}...\n")
    for epoch in range(config.T5_NUM_EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.T5_NUM_EPOCHS}"):
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
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
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss.item()
        avg_val = val_loss / len(val_loader)
        print(f"    train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

        if avg_val < best_loss:
            best_loss = avg_val

    model.save_pretrained(os.path.join(T5_MODEL_DIR, "peft_adapter"))
    tokenizer.save_pretrained(T5_MODEL_DIR)
    base_model.config.save_pretrained(T5_MODEL_DIR)
    print(f"\n  Trained LoRA adapter saved to {T5_MODEL_DIR}/peft_adapter/")
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="T5 + LoRA proposition generation")
    parser.add_argument("--train", action="store_true", help="Train only")
    parser.add_argument("--gen", action="store_true", help="Generate only (use existing model)")
    args = parser.parse_args()

    if args.train and args.gen:
        train_and_generate()
    elif args.train:
        train_and_generate()
    else:
        train_and_generate()
