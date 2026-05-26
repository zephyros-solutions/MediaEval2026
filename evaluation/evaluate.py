"""
Unified evaluation for MediaEval Task 1 (classification) and Task 2 (generation).

Usage (from project root):
    python evaluation/evaluate.py                  # Evaluate all available methods
    python evaluation/evaluate.py --task 1           # Task 1 only
    python evaluation/evaluate.py --task 2           # Task 2 only
    python evaluation/evaluate.py --method tfidf     # Specific classifier
    python evaluation/evaluate.py --method transformer
    python evaluation/evaluate.py --method ollama_zero
    python evaluation/evaluate.py --method ollama_fewshot
    python evaluation/evaluate.py --method template  # Task 2: template generator
    python evaluation/evaluate.py --method ollama     # Task 2: Ollama generator
    python evaluation/evaluate.py --method t5         # Task 2: T5 generator
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import config


from evaluation.metrics import compute_metrics

OUTPUT_DIR = config.OUTPUT_DIR

# ---- Mapping: method key -> (predictions file, evaluation function) ----

TASK1_METHODS = {
    "tfidf": ("predictions_classifiers_tfidf.json", "evaluate_task1"),
    "transformer": ("predictions_classifiers_transformer.json", "evaluate_task1"),
    "ollama_zero": ("predictions_classifiers_ollama_zero.json", "evaluate_task1"),
    "ollama_fewshot": ("predictions_classifiers_ollama_fewshot.json", "evaluate_task1"),
}

TASK2_METHODS = {
    "template": ("task2_generated_propositions.json", "evaluate_task2"),
    "ollama": ("task2_generated_propositions_ollama.json", "evaluate_task2"),
    "t5": ("task2_generated_propositions_enhanced.json", "evaluate_task2"),
}


# ===== TASK 1 EVALUATION =====

def evaluate_task1(preds_path, report_prefix=""):
    """Evaluate a task1 classification prediction file against the CSV."""
    with open(preds_path) as f:
        predictions = json.load(f)

    data = config.load_data()
    id_to_majority = {data["ids"][i]: int(data["majority_labels"][i]) for i in range(len(data["ids"]))}
    id_to_soft = {data["ids"][i]: data["ann_labels"][i] for i in range(len(data["ids"]))}

    y_true = []
    y_pred = []
    prob_dicts = []
    soft_labels = []
    for pred in predictions:
        pid = pred["id"]
        if pid not in id_to_majority:
            continue
        y_true.append(id_to_majority[pid])
        y_pred.append(int(pred["hard_prediction"]))
        prob_dicts.append({config.CLASS_LABELS[j]: float(pred["probabilities"].get(l, 0)) for l, j in enumerate(config.CLASS_LABELS)})
        soft_labels.append(id_to_soft[pid])

    metrics, per_class = compute_metrics(
        y_true=y_true, y_pred=y_pred,
        prob_vectors=prob_dicts, ann_labels=soft_labels, method_name=report_prefix,
    )

    pred_dist = {config.CLASS_LABELS[j]: int(np.sum(np.array(y_pred) == j)) for j in range(3)}

    report = {
        "method": os.path.basename(preds_path).replace(".json", ""),
        "total_instances": len(predictions),
        "matched": len(y_true),
        "metrics": {
            "3class": {
                "macro_f1": metrics["f1_macro_3class"],
                "weighted_f1": metrics["f1_weighted_3class"],
            },
            "2class": {
                "macro_f1": metrics["f1_macro_2class"],
                "weighted_f1": metrics["f1_weighted_2class"],
                "note": "PRIMARY RANKING METRIC",
            },
            "per_class": per_class,
            "cross_entropy": metrics["cross_entropy"],
        },
        "prediction_distribution": pred_dist,
    }

    return report, {
        "f1_3macro": metrics["f1_macro_3class"],
        "f1_2macro": metrics["f1_macro_2class"],
        "ce": metrics["cross_entropy"],
        "per_class": per_class,
    }


# ===== TASK 2 EVALUATION =====

def evaluate_task2(preds_path, data):
    """Evaluate task2 generation predictions against annotator implicit texts."""
    with open(preds_path) as f:
        propositions = json.load(f)

    # Build id -> annotator implicit texts
    id_to_ann = {}
    for i, imp_texts in enumerate(data["implicit_texts"]):
        if imp_texts:
            id_to_ann[data["ids"][i]] = imp_texts

    candidates = [g for g in propositions if g.get("generated_proposition")]
    if not candidates:
        return {}, {}

    prec_scores, rec_scores, f1_scores = [], [], []
    for g in candidates:
        gen = g["generated_proposition"].lower().split()
        refs = id_to_ann.get(g["id"])
        if not refs:
            continue
        for ref_text in refs:
            ref = ref_text.lower().split()
            if not ref:
                continue
            common = len(set(gen) & set(ref))
            p = common / max(len(gen), 1)
            r = common / max(len(ref), 1)
            f = 2 * p * r / max(p + r, 1e-10)
            prec_scores.append(p)
            rec_scores.append(r)
            f1_scores.append(f)

    avg_length = float(np.mean([len(g["generated_proposition"].split()) for g in candidates]))
    coverage = len(candidates) / len(propositions) * 100

    report = {
        "method": os.path.basename(preds_path).replace(".json", ""),
        "total_instances": len(propositions),
        "propositions_generated": len(candidates),
        "coverage_pct": coverage,
        "metrics": {
            "lexical": {
                "precision": float(np.mean(prec_scores)) if prec_scores else 0,
                "recall": float(np.mean(rec_scores)) if rec_scores else 0,
                "f1": float(np.mean(f1_scores)) if f1_scores else 0,
            },
            "avg_length_words": avg_length,
        },
        "prediction_distribution": {
            "premise": sum(1 for g in propositions if g["predicted_label"] == "premise"),
            "conclusion": sum(1 for g in propositions if g["predicted_label"] == "conclusion"),
            "none": sum(1 for g in propositions if g["predicted_label"] == "none"),
        },
    }

    return report, {
        "lexical_f1": float(np.mean(f1_scores)) if f1_scores else 0,
        "coverage_pct": coverage,
        "avg_length": avg_length,
    }


# ===== DISPLAY =====

def print_classification_report(metrics):
    print(f"\n  {'3-Class Macro F1':<25} {metrics['f1_3macro']:.4f}")
    print(f"  {'3-Class Weighted F1':<25} {metrics['f1_3weighted']:.4f}")
    print(f"  {'2-Class Macro F1 (PRIMARY)':<25} {metrics['f1_2macro']:.4f}")
    print(f"  {'2-Class Weighted F1':<25} {metrics['f1_2weighted']:.4f}")
    print(f"  {'Cross-Entropy':<25} {metrics['ce']:.4f}")
    print(f"\n  Per-class F1:")
    for label, f1 in metrics["per_class"].items():
        print(f"    {label:<12} F1={f1['f1']:.4f}  P={f1['precision']:.4f}  R={f1['recall']:.4f}  (n={f1['support']})")


def print_generation_report(metrics):
    lex = metrics["lexical"]
    print(f"\n  {'Lexical Precision':<25} {lex['precision']:.4f}")
    print(f"  {'Lexical Recall':<25} {lex['recall']:.4f}")
    print(f"  {'Lexical F1':<25} {lex['f1']:.4f}")
    print(f"  {'Coverage':<25} {metrics['coverage_pct']:.1f}%")
    print(f"  {'Avg Length':<25} {metrics['avg_length']:.1f} words")


def print_comparison_table(task1_results, task2_results):
    """Print a comparison table across all methods."""
    if not task1_results and not task2_results:
        print("\n  No results to display.")
        return

    lines = []
    lines.append(f"\n{'=' * 70}")
    lines.append("COMPARISON")
    lines.append(f"{'=' * 70}")

    if task1_results:
        lines.append(f"\n  {'Method':<20} {'F1(3-class)':<14} {'F1(2-class)':<14} {'CE':<10}")
        lines.append(f"  {'-' * 18:<20} {'-' * 12:<14} {'-' * 12:<14} {'-' * 8:<10}")
        for name, m in task1_results.items():
            lines.append(f"  {name:<20} {m['f1_3macro']:.4f}{'':<8} {m['f1_2macro']:.4f}{'':<8} {m['ce']:.4f}")

    if task2_results:
        lines.append(f"\n  {'Method':<20} {'LexF1':<10} {'Coverage':<10} {'AvgLen':<10}")
        lines.append(f"  {'-' * 18:<20} {'-' * 8:<10} {'-' * 8:<10} {'-' * 8:<10}")
        for name, m in task2_results.items():
            lines.append(f"  {name:<20} {m['lexical_f1']:.4f}{'':<6} {m['coverage_pct']:.1f}%{'':<5} {m['avg_length']:.1f}")

    lines.append(f"{'=' * 70}")
    print("\n".join(lines))


# ===== MAIN =====

def main():
    parser = argparse.ArgumentParser(description="Unified evaluation for MediaEval Task 1 & 2")
    parser.add_argument("--task", choices=["1", "2"], help="Evaluate only this task")
    parser.add_argument("--task1", help="Shorthand for --task 1")
    parser.add_argument("--task2", help="Shorthand for --task 2")
    parser.add_argument("--method", help="Evaluate only this method (e.g. tfidf, transformer, t5)")
    parser.add_argument("--all", action="store_true", help="Evaluate all available methods")
    args, _ = parser.parse_known_args()

    # Determine which tasks to run
    task1 = args.task == "1" or args.task1 or args.all
    task2 = args.task == "2" or args.task2 or args.all
    if not args.task and not args.task1 and not args.task2 and not args.all:
        task1, task2 = True, True

    # Load CSV for task2 evaluation
    data = config.load_data()

    # ---- Evaluate Task 1 ----
    task1_results = {}
    task1_reports = {}
    if task1:
        print("=" * 70)
        print("TASK 1: CLASSIFICATION EVALUATION")
        print("=" * 70)

        for method_key, (pred_file, eval_fn) in TASK1_METHODS.items():
            if args.method and args.method != method_key:
                continue

            preds_path = os.path.join(OUTPUT_DIR, pred_file)
            if not os.path.exists(preds_path):
                print(f"\n  [{method_key}] {pred_file} not found — skipping")
                continue

            print(f"\n--- {method_key} ---")
            try:
                report, metrics = evaluate_task1(preds_path)
                task1_reports[method_key] = report
                task1_results[method_key] = {
                    "f1_3macro": metrics["f1_3macro"],
                    "f1_2macro": metrics["f1_2macro"],
                    "ce": metrics["ce"],
                    "per_class": metrics["f1_each"],
                }
                print_classification_report(task1_results[method_key])
            except Exception as e:
                print(f"  ERROR: {e}")

        # Save combined task1 report
        if task1_reports:
            task1_report_path = os.path.join(OUTPUT_DIR, "evaluation_report_task1.json")
            with open(task1_report_path, "w") as f:
                json.dump(task1_reports, f, indent=2)
            print(f"\n  Task 1 reports saved to {task1_report_path}")

    # ---- Evaluate Task 2 ----
    task2_results = {}
    task2_reports = {}
    if task2:
        print("\n" + "=" * 70)
        print("TASK 2: GENERATION EVALUATION")
        print("=" * 70)

        for method_key, (pred_file, eval_fn) in TASK2_METHODS.items():
            if args.method and args.method != method_key:
                continue

            preds_path = os.path.join(OUTPUT_DIR, pred_file)
            if not os.path.exists(preds_path):
                print(f"\n  [{method_key}] {pred_file} not found — skipping")
                continue

            print(f"\n--- {method_key} ---")
            try:
                report, metrics = evaluate_task2(preds_path, data)
                task2_reports[method_key] = report
                task2_results[method_key] = metrics
                print_generation_report(metrics)
            except Exception as e:
                print(f"  ERROR: {e}")

        # Save combined task2 report
        if task2_reports:
            task2_report_path = os.path.join(OUTPUT_DIR, "evaluation_report_task2.json")
            with open(task2_report_path, "w") as f:
                json.dump(task2_reports, f, indent=2)
            print(f"\n  Task 2 reports saved to {task2_report_path}")

    # ---- Comparison table ----
    print_comparison_table(task1_results, task2_results)


if __name__ == "__main__":
    main()
