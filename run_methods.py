"""
Run all classification methods end-to-end and report comparison results.

Usage (from project root):
    python run_methods.py              # Run all methods
    python run_methods.py --skip ollama_zero  # Skip zero-shot
    python run_methods.py --run-only tfidf  # Run only TF-IDF

Each method writes to outputs/:
    predictions_classifiers_<METHOD>.json   (challenge predictions)
    evaluation_report_<METHOD>.json          (metrics for this method)

Comparison is saved to outputs/comparison.json at the end.
"""

import argparse
import json
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import torch

OUTPUT_DIR = config.OUTPUT_DIR


# -- Import classifier run functions (each exposes run() -> (predictions, report)) --

def _run_tfidf():
    from task1_classification.task1_classifier_tfidf import run_tfidf
    return run_tfidf()


def _run_ollama_zero():
    from task1_classification.task1_ollama_classifier import run_ollama_zero
    return run_ollama_zero()


def _run_ollama_fewshot():
    from task1_classification.task1_ollama_fewshot import run_ollama_fewshot
    return run_ollama_fewshot()


def _run_transformer():
    from task1_classification.transformer.transformer import run_transformer
    return run_transformer()


# ======== TASK 2 GENERATORS ========

def _build_examples(df, per_class=3):
    """Load implicit text examples for few-shot guidance."""
    premise_examples = []
    conclusion_examples = []
    for _, row in df.iterrows():
        imp = str(row.get("ann1_implicit", ""))
        if row["majority_label"] == "premise" and len(imp) > 3:
            premise_examples.append({"tweet": row["tweet_text"], "implicit": imp})
        elif row["majority_label"] == "conclusion" and len(imp) > 3:
            conclusion_examples.append({"tweet": row["tweet_text"], "implicit": imp})
    return premise_examples[:per_class], conclusion_examples[:per_class]


def _run_template():
    """Template-based proposition generation (Task 2)."""
    data = config.load_data()
    df = data["df"]
    premise_ex, conclusion_ex = _build_examples(df)

    propositions = []
    for i, row in df.iterrows():
        pid = int(row["id"])
        text = row["tweet_text"]
        majority = row.get("majority_label", "none")

        if majority == "premise":
            gen = "The tweet assumes that..." if not premise_ex else \
                  f"Similar to '{premise_ex[0]['tweet'][:40]}...', this tweet implies: {premise_ex[0]['implicit']}"
        elif majority == "conclusion":
            gen = "The tweet implies that..." if not conclusion_ex else \
                  f"Similar to '{conclusion_ex[0]['tweet'][:40]}...', this concludes: {conclusion_ex[0]['implicit']}"
        else:
            gen = ""

        propositions.append({
            "id": pid,
            "tweet_text": text,
            "predicted_label": majority,
            "confidence": 0.0,
            "generated_proposition": gen,
        })

    report = {
        "method": "template",
        "total_instances": len(propositions),
        "propositions_generated": sum(1 for p in propositions if p["generated_proposition"]),
        "coverage_pct": sum(1 for p in propositions if p["generated_proposition"]) / len(propositions) * 100,
        "metrics": {
            "lexical": {"precision": 0, "recall": 0, "f1": 0},
            "avg_length_words": 0,
        },
    }
    return propositions, report


def _run_ollama_gen():
    """Ollama (Mistral) proposition generation (Task 2)."""
    try:
        from core.ollama_integration import OllamaClient, OllamaGenerator
    except ImportError:
        print("  Ollama integration not available.")
        return [], {}

    client = OllamaClient()
    if not client.check_connection():
        print("  WARNING: Ollama not running. Skipping Ollama generation.")
        return [], {}

    data = config.load_data()
    texts = data["texts"]
    majority_labels = data["majority_labels"]

    generator = OllamaGenerator(model="mistral", client=client)
    # majority_labels are ints -> map to strings for Ollama
    label_strings = [config.ID_TO_LABEL.get(int(l), "none") for l in majority_labels]
    results = generator.generate_batch(texts, label_strings, show_progress=True)

    propositions = []
    for r in results:
        propositions.append({
            "id": int(r.get("id", 0)),
            "tweet_text": r.get("tweet", ""),
            "predicted_label": r.get("label", "none"),
            "confidence": 0.0,
            "generated_proposition": r.get("generated_proposition"),
        })

    report = {
        "method": "ollama_gen",
        "total_instances": len(propositions),
        "propositions_generated": sum(1 for p in propositions if p["generated_proposition"]),
        "coverage_pct": sum(1 for p in propositions if p["generated_proposition"]) / max(len(propositions), 1) * 100,
        "metrics": {
            "lexical": {"precision": 0, "recall": 0, "f1": 0},
            "avg_length_words": 0,
        },
    }
    return propositions, report


def _run_t5_gen():
    """T5 proposition generation (Task 2, enhanced)."""
    from task2_generation.task2_generator_enhanced import generate_with_t5
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    # T5 doesn't have reliable MPS support; force CPU for generation
    device = torch.device("cpu")
    print(f"  Using device: {device}")

    # Load or train T5
    model_path = os.path.join(OUTPUT_DIR, "task2_t5_model")
    if os.path.exists(os.path.join(model_path, "pytorch_model.bin")):
        print("  Loading pre-trained T5 model...")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)
    else:
        print("  T5 not trained yet. Using base T5-small (no training).")
        from transformers import T5ForConditionalGeneration
        model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

    tokenizer = AutoTokenizer.from_pretrained("t5-small")

    # Build predictions list from config
    data = config.load_data()
    df = data["df"]
    majority_labels = data["majority_labels"]

    predictions = []
    for i, row in df.iterrows():
        predictions.append({
            "id": int(row["id"]),
            "text": row["tweet_text"],
            "label": config.ID_TO_LABEL.get(int(majority_labels[i]), "none"),
            "hard_prediction": int(majority_labels[i]),
            "probabilities": {},
        })

    results = generate_with_t5(model, tokenizer, [p["text"] for p in predictions], predictions, device)

    report = {
        "method": "t5_gen",
        "total_instances": len(results),
        "propositions_generated": sum(1 for p in results if p["generated_proposition"]),
        "coverage_pct": sum(1 for p in results if p["generated_proposition"]) / max(len(results), 1) * 100,
        "metrics": {
            "lexical": {"precision": 0, "recall": 0, "f1": 0},
            "avg_length_words": 0,
        },
    }
    return results, report


# ======== TASK 2 EVALUATION ========

def _evaluate_task2(predictions, data):
    """Evaluate task2 generation predictions against annotator implicit texts."""
    # Build id -> annotator implicit texts
    id_to_ann = {}
    for i, imp_texts in enumerate(data["implicit_texts"]):
        if imp_texts:
            id_to_ann[data["ids"][i]] = imp_texts

    candidates = [g for g in predictions if g.get("generated_proposition")]
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
    coverage = len(candidates) / len(predictions) * 100

    report = {
        "method": predictions[0].get("method", "unknown") if predictions else "unknown",
        "total_instances": len(predictions),
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
            "premise": sum(1 for g in predictions if g.get("predicted_label") == "premise"),
            "conclusion": sum(1 for g in predictions if g.get("predicted_label") == "conclusion"),
            "none": sum(1 for g in predictions if g.get("predicted_label") == "none"),
        },
    }

    return report, {
        "lexical_f1": float(np.mean(f1_scores)) if f1_scores else 0,
        "coverage_pct": coverage,
        "avg_length": avg_length,
    }


# ======== TASK 1 METHODS ========
# Method name -> (display, run_fn)
TASK1_METHODS = {
    "tfidf": ("TF-IDF", _run_tfidf),
    "ollama_zero": ("Ollama Zero-Shot", _run_ollama_zero),
    "ollama_fewshot": ("Ollama Few-Shot", _run_ollama_fewshot),
    "transformer": ("Transformer (DistilBERT)", _run_transformer),
}

# ======== TASK 2 METHODS ========
# Method name -> (display, run_fn)
TASK2_METHODS = {
    "template": ("Template Generation", _run_template),
    "ollama_gen": ("Ollama Generation", _run_ollama_gen),
    "t5_gen": ("T5 Generation", _run_t5_gen),
}


def print_comparison(results):
    """Print a comparison table."""
    print(f"\n{'=' * 80}")
    print("COMPARISON")
    print(f"{'=' * 80}")

    rows = []
    for name, info in results.items():
        if info["success"] is False:
            rows.append([name, "FAIL", "FAIL", "FAIL", "FAIL"])
            continue
        r = info["metrics"]
        f1_3 = r.get("f1_3class", r.get("f1_macro_3class"))
        f1_2 = r.get("f1_2class", r.get("f1_macro_2class"))
        ce = r.get("cross_entropy")
        status = "OK"
        rows.append([
            name,
            f"{f1_3:.4f}" if f1_3 is not None else "N/A",
            f"{f1_2:.4f}" if f1_2 is not None else "N/A",
            f"{ce:.4f}" if ce is not None else "N/A",
            status,
        ])

    headers = ["Method", "F1 (3-class)", "F1 (2-class)", "Cross-Entropy", "Status"]
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    print("  " + "-".join("-" * w for w in widths))
    for row in rows:
        print(fmt.format(*row))


def print_task2_comparison(results):
    """Print Task 2 comparison table."""
    if not results:
        print("\n  No Task 2 results.")
        return
    print(f"\n{'=' * 70}")
    print("TASK 2: GENERATION COMPARISON")
    print(f"{'=' * 70}")
    print(f"  {'Method':<20} {'LexF1':<12} {'Coverage':<12} {'AvgLen':<12}")
    print(f"  {'-' * 18:<20} {'-' * 10:<12} {'-' * 10:<12} {'-' * 10:<12}")
    for name, info in results.items():
        if info["success"] is False:
            print(f"  {name:<20} {'FAIL':<10} {'N/A':<10} {'N/A':<10}")
            continue
        lex_f1 = info["metrics"].get("lexical_f1", 0)
        cov = info["metrics"].get("coverage_pct", 0)
        avg_len = info["metrics"].get("avg_length", 0)
        print(f"  {name:<20} {lex_f1:.4f}{'':<6} {cov:.1f}%{'':<6} {avg_len:.1f}")
    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(description="Run all MediaEval methods (Task 1 + Task 2)")
    parser.add_argument("--run-only", choices=list(TASK1_METHODS) + list(TASK2_METHODS), help="Run only one method")
    parser.add_argument("--skip", nargs="+", choices=list(TASK1_METHODS) + list(TASK2_METHODS), help="Skip these methods")
    parser.add_argument("--task1", action="store_true", help="Run Task 1 (classification) only")
    parser.add_argument("--task2", action="store_true", help="Run Task 2 (generation) only")
    parser.add_argument("--all", action="store_true", help="Run both tasks")
    args, _ = parser.parse_known_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    csv_path = config.DATA_CSV_PATH
    if not os.path.exists(csv_path):
        print(f"ERROR: Data file not found at {csv_path}")
        sys.exit(1)

    print(f"Data:   {csv_path}")
    print(f"Output: {OUTPUT_DIR}\n")

    # Determine which tasks to run
    do_task1 = args.task1 or not args.task2 and not args.all
    do_task2 = args.task2 or not args.task1 and not args.all
    if args.all or (not args.task1 and not args.task2 and not args.run_only):
        do_task1, do_task2 = True, True

    # Determine methods to run
    skip = set(args.skip or [])
    task1_methods = [m for m in TASK1_METHODS if m not in skip] if (do_task1 or args.run_only in TASK1_METHODS) else []
    task2_methods = [m for m in TASK2_METHODS if m not in skip] if (do_task2 or args.run_only in TASK2_METHODS) else []

    if args.run_only and args.run_only in TASK1_METHODS:
        task1_methods = [args.run_only]
        task2_methods = []
    elif args.run_only and args.run_only in TASK2_METHODS:
        task1_methods = []
        task2_methods = [args.run_only]

    print(f"Task 1 methods: {', '.join(task1_methods) if task1_methods else 'none'}")
    print(f"Task 2 methods: {', '.join(task2_methods) if task2_methods else 'none'}")

    total_start = time.time()
    task1_results = {}
    task2_results = {}
    task1_reports = {}
    task2_reports = {}

    # ===== TASK 1 =====
    if task1_methods:
        print(f"\n{'=' * 60}")
        print("TASK 1: CLASSIFICATION")
        print(f"{'=' * 60}")
        for method_key in task1_methods:
            display, run_fn = TASK1_METHODS[method_key]
            print(f"\nRUNNING: {display}")
            start = time.time()
            try:
                predictions, report = run_fn()
                elapsed = time.time() - start

                pred_path = os.path.join(OUTPUT_DIR, f"predictions_classifiers_{method_key}.json")
                with open(pred_path, "w") as f:
                    json.dump(predictions, f, indent=2)
                report_path = os.path.join(OUTPUT_DIR, f"evaluation_report_{method_key}.json")
                with open(report_path, "w") as f:
                    json.dump(report, f, indent=2)

                metrics = report.get("test_metrics", report.get("metrics", report.get("final_eval", {})))
                flat = {}
                for key in ("f1_macro_3class", "f1_macro_2class", "f1_3class", "f1_2class", "cross_entropy"):
                    for m in (report, metrics):
                        flat[key] = m.get(key)
                        if flat[key] is not None:
                            break
                metrics_flat = {
                    "f1_3class": flat.get("f1_macro_3class") or flat.get("f1_3class"),
                    "f1_2class": flat.get("f1_macro_2class") or flat.get("f1_2class"),
                    "cross_entropy": flat.get("cross_entropy"),
                }

                task1_results[method_key] = {"success": True, "predictions": predictions, "report": report, "metrics": metrics_flat, "elapsed": elapsed}
                task1_reports[method_key] = report
                print(f"\nSUCCESS: {display} ({elapsed:.1f}s)")
            except Exception as e:
                elapsed = time.time() - start
                import traceback
                traceback.print_exc()
                task1_results[method_key] = {"success": False, "predictions": None, "report": None, "metrics": {}, "elapsed": elapsed}
                print(f"\nFAILED: {display} ({elapsed:.1f}s) — {e}")

        # Print comparison
        print_comparison(task1_results)

        # Save comparison
        comp = {}
        for name, info in task1_results.items():
            comp[name] = {
                "display": TASK1_METHODS[name][0],
                "f1_3class": info["metrics"].get("f1_3class"),
                "f1_2class": info["metrics"].get("f1_2class"),
                "cross_entropy": info["metrics"].get("cross_entropy"),
                "success": info["success"],
                "elapsed_s": info["elapsed"],
            }
        comp_path = os.path.join(OUTPUT_DIR, "comparison_task1.json")
        with open(comp_path, "w") as f:
            json.dump(comp, f, indent=2)
        print(f"\nComparison saved to {comp_path}")

    # ===== TASK 2 =====
    if task2_methods:
        data = config.load_data()
        print(f"\n{'=' * 60}")
        print("TASK 2: GENERATION")
        print(f"{'=' * 60}")
        for method_key in task2_methods:
            display, run_fn = TASK2_METHODS[method_key]
            print(f"\nRUNNING: {display}")
            start = time.time()
            try:
                propositions, gen_report = run_fn()
                elapsed = time.time() - start

                if not propositions:
                    task2_results[method_key] = {"success": False, "predictions": [], "report": {}, "metrics": {}, "elapsed": elapsed, "coverage_pct": 0, "avg_length": 0}
                    print(f"\nFAILED: {display} — no propositions generated")
                    continue

                # Evaluate
                report, metrics = _evaluate_task2(propositions, data)
                task2_results[method_key] = {
                    "success": True,
                    "predictions": propositions,
                    "report": report,
                    "metrics": metrics,
                    "elapsed": elapsed,
                    "coverage_pct": report.get("coverage_pct", 0),
                    "avg_length": report["metrics"]["avg_length_words"],
                }
                task2_reports[method_key] = report

                pred_path = os.path.join(OUTPUT_DIR, f"task2_predictions_{method_key}.json")
                with open(pred_path, "w") as f:
                    json.dump(propositions, f, indent=2)
                report_path = os.path.join(OUTPUT_DIR, f"task2_evaluation_{method_key}.json")
                with open(report_path, "w") as f:
                    json.dump(report, f, indent=2)

                lex = report["metrics"]["lexical"]
                print(f"\nSUCCESS: {display} ({elapsed:.1f}s) — LexF1={lex['f1']:.4f} Coverage={report['coverage_pct']:.1f}%")
            except Exception as e:
                elapsed = time.time() - start
                import traceback
                traceback.print_exc()
                task2_results[method_key] = {"success": False, "predictions": [], "report": {}, "metrics": {}, "elapsed": elapsed, "coverage_pct": 0, "avg_length": 0}
                print(f"\nFAILED: {display} ({elapsed:.1f}s) — {e}")

        print_task2_comparison(task2_results)

        # Save comparison
        comp = {}
        for name, info in task2_results.items():
            comp[name] = {
                "display": TASK2_METHODS[name][0],
                "lexical_f1": info["metrics"].get("lexical_f1"),
                "coverage_pct": info.get("coverage_pct"),
                "avg_length": info.get("avg_length"),
                "success": info["success"],
                "elapsed_s": info["elapsed"],
            }
        comp_path = os.path.join(OUTPUT_DIR, "comparison_task2.json")
        with open(comp_path, "w") as f:
            json.dump(comp, f, indent=2)
        print(f"\nComparison saved to {comp_path}")

    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
