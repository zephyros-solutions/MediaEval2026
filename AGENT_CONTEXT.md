# AGENT_CONTEXT.md - Architecture & Extension Guide

This document describes the codebase architecture, design decisions, and extension patterns for agents working to improve or extend this project.

**Audience**: Developers, AI agents, and automation tools enhancing this codebase  
**Last Updated**: May 16, 2026  
**Status**: Complete and tested

---

## Architecture Overview

### Root Config (`config.py`) - Single Source of Truth

All configuration flows from the root `config.py`:
- Data paths, output directories, label mappings
- Train/val splits, CV folds
- Transformer hyperparameters, few-shot parameters
- `load_data()` helper that reads CSV and returns IDs, texts, soft labels, majority labels, implicit texts

### `run_methods.py` - Unified Runner

Runs all classification methods and produces comparison table.
- `--run-only <method>` - Run single method
- `--skip <methods>` - Skip methods
- Outputs: `outputs/comparison.json` + per-method reports
- Child stdout/stderr flows directly to terminal (no buffering)
- Uses `final_eval` key in transformer reports (fallback in comparison builder)

### Architecture Diagram

```
┌────────────────────────────────────────────────────────────┐
│  Root config.py (single source of truth)                  │
│  run_methods.py (unified runner)                          │
├────────────────────────────────────────────────────────────┤
│  User Scripts (task1_*, task2_*)                          │
│  Entry points: Single responsibility per file             │
├────────────────────────────────────────────────────────────┤
│  Core Modules (core/)                                     │
│  - ollama_integration.py (Ollama client, URL: localhost:11434)    │
│  - common_utils.py (labels, metrics)                      │
│  - domain_feature_engineering.py (linguistic features)    │
│  - explore_data.py (data exploration)                     │
├────────────────────────────────────────────────────────────┤
│  External Dependencies                                    │
│  - PyTorch (device: cuda > mps > cpu)                    │
│  - Transformers (DistilBERT frozen features)             │
│  - scikit-learn (LinearSVC, LogisticRegression, etc.)    │
│  - Ollama (local LLM)                                    │
└────────────────────────────────────────────────────────────┘
```

### Module Dependencies

```
task1_classification/*.py  →  config.py  →  External packages
task2_generation/*.py      →  config.py  →  External packages
transformer/*.py           →  config.py  →  External packages
evaluation/*.py            →  config.py  →  External packages
core/*.py                  →  External packages
```

No circular dependencies.

---

## Module Responsibilities

### `config.py` (Root)

**Purpose**: Central configuration and data loading  
**Key exports**:
- `DATA_CSV_PATH` - Annotation CSV location
- `OUTPUT_DIR` - Output directory
- `CLASS_LABELS = ["premise", "conclusion", "none"]`
- `LABEL_TO_ID`, `ID_TO_LABEL` - Bidirectional label mapping
- `TRAIN_VAL_SPLIT = 0.8`, `CV_N_FOLDS = 5`
- `LABEL_SMOOTHING = 0.1`
- `TRANSFORMER_MODEL_NAME`, `TRANSFORMER_BATCH_SIZE`, etc.
- `FEWSHOT_*` - Ollama few-shot parameters
- `load_data()` - Reads CSV, returns `{ids, texts, ann_labels, majority_labels, implicit_texts, df}`

### `core/ollama_integration.py`

**Purpose**: Ollama client interface  
**URL**: `http://localhost:11434` (hardcoded in `OllamaConfig.BASE_URL`)  
**Classes**:
- `OllamaClient(model="mistral")` - Connection management, health check
- `OllamaClassifier` - Classification interface
- `OllamaGenerator` - Generation interface
- `test_connection()` - Verify Ollama is running

**Design**: Request retry logic (max 3 attempts), JSON response parsing, error handling.

### `core/common_utils.py`

**Purpose**: Shared utilities for labels, metrics, formatting  
**Key exports**:
- `label_to_id(label)`, `id_to_label(id)` - Label conversion
- `normalize_label(label)` - Handles variations ("no_premise" → "none")
- `compute_classification_metrics(y_true, y_pred)` - F1, accuracy, precision, recall
- `save_predictions(predictions, filename)` - JSON serialization

### `core/domain_feature_engineering.py`

**Purpose**: Linguistic feature extraction  
**Class**: `DomainFeatureExtractor`  
**Features**: 50+ features (word count, punctuation, capitalization, argument indicators)

### `evaluation/evaluate.py`

**Purpose**: Unified evaluation of both Task 1 (classification) and Task 2 (generation)  
**Task 1 Input**: `outputs/predictions_classifiers_<METHOD>.json`  
**Task 2 Input**: `outputs/task2_generated_propositions*.json`  
**Task 1 Computes**: 2-class macro F1, 3-class macro F1, cross-entropy loss, per-class precision/recall/F1  
**Task 2 Computes**: Lexical overlap (precision/recall/F1), coverage, avg length  
**Output**: `outputs/evaluation_report_task1.json` + `outputs/evaluation_report_task2.json`

---

## Task-Specific Modules

### `task1_classification/` - Classification

**Design**: Each approach in separate file for independent execution and comparison.

| File | Type | Model | F1 (2-class) | F1 (3-class) | CE Loss | Time |
|------|------|-------|----|----|----|--|
| `transformer/transformer.py` | Feature extraction + classifier | Frozen DistilBERT + LinearSVC | **0.635** | **0.470** | 0.955 | ~29s |
| `task1_classifier_tfidf.py` | Train+predict | TF-IDF + Random Forest | 0.432 | 0.277 | 0.815 | ~21s |
| `task1_ollama_classifier.py` | Inference | Mistral 7B (Ollama) | 0.419 | 0.060 | 2.250 | ~43min |
| `task1_ollama_fewshot.py` | Inference | Mistral 7B + examples (Ollama) | 0.399 | 0.141 | 24.611 | ~55min |

**Winner**: Transformer (DistilBERT feature extraction + classifier)

#### `transformer/transformer.py`

Now uses **DistilBERT feature extraction + classifier** (not full fine-tuning):
1. Freeze DistilBERT, extract [CLS] embeddings from all tweets
2. L2-normalize embeddings
3. 5-fold stratified CV: test SVM (balanced/weighted/fine-tuned), LogisticRegression (balanced/weighted), SGD (custom/balanced)
4. LinearSVC wins in CV
5. Balanced LogisticRegression used for final model (proper probability estimates)
6. Output: predictions with calibrated probabilities

**Key design**: Feature extraction avoids class collapse (full fine-tuning predicts only "none" on this imbalanced dataset) and runs in ~29s vs 30-60min for fine-tuning.

**Device handling**:
```bash
python task1_classification/transformer/transformer.py       # Auto-detect (CUDA → MPS → CPU)
python task1_classification/transformer/transformer.py --device cpu  # Force CPU
```

#### `task1_classifier_tfidf.py`

TF-IDF vectorizer + Random Forest with class weights. Trained on 80/20 train/val split.

#### `task1_ollama_classifier.py`

Direct Mistral 7B prompt classification. Struggles without examples (F1=0.060 on 3-class).

#### `task1_ollama_fewshot.py`

Balanced few-shot examples with multi-round averaging. Still has probability calibration issues (CE=24.6).

### `task2_generation/` - Generation

| File | Type | Quality | Time |
|------|------|----|--|
| `task2_ollama_generator.py` | LLM prompting | High | ~2min |
| `task2_generator.py` | Templates | Medium | ~10s |
| `task2_generator_enhanced.py` | T5 fine-tuning + generation | Train: ~5min, Gen: ~30s | ~3min |

---

## Data Flow

### Classification Pipeline

```
CSV data (config.load_data)
    ↓
[Transformer] Extract [CLS] embeddings → L2 normalize → LinearSVC
[TF-IDF] TF-IDF vectorize → Random Forest
[Ollama] Prompt Mistral 7B (with/without few-shot)
    ↓
Unified output: {id, text, label, probabilities, hard_prediction}
    ↓
outputs/predictions_classifiers.json
outputs/evaluation_report.json
outputs/comparison.json (via run_methods.py)
```

### Transformer-Specific Flow

```
CSV → config.load_data() → IDs, texts, soft labels (5 annotators)
    ↓
Frozen DistilBERT → extract_features() → [CLS] embeddings (1333 x 768)
    ↓
L2 normalize
    ↓
5-fold CV: LinearSVC > LogisticRegression > SGD (by F1)
    ↓
Final: Balanced LogisticRegression on 80% train split
    ↓
Predictions on full 1333 samples
```

---

## Design Decisions

### Why Feature Extraction Instead of Fine-Tuning?

Full fine-tuning of DistilBERT on this small dataset (1333 samples, 4.3% conclusion) collapses to predicting only "none". Feature extraction avoids this:
- Frozen embeddings capture semantics without overfitting
- Linear classifier on L2-normalized features is fast (~29s)
- Outperforms both TF-IDF and full fine-tuning on F1

### Why Root `config.py`?

Eliminates all hardcoded paths, labels, and splits across the entire codebase. Every script imports from a single source.

### Why `run_methods.py`?

Single command to run all classifiers and produce comparison. Child output flows directly to terminal (no buffering). Handles different report formats (TF-IDF uses `test_metrics`, transformer uses `final_eval`).

### Why Unified Output Format?

All classifiers produce `{id, text, label, probabilities, hard_prediction}` for easy comparison and evaluation. `evaluation/evaluate.py` matches predictions against ground truth and computes all metrics in one pass.

### Why No Abstract Base Classes?

Each approach has a fundamentally different interface. Direct implementation is simpler and more Pythonic than forcing a common contract.

---

## How to Add a New Classification Approach

### 1. Create file in `task1_classification/`

```bash
touch task1_classification/task1_classifier_newmethod.py
```

### 2. Use root config for everything

```python
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

data = config.load_data()
df = data["df"]
texts = data["texts"]
labels = data["majority_labels"]
```

### 3. Output in unified format

```python
predictions = []
for i, row in enumerate(df):
    predictions.append({
        "id": int(row["id"]),
        "text": row["text"],
        "label": config.ID_TO_LABEL[pred_id],
        "probabilities": {label: float(prob) for label, prob in probs.items()},
        "hard_prediction": pred_id,
    })

with open(config.OUTPUT_DIR + "/predictions_classifiers.json", "w") as f:
    json.dump(predictions, f, indent=2)
```

### 4. Add to `run_methods.py` METHODS dict

```python
METHODS = {
    ...
    "newmethod": ("New Method",
                  "task1_classification/task1_classifier_newmethod.py",
                  "newmethod", "newmethod"),
}
```

### 5. Test

```bash
python run_methods.py --run-only newmethod
```

---

## Device Handling

Transformer supports cross-platform device handling:

```python
# Auto-detect: CUDA → MPS → CPU
device, name = detect_device()

# Or force:
python transformer.py --device cuda
python transformer.py --device mps
python transformer.py --device cpu
```

**MPS caveat**: PyTorch has known bugs with MPS embedding layers. The code handles this with:
1. `device_map="auto"` (accelerate) for proper HuggingFace placement
2. CPU fallback if accelerate isn't installed
3. CPU fallback if `device_map` fails
4. `--device cpu` to force CPU regardless

---

## Current Results

### All Classifiers (`outputs/comparison.json`)

| Method | F1 (3-class) | F1 (2-class) | CE Loss | Time |
|--|--|--|--|--|
| **Transformer** | 0.470 | **0.635** | 0.955 | ~29s |
| TF-IDF + RF | 0.277 | 0.432 | 0.815 | ~21s |
| Ollama Zero-shot | 0.060 | 0.419 | 2.250 | ~43min |
| Ollama Few-shot | 0.141 | 0.399 | 24.611 | ~55min |

### Transformer Per-Class (Best Method)

| Class | Precision | Recall | F1 | Support |
|--|--|--|--|--|
| none | 0.788 | 0.637 | 0.705 | 882 |
| premise | 0.480 | 0.505 | 0.492 | 394 |
| conclusion | 0.137 | 0.491 | 0.214 | 57 |

**Key insight**: Conclusion class (4.3% of data) is hardest for all methods. Transformer improves over TF-IDF by ~0.20 absolute F1 (2-class).

### Cross-Entropy Issues

- Ollama few-shot has abnormally high CE (24.6) despite decent F1 (0.40), indicating probability calibration problems
- Transformer uses balanced LogisticRegression for calibrated probabilities (CE=0.96)
- TF-IDF has competitive CE (0.82) with lower F1

---

## Debugging

### Check Configuration
```bash
python -c "import config; print(config.DATA_CSV_PATH)"
python -c "import config; print(config.LABEL_TO_ID)"
python -c "from task1_classification.transformer.config import print_config; print_config()"
```

### Check Device
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('MPS:', hasattr(torch.backends, 'mps') and torch.backends.mps.is_available())"
```

### Check Ollama
```bash
python -c "from core.ollama_integration import OllamaClient; OllamaClient().test_connection()"
```

### Check Data
```bash
python -c "import config; d = config.load_data(); print(d['df'].shape); print(d['majority_labels'].value_counts())"
```

### Check Output Format
```bash
python -c "import json; p = json.load(open('outputs/predictions_classifiers.json')); print(json.dumps(p[0], indent=2))"
```

---

## Roadmap

### Completed
- [x] All classifiers implemented and compared
- [x] Transformer is the best method (0.635 F1 2-class)
- [x] Unified output format across all methods
- [x] Cross-platform device handling
- [x] run_methods.py unified runner
- [x] Root config.py as single source of truth
- [x] evaluation/evaluate.py for unified evaluation (Task 1 + Task 2)

### Future
- [ ] Ensemble approach (combine all methods)
- [ ] Active learning for data selection
- [ ] Generation evaluation (BERTScore, human)
- [x] T5 fine-tuning for generation
- [ ] Deploy as REST API

---

## Extension Support

When adding new approaches or features:

1. **Use root config** - Don't duplicate label/metric/path constants
2. **Unified output** - Match `{id, text, label, probabilities, hard_prediction}` format
3. **Add to run_methods.py** - Include in comparison table
4. **Test with run_methods.py** - Run `python run_methods.py --run-only <name>`
5. **Cross-platform device** - Use `detect_device()` if using PyTorch
6. **Direct terminal output** - Don't use `capture_output=True` in subprocess (child inherits stdout/stderr)

---

**Document Purpose**: Enable agents and developers to understand system architecture, design patterns, and how to safely extend functionality  
**Last Verified**: May 16, 2026 | **All Systems**: ✅ Operational
