# AGENT_CONTEXT.md - Architecture & Extension Guide

**Audience**: Developers, AI agents, automation tools enhancing this codebase
**Last Updated**: May 27, 2026
**Status**: Classification pipeline complete | Submission generated

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
- Outputs: `outputs/comparison_task1.json` + per-method reports

### `submit.py` - Master Runner & Submission Generator

**`submit.run_all_and_submit()`** - Single function that:
1. Runs all base classifiers (TF-IDF, Transformer, Ollama if available)
2. Runs all 6 ensemble strategies (soft_voting, weighted_voting, majority_voting, feature_fusion, sbert, bagging)
3. Compares all results by F1(3-class) on held-out 20% test split
4. Selects the best method
5. Generates submission files (test set + full dataset) in challenge format

**`submit.submit_task1()`** - Trains TF-IDF + Transformer on all 1333 instances, generates predictions for test set via weighted soft voting ensemble.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│  Root config.py (single source of truth)                  │
│  submit.run_all_and_submit() (master runner)              │
├─────────────────────────────────────────────────────────────┤
│  User Scripts (task1_*, task2_*)                          │
│  Entry points: Single responsibility per file             │
├─────────────────────────────────────────────────────────────┤
│  Core Modules (core/)                                     │
│  - ollama_integration.py (Ollama client, URL: localhost:11434)│
│  - common_utils.py (labels, metrics)                      │
│  - domain_feature_engineering.py (linguistic features)    │
│  - explore_data.py (data exploration)                     │
├─────────────────────────────────────────────────────────────┤
│  External Dependencies                                    │
│  - PyTorch (device: cuda > mps > cpu)                    │
│  - Transformers (DistilBERT frozen features)             │
│  - scikit-learn (LogisticRegression, RF, etc.)           │
│  - Ollama (local LLM, optional)                          │
└─────────────────────────────────────────────────────────────┘
```

### Module Dependencies

```
task1_classification/*.py  →  config.py  →  External packages
task2_generation/*.py      →  config.py  →  External packages
transformer/*.py           →  config.py  →  External packages
evaluation/*.py            →  config.py  →  External packages
core/*.py                  →  External packages
submit.py                  →  config.py + individual classifier modules  →  External packages
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
- `load_training_data()` - Thin wrapper returning df with convenience columns

### `core/ollama_integration.py`

**Purpose**: Ollama client interface
**URL**: `http://localhost:11434`
**Classes**: `OllamaClient(model="mistral")`, `OllamaClassifier`, `OllamaGenerator`

### `core/common_utils.py`

**Purpose**: Shared utilities for labels, metrics, formatting
**Key exports**: `label_to_id`, `id_to_label`, `normalize_label`, `compute_classification_metrics`, `save_predictions`

### `core/domain_feature_engineering.py`

**Purpose**: Linguistic feature extraction
**Class**: `DomainFeatureExtractor` — 50+ features (word count, punctuation, capitalization, argument indicators)

### `evaluation/evaluate.py`

**Purpose**: Unified evaluation of Task 1 (classification) and Task 2 (generation)
**Task 1 Computes**: 2-class macro F1, 3-class macro F1, cross-entropy loss, per-class precision/recall/F1
**Task 2 Computes**: Lexical overlap (precision/recall/F1), coverage, avg length
**Output**: `outputs/evaluation_report_task1.json` + `outputs/evaluation_report_task2.json`

---

## Task-Specific Modules

### `task1_classification/` - Classification

**Best approach**: DistilBERT feature extraction + Balanced LogisticRegression

| File | Type | F1 (2-class) | F1 (3-class) | CE Loss | Time |
|------|------|------|------|--|------|
| `transformer/transformer.py` | DistilBERT features + LR | **0.635** | **0.470** | 0.955 | ~29s |
| `task1_classifier_tfidf.py` | TF-IDF + RF | 0.432 | 0.277 | 0.815 | ~21s |
| `task1_ollama_classifier.py` | Mistral 7B (Ollama) | 0.419 | 0.060 | 2.250 | ~43min |
| `task1_ollama_fewshot.py` | Mistral 7B + few-shot | 0.399 | 0.141 | 24.611 | ~55min |

**Winner**: Transformer (DistilBERT feature extraction + Balanced LogisticRegression)

**Note on ensemble scores**: Earlier runs reported ensemble F1 of ~0.99. These were evaluated on training data and are not meaningful benchmarks. The only valid F1 scores are from the held-out 20% test split shown above.

**Ensemble methods** (`task1_classification/task1_ensemble.py`):
6 strategies, all evaluated on the SAME held-out 20% test split as standalone methods.

| Method | F1(3-class) | F1(2-class) | CE | Time |
|--|--|--|--|--|
| soft_voting | **0.4723** | **0.6278** | 1.021 | ~61s |
| weighted_voting | **0.4723** | **0.6278** | 1.005 | ~56s |
| majority_voting | **0.4723** | **0.6278** | 6.509 | ~55s |
| feature_fusion | 0.4300 | 0.5983 | 1.001 | ~93s |
| sbert | 0.4047 | 0.5694 | 0.921 | ~62s |
| bagging | 0.2838 | 0.4222 | 0.809 | ~55s |

**Important**: The top 3 voting methods are identical because with only 2 base classifiers, voting-based ensembles cannot add diversity. The ensemble F1 matches the Transformer alone (0.4723), confirming that ensemble methods require more diverse base classifiers to improve over standalone approaches.

Usage: `python task1_classification/task1_ensemble.py --method all`

#### `transformer/transformer.py`

DistilBERT feature extraction + classifier (not fine-tuning):
1. Freeze DistilBERT, extract [CLS] embeddings from all tweets
2. L2-normalize embeddings
3. 5-fold stratified CV: test lr_balanced, lr_weighted, sgd_log_balanced, sgd_log_weighted
4. Winner: Balanced LogisticRegression (C=1.0)
5. Final model: Balanced LogisticRegression on 80% train split

**Key design**: Feature extraction avoids class collapse (fine-tuning predicts only "none" on this imbalanced dataset) and runs in ~29s.

**Classifier interface**:
- `_build_classifier(name, class_weight=None)` — single source of truth for classifier construction
- `run_cv(X, y, texts)` — uses `_build_classifier()` to evaluate candidates
- `run_transformer()` — main callable, returns (predictions, report)
- `run_transformer_for_ensemble()` — runs CV + trains on ALL data, returns (predictions, report, artifacts)
- `get_full_data_predictions()` — trains on ALL data using CV-selected classifier

**Device handling**: `--device cpu/cuda/mps` override, `device_map="auto"` with CPU fallback

#### `task1_classifier_tfidf.py`

TF-IDF vectorizer + Random Forest with class weights.
- 5-fold stratified CV via GridSearchCV for hyperparameter tuning
- Soft-label sample weights from 5 annotators
- Best params from CV: `n_estimators=100, max_depth=None, min_samples_leaf=1`

**Interface**:
- `_run_pipeline(use_full_data=False)` — returns (predictions, report, artifacts)
- `run_tfidf()` — main callable
- `run_tfidf_for_ensemble()` — trains on ALL data, returns (predictions, report, artifacts)
- `get_full_data_predictions()` — trains on ALL data using CV-selected params

#### `task1_ollama_classifier.py`

Direct Mistral 7B prompt classification.

#### `task1_ollama_fewshot.py`

Balanced few-shot examples with multi-round averaging.

### `task2_generation/` - Generation

| File | Type | Quality | Time |
|------|------|--|------|
| `task2_ollama_generator.py` | LLM prompting | High | ~2min |
| `task2_generator.py` | Templates | Medium | ~10s |
| `task2_generator_enhanced.py` | T5 fine-tuning | Train: ~5min, Gen: ~30s |

### `submit.py` - Submission

**`submit_task1()`** — Trains TF-IDF + Transformer on ALL 1333 instances, generates predictions for test set via weighted soft voting (0.28 TF-IDF + 0.47 Transformer).

**`run_all_and_submit()`** — Master function that runs everything:
1. All base classifiers via `run_methods.py`'s TASK1_METHODS
2. All 6 ensemble strategies via `task1_ensemble.py`'s METHODS
3. Compares by F1(3-class) on held-out test split
4. Selects best method
5. Generates `submit_task1_test.json` (148 test tweets) + `submit_task1_classifiers.json` (1333 full dataset)

**CLI**: `python submit.py --run-all`, `--task1`, `--task2 --t5`, `--task2 --ollama`

---

## Data Flow

### Classification Pipeline

```
CSV data (config.load_data)
    ↓
[Transformer] Extract [CLS] embeddings → L2 normalize → Balanced LogisticRegression (CV-selected)
[TF-IDF] TF-IDF vectorize → Random Forest (GridSearchCV-selected)
[Ollama] Prompt Mistral 7B (with/without few-shot)
    ↓
Unified output: {id, text, label, probabilities, hard_prediction}
    ↓
outputs/predictions_classifiers_<METHOD>.json
outputs/evaluation_report_<METHOD>.json
outputs/comparison_task1.json (via run_methods.py)
```

### Transformer-Specific Flow

```
CSV → config.load_data() → IDs, texts, soft labels (5 annotators)
    ↓
Frozen DistilBERT → extract_features() → [CLS] embeddings (1333 x 768)
    ↓
L2 normalize
    ↓
5-fold CV: Balanced LogisticRegression > SGD-log_loss (by F1)
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

Single command to run all classifiers and produce comparison. Handles different report formats (TF-IDF uses `test_metrics`, transformer uses `final_eval`).

### Why Unified Output Format?

All classifiers produce `{id, text, label, probabilities, hard_prediction}` for easy comparison. `evaluation/evaluate.py` matches predictions against ground truth and computes all metrics in one pass.

---

## Results

### All Classifiers (on held-out 20% test split)

| Method | F1 (3-class) | F1 (2-class) | CE Loss | Time |
|--|--|--|--|--|
| **Transformer** | **0.470** | **0.635** | **0.955** | **~29s** |
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

### Ensemble Results (6 strategies, evaluated on held-out 20% test split)

| Method | F1 (3-class) | F1 (2-class) | CE | Time |
|--|--|--|--|--|
| soft_voting | 0.4723 | 0.6278 | 34.539 | ~50s |
| weighted_voting | 0.4723 | 0.6278 | 34.539 | ~45s |
| majority_voting | 0.4723 | 0.6278 | 34.539 | ~43s |
| feature_fusion | 0.4300 | 0.5983 | 34.539 | ~82s |
| sbert | 0.4047 | 0.5694 | 34.539 | ~52s |
| bagging | 0.2838 | 0.4222 | 34.539 | ~48s |

**Key finding**: Voting-based ensembles (soft_voting, weighted_voting, majority_voting) produce identical results because with only 2 base classifiers, there is no diversity to exploit. The ensemble F1(3-class) = 0.4723 matches the Transformer alone (0.470), confirming that ensembles require more diverse base classifiers to improve over standalone approaches.

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
- [x] submit.py for test set submissions
- [x] run_all_and_submit() master runner
- [x] Submission files generated
- [x] Ensemble evaluation on held-out 20% split (fair comparison)
- [x] All 6 ensemble methods working correctly
- [x] Base classifier caching in ensemble (no redundant re-runs)

### Known Limitations
- [ ] Only 2 base classifiers available without Ollama (TF-IDF + Transformer)
- [ ] Voting ensembles cannot improve over Transformer with only 2 classifiers
- [ ] No CV-on-all-data ensemble selection strategy implemented
- [ ] High CE (34.54) across all methods indicates probability calibration issues

### Future
- [ ] Proper ensemble selection via CV on all 1333 instances
- [ ] Add more diverse base classifiers for ensemble
- [ ] Fix feature_fusion dtype bug
- [ ] Active learning for data selection
- [ ] Generation evaluation (BERTScore, human)
- [ ] T5 fine-tuning with more epochs/larger model

---

## Extension Support

When adding new approaches or features:

1. **Use root config** — Don't duplicate label/metric/path constants
2. **Unified output** — Match `{id, text, label, probabilities, hard_prediction}` format
3. **Add to run_methods.py** — Include in `TASK1_METHODS` dict for comparison
4. **Add to submit.py** — Include in `submit_task1()` for submission
5. **Add to task1_ensemble.py** — Include in `METHODS` dict for ensemble comparison
6. **Cross-platform device** — Use `get_device()` if using PyTorch
7. **Return artifacts** — `_run_pipeline()` returns `(predictions, report, artifacts)` where artifacts contains the trained model for external reuse (ensemble, test-set prediction)

---

**Document Purpose**: Enable agents and developers to understand system architecture, design patterns, and how to safely extend functionality
**Last Verified**: May 27, 2026
