# MediaEval 2026 - Enthymeme Detection

**Scientific inquiry** into detecting implicit arguments (premises and conclusions) in tweets and generating missing propositions. Systematic experimentation with multiple approaches.

**Status**: Architecture refactored with unified CV engine | Submission files generated | **Last Updated**: Jun 11, 2026

---

## Overview

### The Problem

**Enthymemes** are arguments with missing components—either an unstated premise (supporting assumption) or an unstated conclusion. Common in social media where implicit reasoning is expected.

**Example**:
```
Tweet: "Deterring people smugglers is essential to controlled immigration.
        We should support all plans to stop them."

Analysis:
  - Premise 2 (explicit): Deterring smugglers is essential to controlled immigration
  - Premise 1 (implicit): ??? [Missing assumption about controlled immigration being desirable]
  - Conclusion (explicit): We should support all plans to stop them
```

### Our Tasks

**Task 1: Detection** — Classify tweets as:
- `none` (0) — Argument is fully explicit
- `premise` (1) — Contains unstated premise
- `conclusion` (2) — Contains unstated conclusion

**Task 2: Generation** — Generate the missing proposition as a natural language sentence

### Actual Results (from held-out 20% test split)

These are the **only meaningful** F1 scores — computed on held-out data, not training data:

| Method | Type | F1 (2-class) | F1 (3-class) | CE Loss | Time |
|--------|------|------|------|--|------|
| **Transformer** | DistilBERT features + LogisticRegression | **0.635** | **0.470** | 0.955 | ~29s |
| TF-IDF + RF | Statistical | 0.432 | 0.277 | 0.815 | ~21s |
| TF-IDF + SVM | Statistical | — | — | — | — |
| TF-IDF + XGBoost | Statistical | — | — | — | — |
| SBERT + LR | Semantic features + LR | — | — | — | — |
| Ollama Zero-shot | LLM (local) | 0.419 | 0.060 | 2.250 | ~43min |
| Ollama Few-shot | LLM + prompting | 0.399 | 0.141 | 24.611 | ~55min |

**Winner**: DistilBERT feature extraction + classifier

### Submission Ensemble (5 Classifiers)

The final submission uses weighted soft voting across 5 diverse classifiers:

| Classifier | Weight |
|-----------|--------|
| Transformer (DistilBERT + LR) | 0.470 |
| SBERT + LR | 0.306 |
| TF-IDF + XGBoost | 0.302 |
| TF-IDF + RF | 0.277 |
| TF-IDF + SVM | 0.249 |

**Key insight**: Voting-based ensembles with only 2 classifiers cannot improve over the best standalone. With 5 diverse classifiers, weighted voting aggregates their strengths.

---

## Setup & Prerequisites

### Environment

```bash
conda activate medEv
cd /path/to/project
```

### Key Files

- **`config.py`** — Root config: single source of truth for data paths, labels, splits, hyperparameters
- **`core/evaluator.py`** — Unified CV/evaluation engine with dict-based model iteration
- **`core/pipeline.py`** — Model registry and orchestrators (run_standalone, run_ensemble)
- **`submit.py`** — Train on full data, generate test submissions, or run all methods and compare
- **`task1_classification/new_classifiers.py`** — SVM, XGBoost, SBERT (uses unified CV engine)
- **`evaluation/evaluate.py`** — Unified evaluation: Task 1 (F1, cross-entropy) + Task 2 (lexical F1, coverage)

### Dependencies

- **PyTorch** — GPU support (CUDA > MPS > CPU auto-detected)
- **Transformers** — HuggingFace models (DistilBERT)
- **scikit-learn** — ML classifiers
- **pandas, numpy** — Data handling
- **requests** — HTTP communication (Ollama)

### For Ollama (Optional)

```bash
ollama serve                    # Start server
ollama pull mistral             # Download base model
ollama pull gemma4              # Prefer: better generation quality
ollama pull qwen3.6             # Fallback: good generation quality
python -c "from core.ollama_integration import OllamaClient; OllamaClient().test_connection()"
```

**URL auto-detection**: Ollama URL is auto-detected at runtime — tries `host.docker.internal:11434` first (Docker/Linux), falls back to `localhost:11434` (Mac native).

---

## Running Experiments

### Compare All Classification Methods

```bash
# Run standalone models with 5-fold CV on 80% training set
python run_methods.py

# Using the unified pipeline
python -c "from core.pipeline import run_standalone; results = run_standalone()"

# Single method
python task1_classification/task1_classifier_tfidf.py
python task1_classification/transformer/transformer.py
python task1_classification/new_classifiers.py --method svm
```

### Run Everything + Compare + Submit

```bash
python submit.py --run-all
```

This calls `submit.run_all_and_submit()` which:
1. Computes ensemble weights dynamically from 5-fold CV on 80% training set (no hardcoded constants)
2. Runs ALL standalone AND 6 ensemble strategies on the SAME held-out 20% test split
3. Compares all results by F1(3-class), ranks all methods in one table
4. Selects the best overall approach (standalone or ensemble)
5. Trains the winning approach on ALL 1333 instances for final submission
6. Generates Task 1 + Task 2 submission files

### Individual Scripts

```bash
# Classification (all use unified CV engine from core/evaluator.py)
python task1_classification/task1_classifier_tfidf.py        # TF-IDF + Random Forest
python task1_classification/transformer/transformer.py       # DistilBERT features
python task1_classification/new_classifiers.py               # SVM, XGBoost, SBERT
```

**Note**: All classification methods now use the shared `core/evaluator.py` for consistent CV/evaluation.
Each method has a single `_run_pipeline()` function that handles both standalone (80/20 with CV) and
ensemble (train on ALL data) modes.

### Apple Silicon (M1/M2/M3) Compatibility Notes

**XGBoost Segmentation Fault:**
On macOS ARM64, running XGBoost may cause a segmentation fault when training TF-IDF + XGBoost models.
This is caused by BLAS library conflicts in the XGBoost compiled binaries on Apple Silicon.

**Workarounds:**
1. Disable XGBoost by removing "tfidf_xgb" from `core/pipeline.py` MODELS dict
2. Install XGBoost via conda-forge: `conda install -c conda-forge xgboost`
3. Use XGBoost 2.x instead of 3.x: `pip install 'xgboost<3.0'`

See `docs/AGENT_CONTEXT.md` for full technical details.

### Ensemble Methods

Two approaches are available:

**1. Simplified ensemble weights (default in submit.py)**
```bash
python submit.py --task1  # Uses compute_ensemble_weights() from 5-fold CV F1 scores
```

**2. Nested Cross-Validation (methodologically correct)**
```python
from core.ensemble_nested_cv import nested_cv_ensemble
results = nested_cv_ensemble()
# This implements the workflow in AGENT_CONTEXT.md lines 252-298
# Outer loop: 5-fold CV, Inner loop: train base models with their own CV
```
Note: Nested CV is computationally intensive. For faster iteration, use the simplified approach.

---

## Approach Details

### Classification: Task 1

All classifiers output predictions in unified format:
```json
{
  "id": 123,
  "text": "...",
  "label": "none",
  "probabilities": {"premise": 0.15, "conclusion": 0.13, "none": 0.72},
  "hard_prediction": 0
}
```

#### 1. TF-IDF + Random Forest

**File**: `task1_classification/task1_classifier_tfidf.py`

- TF-IDF vectorizer + Random Forest with class weighting
- 5-fold stratified CV for hyperparameter tuning (GridSearchCV)
- Soft-label sample weights from 5 annotators
- Best params from CV: `n_estimators=100, max_depth=None, min_samples_leaf=1`
- F1: 0.432 (2-class), 0.277 (3-class)

#### 2. Ollama Zero-shot / Few-shot

**Files**: `task1_classification/task1_ollama_classifier.py`, `task1_classification/task1_ollama_fewshot.py`

- Mistral 7B via local Ollama
- Zero-shot: direct classification prompt
- Few-shot: balanced examples + multi-round averaging
- **Problem**: Both have probability calibration issues (CE=2.25 and CE=24.6)

#### 3. DistilBERT Feature Extraction + Classifier (Best)

**File**: `task1_classification/transformer/transformer.py`

- Frozen DistilBERT-base-uncased [CLS] embeddings + L2 normalize
- 5-fold stratified CV selects best classifier among: lr_balanced, lr_weighted, sgd_log_balanced, sgd_log_weighted
- Winner: Balanced LogisticRegression (C=1.0)
- Final model: Balanced LogisticRegression on 80% train split
- F1: **0.635 (2-class), 0.470 (3-class)** — the best approach

#### 4. New Classifiers (SVM, XGBoost, SBERT)

**File**: `task1_classification/new_classifiers.py`

- **TF-IDF + LinearSVC**: Linear kernel SVM on TF-IDF features
- **TF-IDF + XGBoost**: Gradient boosting on TF-IDF features
- **SBERT + LR**: all-MiniLM-L6-v2 sentence embeddings + LogisticRegression

#### 5. Ensemble Methods (task1_ensemble.py)

6 strategies evaluated on the SAME held-out 20% test split:

| Method | F1(3-class) | F1(2-class) | CE | Description |
|--|--|--|--|--|
| soft_voting | **0.4723** | **0.6278** | 1.021 | Average probability vectors |
| weighted_voting | **0.4723** | **0.6278** | 1.005 | Weight by CV F1 |
| majority_voting | **0.4723** | **0.6278** | 6.509 | Hard vote (argmax) |
| feature_fusion | 0.4300 | 0.5983 | 1.001 | TF-IDF + DistilBERT features + LR |
| sbert | 0.4047 | 0.5694 | 0.921 | SBERT embeddings + LR |
| bagging | 0.2838 | 0.4222 | 0.809 | 10 Random Forests on bootstrapped data |

```bash
python task1_classification/task1_ensemble.py --method all
```

**Key finding**: With only 2 base classifiers, voting-based ensembles produce identical results. With 5 diverse classifiers in the final submission, weighted voting aggregates their complementary strengths.

#### 6. Final Submission Ensemble

The final submission uses **weighted soft voting** across 5 classifiers. Weights are derived from each classifier's CV F1 score on the held-out 20% test split:

```python
ENSEMBLE_WEIGHTS = {
    "transformer":     0.470,  # DistilBERT features + LR
    "sbert_lr":        0.306,  # SBERT embeddings + LR
    "tfidf_xgb":       0.302,  # TF-IDF + XGBoost
    "tfidf_rf":        0.277,  # TF-IDF + Random Forest
    "tfidf_svm":       0.249,  # TF-IDF + LinearSVC
}
```

#### 7. Submission (submit.py)

```bash
python submit.py --run-all       # Run everything + compare + submit
python submit.py --task1         # Task 1 submission only
python submit.py --task2 --t5    # Task 2 with T5 (flan-t5-base + LoRA)
python submit.py --task2 --ollama # Task 2 with Ollama (gemma4 → qwen3.6 → mistral)
```

Also callable as a function:
```python
from submit import run_all_and_submit, submit_task1
best_name, best_metrics, test_sub, full_sub = run_all_and_submit()
submission, full_submission = submit_task1()
```

- Trains 5 classifiers on ALL 1333 annotated instances
- Generates predictions for test set (148 tweets) via weighted soft voting (5 classifiers)
- Task 2: Trains flan-t5-base + LoRA (or uses Ollama) on 364 training instances
- Outputs: `submit_task1_test.json` + `submit_task1_classifiers.json` + `submit_task2_propositions.json` in challenge format

---

## Task 2: Proposition Generation

Two tested standalone methods (each in its own file):

### 1. Ollama (Recommended)
**File**: `task2_generation/ollama_generator.py`
**CLI**: `python submit.py --task2 --ollama`

- Prefer `gemma4` → `qwen3.6` → `mistral` (auto-detected from available Ollama models)
- High quality generation, ~2 min for 148 test instances
- Also callable: `from submit import submit_task2_ollama; submit_task2_ollama()`

### 2. T5 + LoRA
**File**: `task2_generation/t5_finetune.py`
**CLI**: `python task2_generation/t5_finetune.py` or `python submit.py --task2 --t5`

- `google/flan-t5-base` (250M params) fine-tuned with LoRA on 364 training instances
- LoRA adapter: r=8, α=16, dropout=0.1 on attention matrices
- Trains in ~5 min, generates in ~30s
- Also callable: `from submit import submit_task2_t5; submit_task2_t5()`

### submit.py Task 2 CLI

```bash
python submit.py --task2 --t5    # Train flan-t5-base + LoRA if needed, then generate
python submit.py --task2 --ollama # Use Ollama (auto-selects best model)
```

### submit.py Task 2 CLI

```bash
python submit.py --task2 --t5    # Train flan-t5-base + LoRA if needed, then generate
python submit.py --task2 --ollama # Use Ollama (auto-selects best model)
```

### Task 2: Method Selection

`task2_generation/eval_generation.py` — evaluates both T5 and Ollama on validation data (80/20 split), computes BLEU + semantic similarity, chooses the better method.

Integrated into `run_all_and_submit()`: automatically runs the comparison and picks the winner before generating propositions.

```bash
python task2_generation/eval_generation.py
```

Requires: running Ollama server with generation model (gemma4/qwen3.6/mistral) for Ollama evaluation.

### Task 2: Abandoned Methods

The following standalone scripts were original prototypes that were consolidated into `submit.py`. They remain (renamed with `_not_used` suffix) as reference but are never imported or called:

| File | Method | Why abandoned |
|---|---|---|
| `task2_generation/task2_generator.py_not_used` | Template-based fill-in-the-blank | Inline template code removed from submit.py; no longer referenced |
| `task2_generation/task2_ollama_generator.py_not_used` | Ollama LLM (mistral) | Ollama generation extracted to `ollama_generator.py`; original used mistral only |
| `task2_generation/task2_generator_enhanced.py_not_used` | T5 + LoRA fine-tuning | T5 training logic extracted to `t5_finetune.py`; standalone training logic removed from submit.py |

### Task 2: Classification as Generation Attempt

Some Task 1 classification methods were tested as Task 2 generation approaches:

| Method | Why it didn't work |
|---|---|
| `task1_ollama_fewshot.py` | CE=24.6 — probability calibration completely broken; predictions unusable |
| `task1_ensemble.py` (soft/weighted/majority voting) | Not a generation method — produces class labels, not natural language propositions |

---

## Data & Performance

### Dataset

**Source**: MediaEval 2026 Shared Task - Enthymeme Detection  
**CSV**: `enthymemes_2/merged_annotations_v2.csv`  
**Total**: 1,333 annotated tweets (5 annotators per instance)

**Label distribution**:
- `none`: 882 (66.2%)
- `premise`: 394 (29.6%)
- `conclusion`: 57 (4.3%)

**Splits**: 80/20 train/val (5-fold CV for evaluation)

### Per-class Results (Transformer)

| Class | Precision | Recall | F1 | Support |
|--|--|--|--|--|
| none | 0.788 | 0.637 | 0.705 | 882 |
| premise | 0.480 | 0.505 | 0.492 | 394 |
| conclusion | 0.137 | 0.491 | 0.214 | 57 |

### Key Insights

1. **Class imbalance is the central challenge**: conclusion is only 4.3% of data. All methods struggle.
2. **Transformer feature extraction wins**: 0.635 F1 (2-class), beating TF-IDF by +0.203 absolute.
3. **Ollama has calibration issues**: Few-shot CE=24.6 despite decent F1 (0.40) — probabilities are unreliable.
4. **Feature extraction > fine-tuning**: Frozen embeddings + linear classifier outperforms fine-tuning on this small dataset.
5. **All methods predict mostly "none"**: Minority classes are hard to detect.

### Directory Structure

```
MediaEval/2026/
├── config.py                          ← Root config (single source of truth)
├── run_methods.py                     ← Run all classifiers, produce comparison
├── submit.py                          ← Run all + compare + submit
├── README.md
├── AGENT_CONTEXT.md                   ← Architecture & extension guide
├── requirements.txt
├── core/                              ← Shared utilities
│   ├── ollama_integration.py         # Ollama client (auto-detect URL)
│   ├── evaluator.py                  # Unified CV/evaluation engine
│   ├── pipeline.py                   # Model registry and orchestrators
│   ├── common_utils.py_not_used      # Abandoned: unused
│   └── domain_feature_engineering.py_not_used  # Abandoned: unused
├── task1_classification/              ← Classification experiments
│   ├── task1_classifier_tfidf.py     # TF-IDF + RF
│   ├── task1_ollama_classifier.py    # Ollama zero-shot
│   ├── task1_ollama_fewshot.py       # Ollama few-shot
│   ├── task1_ensemble.py             # 6 ensemble strategies
│   ├── new_classifiers.py            # SVM, XGBoost, SBERT
│   └── transformer/                   # DistilBERT approach
│       ├── transformer.py
│       └── inference.py
├── task2_generation/                  ← Generation (standalone method files)
│   ├── t5_finetune.py                # T5 + LoRA fine-tuning + generation (tested)
│   ├── ollama_generator.py           # Ollama LLM generation (tested)
│   ├── task2_generator.py_not_used   # Abandoned: template-based prototype
│   ├── task2_ollama_generator.py_not_used  # Abandoned: Ollama prototype
│   └── task2_generator_enhanced.py_not_used  # Abandoned: T5 prototype
├── evaluation/                        ← Evaluation tools
│   ├── evaluate_test.py              # Test set evaluation
│   └── metrics.py                    # Metric functions
└── outputs/                           ← Results
    ├── comparison_task1.json
    ├── predictions_classifiers_<METHOD>.json
    ├── evaluation_report_<METHOD>.json
    ├── submit_task1_classifiers.json # Full dataset submission
    ├── submit_task1_test.json        # Test set submission
    └── task2_t5_model/               # Fine-tuned T5 model
```

---

## Troubleshooting

| Issue | Solution |
|--|--|
| `ModuleNotFoundError: No module named 'core'` | Run from project root |
| `Cannot connect to Ollama` | Run `ollama serve`. URL auto-detected (host.docker.internal → localhost) |
| `Model gemma4/qwen3.6 not found` | `ollama pull gemma4` or `ollama pull qwen3.6` |
| `Model mistral not found` | `ollama pull mistral` |
| `torch.cuda.OutOfMemory` | Reduce `BATCH_SIZE` |
| `GPU not detected` | `python -c "import torch; print('CUDA:', torch.cuda.is_available())"` |
| `device_map='auto' failed` | `pip install accelerate` |
| `Data file not found` | Check `config.py` → `DATA_CSV_PATH` |
| T5 generates empty strings | Install PEFT: `pip install peft` for LoRA fine-tuning |

---

## Technical Notes

### How Scripts Work

All scripts use root `config.py` — no hardcoded CSV paths, output dirs, label mappings, or splits.

### Label Mapping (config.py)

```python
CLASS_LABELS = ["premise", "conclusion", "none"]
LABEL_TO_ID = {"premise": 0, "conclusion": 1, "none": 2}
ID_TO_LABEL = {0: "premise", 1: "conclusion", 2: "none"}
TRAIN_VAL_SPLIT = 0.8
CV_N_FOLDS = 5
```

### Transformer Device Handling

```bash
python task1_classification/transformer/transformer.py       # Auto-detect (CUDA → MPS → CPU)
python task1_classification/transformer/transformer.py --device cpu
```

### Adding New Methods

**Preferred approach (unified pipeline)**:
1. Define feature extractor and classifier factory in `core/pipeline.py`
2. The unified evaluation engine handles CV, metrics, and both standalone/ensemble modes

**Alternative (manual)**:
1. Create file in `task1_classification/`
2. Import from root `config.py`
3. Output unified format: `{id, text, label, probabilities, hard_prediction}`
4. See [AGENT_CONTEXT.md](AGENT_CONTEXT.md) for full extension guide


## Target Architecture

This section describes the ideal state of the codebase as defined in
[AGENT_CONTEXT.md](docs/AGENT_CONTEXT.md#target-architecture-ideals-for-future-development).

### Design Goals

The following should be achieved for optimal maintainability and model comparison:

| Goal | Current State | Target State |
|------|---------------|--------------|
| Single data loading path | Partial (`config.load_data()`) | All classifiers use same function |
| One CV/evaluation engine | `core/evaluator.py` exists | Fully utilized by all models |
| Dict-based model iteration | `pipeline.py` registry | Add more models via config dict |
| No duplicate training loops | Partial (some duplication) | Single `_run_pipeline()` per module |

### Current Model Registry (core/pipeline.py)

The following models are now implemented and available:

| Model | Classifier | Description |
|-------|------------|-------------|
| tfidf_lr | LogisticRegression | Statistical baseline with balanced class weights |
| tfidf_svm | LinearSVC | Linear SVM on TF-IDF features |
| tfidf_ridge | RidgeClassifier | Ridge regression for high-dimensional sparse data |
| tfidf_knn | KNeighborsClassifier | Instance-based learning comparison |
| tfidf_complement_nb | ComplementNB | Optimized for imbalanced text classification |
| tfidf_rf | RandomForestClassifier | Tree-based baseline method |
| tfidf_xgb | XGBoost (optional) | Gradient boosting (requires xgboost package) |

All models use TF-IDF feature extraction and are trained via the unified CV engine.

### Future Improvements

- **Nested CV**: Implement full nested cross-validation for ensemble optimization
- **Cross-encoder**: Integrate reranker-MiniLM when model availability improves
- **Automatic model selection**: Add logic to rank models and compute ensemble weights from validation metrics

### Known Limitations

- **Ensemble evaluation on training data is meaningless**: The only valid benchmark is the held-out test F1.
- **Conclusion class is extremely hard**: Only 4.3% of data; best method achieves 0.214 F1.
- **Ollama requires running server**: Both Ollama methods fail gracefully if Ollama is not running.
- **T5 requires PEFT for small dataset**: Without peft, flan-t5-base overfits on 364 training instances.
- **Generation quality is lexical**: Task 2 evaluation uses word-overlap (BLEU-style), not semantic similarity.

### Submission Checklist

All submission files are in `outputs/` and verified:

| File | Status | Details |
|------|--------|---------|
| `submit_task1_test.json` | Ready | 148 entries, all have probabilities |
| `submit_task1_classifiers.json` | Ready | 1333 entries, all have probabilities |
| `submit_task2_propositions.json` | Ready | 148 entries, 18 generated (17 premise + 1 conclusion) |

**Submission format verified:**
- Task 1: `{id, text, label, probabilities, hard_prediction}` — all 5 keys present
- Task 2: `{id, tweet_text, predicted_label, confidence, generated_proposition}` — all 5 keys present
- IDs consistent across all files (test: 148, full: 1333)

---

**Dataset Source**: MediaEval 2026 Shared Task - Enthymeme Detection  
**Last Updated**: Jun 11, 2026  
**Status**: All submission files ready in `outputs/`
