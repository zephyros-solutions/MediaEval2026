# MediaEval 2026 - Enthymeme Detection

**Scientific inquiry** into detecting implicit arguments (premises and conclusions) in tweets and generating missing propositions. Systematic experimentation with multiple approaches.

**Status**: Classification pipeline complete | Submission files generated | **Last Updated**: May 27, 2026

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
| Ollama Zero-shot | LLM (local) | 0.419 | 0.060 | 2.250 | ~43min |
| Ollama Few-shot | LLM + prompting | 0.399 | 0.141 | 24.611 | ~55min |

**Winner**: DistilBERT feature extraction + classifier

### Important Note on Ensemble Scores

The only valid benchmark is the Transformer's **0.470 F1(3-class)** on the held-out 20% split. The ensemble's best F1(3-class) is **0.472** — matching the Transformer because voting-based ensembles with only 2 base classifiers cannot add diversity. Ensembles require more diverse base classifiers to improve over standalone approaches.

---

## Setup & Prerequisites

### Environment

```bash
conda activate medEv
cd /path/to/project
```

### Key Files

- **`config.py`** — Root config: single source of truth for data paths, labels, splits, hyperparameters
- **`run_methods.py`** — Run all classifiers and compare results
- **`submit.py`** — Train on full data, generate test submissions, or run all methods and compare
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
ollama pull mistral             # Download model
python -c "from core.ollama_integration import OllamaClient; OllamaClient().test_connection()"
```

---

## Running Experiments

### Compare All Classification Methods

```bash
python run_methods.py                          # Run all methods
python run_methods.py --run-only transformer   # Single method
python run_methods.py --skip ollama_zero       # Skip unavailable methods
```

### Run Everything + Compare + Submit

```bash
python submit.py --run-all
```

This calls `submit.run_all_and_submit()` which:
1. Runs all base classifiers (TF-IDF, Transformer, Ollama methods if available)
2. Runs all 6 ensemble strategies (soft_voting, weighted_voting, majority_voting, feature_fusion, sbert, bagging)
3. Compares all results by F1(3-class) on the held-out 20% test split
4. Selects the best method
5. Generates submission files (test + full dataset)

### Individual Scripts

```bash
# Classification
python task1_classification/task1_classifier_tfidf.py
python task1_classification/transformer/transformer.py
python task1_classification/task1_ollama_classifier.py
python task1_classification/task1_ollama_fewshot.py
python task1_classification/task1_ensemble.py --method all   # Run all 6 ensemble strategies

# Generation
python task2_generation/task2_generator.py           # Template-based
python task2_generation/task2_ollama_generator.py    # Ollama
python task2_generation/task2_generator_enhanced.py --both   # T5 train + generate
```

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

#### 4. Ensemble Methods

**File**: `task1_classification/task1_ensemble.py`

6 strategies evaluated on the SAME held-out 20% test split as standalone methods:

| Method | F1(3-class) | F1(2-class) | CE | Description |
|--|--|--|--|--|
| soft_voting | **0.4723** | **0.6278** | 1.021 | Average probability vectors |
| weighted_voting | **0.4723** | **0.6278** | 1.005 | Weight by CV F1 (Transformer ~0.47, TF-IDF ~0.28) |
| majority_voting | **0.4723** | **0.6278** | 6.509 | Hard vote (argmax) |
| feature_fusion | 0.4300 | 0.5983 | 1.001 | TF-IDF + DistilBERT features concatenated + LogisticRegression |
| sbert | 0.4047 | 0.5694 | 0.921 | SBERT embeddings (all-MiniLM-L6-v2) + LogisticRegression |
| bagging | 0.2838 | 0.4222 | 0.809 | 10 Random Forest models on bootstrapped data |

```bash
python task1_classification/task1_ensemble.py --method soft_voting
python task1_classification/task1_ensemble.py --method all
```

**Key finding**: Voting-based ensembles (soft_voting, weighted_voting, majority_voting) produce identical results because with only 2 base classifiers there is no diversity to exploit. The ensemble F1 matches the Transformer alone, confirming that ensembles require more diverse base classifiers to improve over standalone approaches.

#### 5. Submission

**File**: `submit.py`

```bash
python submit.py --run-all       # Run everything + compare + submit
python submit.py --task1         # Task 1 submission only
python submit.py --task2 --t5    # Task 2 with T5
python submit.py --task2 --ollama # Task 2 with Ollama
```

Also callable as a function:
```python
from submit import run_all_and_submit, submit_task1
best_name, best_metrics, test_sub, full_sub = run_all_and_submit()
submission, full_submission = submit_task1()
```

- Trains TF-IDF + Transformer on ALL 1333 annotated instances
- Generates predictions for test set (148 tweets) via weighted soft voting ensemble
- Outputs: `submit_task1_test.json` + `submit_task1_classifiers.json` in challenge format

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
├── task1_classification/              ← Classification experiments
│   ├── task1_classifier_tfidf.py     # TF-IDF + RF
│   ├── task1_ollama_classifier.py    # Ollama zero-shot
│   ├── task1_ollama_fewshot.py       # Ollama few-shot
│   ├── task1_ensemble.py             # 6 ensemble strategies
│   └── transformer/                   # DistilBERT approach
│       ├── config.py
│       ├── transformer.py
│       └── __init__.py
├── task2_generation/                  ← Generation experiments
│   ├── task2_generator.py            # Template-based
│   ├── task2_ollama_generator.py     # Ollama generation
│   └── task2_generator_enhanced.py   # T5 fine-tuning + generation
├── evaluation/                        ← Evaluation tools
│   └── evaluate.py                   # Unified evaluation
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
| `Cannot connect to Ollama` | Run `ollama serve`. URL: `http://localhost:11434` |
| `Model mistral not found` | `ollama pull mistral` |
| `torch.cuda.OutOfMemory` | Reduce `BATCH_SIZE` |
| `GPU not detected` | `python -c "import torch; print('CUDA:', torch.cuda.is_available())"` |
| `device_map='auto' failed` | `pip install accelerate` |
| `Data file not found` | Check `config.py` → `DATA_CSV_PATH` |

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

1. Create file in `task1_classification/`
2. Import from root `config.py`
3. Output unified format: `{id, text, label, probabilities, hard_prediction}`
4. Add to `TASK1_METHODS` in `run_methods.py` and `METHODS` in `task1_ensemble.py`
5. See [AGENT_CONTEXT.md](AGENT_CONTEXT.md) for full extension guide

### Known Limitations

- **Only 2 available base classifiers without Ollama**: TF-IDF + Transformer. Ensemble diversity is limited.
- **Ensemble evaluation on training data is meaningless**: The only valid benchmark is the held-out test F1.
- **Conclusion class is extremely hard**: Only 4.3% of data; best method achieves 0.214 F1.
- **Ollama requires running server**: Both Ollama methods fail gracefully if Ollama is not running.

---

**Dataset Source**: MediaEval 2026 Shared Task - Enthymeme Detection  
**Last Updated**: May 27, 2026  
**Status**: ✅ Classification pipeline complete | Submission files generated in `outputs/`
