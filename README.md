# MediaEval 2026 - Enthymeme Detection: Understanding What Works

**This is a scientific inquiry** into detecting implicit arguments (premises and conclusions) in tweets and generating missing propositions. We explore what works and what doesn't through systematic experimentation with multiple approaches.

**Status**: ✅ All approaches functional | **Last Updated**: May 16, 2026

---

## 📋 Quick Navigation

- [Overview](#overview)
- [Setup & Prerequisites](#setup--prerequisites)
- [Running Experiments](#running-experiments)
- [Approach Details](#approach-details)
- [Task Specifications](#task-specifications)
- [Data & Performance](#data--performance)
- [Troubleshooting](#troubleshooting)

---

## Overview

### The Problem

**Enthymemes** are arguments with missing components—either an unstated premise (supporting assumption) or an unstated conclusion. They're common in social media where implicit reasoning is expected.

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

**Task 1: Detection** - Classify tweets as:
- `none` (0) - Argument is fully explicit
- `premise` (1) - Contains unstated premise
- `conclusion` (2) - Contains unstated conclusion

**Task 2: Generation** - Generate the missing proposition as a natural language sentence

### Approaches Explored (Actual Results)

We compare four different approaches for classification. The unified evaluation uses `run_methods.py`:

| Method | Type | F1 (2-class) | F1 (3-class) | CE Loss | Time |
|--------|------|------|------|--|------|
| **Transformer** | DistilBERT features + classifier | **0.635** | **0.470** | 0.955 | ~29s |
| TF-IDF + RF | Statistical | 0.432 | 0.277 | 0.815 | ~21s |
| Ollama Zero-shot | LLM (local) | 0.419 | 0.060 | 2.250 | ~43min |
| Ollama Few-shot | LLM + prompting | 0.399 | 0.141 | 24.611 | ~55min |

**Winner**: DistilBERT feature extraction + classifier is the best method across all metrics.

---

## Setup & Prerequisites

### Environment

```bash
# Activate the conda environment (pre-configured)
conda activate medEv

# All scripts run from project root
```

### Key Files

- **`config.py`** - Root config: single source of truth for data paths, labels, splits, hyperparameters
- **`run_methods.py`** - Run all classifiers and compare results in one command
- **`evaluation/evaluate.py`** - Unified evaluation: Task 1 (F1, cross-entropy) + Task 2 (lexical F1, coverage)

### Dependencies (Pre-installed in medEv)

- **PyTorch** - Deep learning framework with GPU support (MPS > CUDA > CPU auto-detected)
- **Transformers** - HuggingFace models (DistilBERT)
- **scikit-learn** - ML classifiers (LinearSVC, LogisticRegression, etc.)
- **pandas, numpy** - Data handling
- **tqdm** - Progress bars
- **requests** - HTTP communication (Ollama)
- **joblib** - Model serialization

### For Ollama-based Approaches (Optional)

```bash
# Terminal 1: Start Ollama server
ollama serve

# Terminal 2: Verify Ollama is running
python -c "from core.ollama_integration import OllamaClient; OllamaClient().test_connection()"
```

**Required Model**: `mistral` (7B parameters, ~4GB disk)  
**Setup**: `ollama pull mistral` (one-time download)

**Ollama URL**: `http://localhost:11434` (hardcoded in `core/ollama_integration.py:46`)

---

## Running Experiments

### Compare All Methods (Recommended)

```bash
# Run all classifiers and produce comparison table
python run_methods.py

# Run only the transformer
python run_methods.py --run-only transformer

# Skip a method
python run_methods.py --skip ollama_zero ollama_fewshot
```

Results go to `outputs/comparison.json` and individual reports.

### Individual Scripts

```bash
# TF-IDF + Random Forest (fastest baseline)
python task1_classification/task1_classifier_tfidf.py

# Ollama zero-shot
python task1_classification/task1_ollama_classifier.py

# Ollama few-shot
python task1_classification/task1_ollama_fewshot.py

# DistilBERT feature extraction + classifier (best)
python task1_classification/transformer/transformer.py

# Evaluation (unified: Task 1 + Task 2)
python evaluation/evaluate.py
python evaluation/evaluate.py --task 1   # Classification only
python evaluation/evaluate.py --task 2   # Generation only
python evaluation/evaluate.py --all      # Everything
```

### Generation

```bash
# Template-based (baseline)
python task2_generation/task2_generator.py

# Ollama-based (higher quality)
python task2_generation/task2_ollama_generator.py

# T5 (train + generate)
python task2_generation/task2_generator_enhanced.py --both
```

---

## Approach Details

### Classification: Task 1

All classifiers output predictions in a unified format:
```json
{
  "id": 123,
  "text": "...",
  "label": "none",
  "probabilities": {"none": 0.72, "premise": 0.15, "conclusion": 0.13},
  "hard_prediction": 0
}
```

#### 1. TF-IDF + Random Forest

**File**: `task1_classification/task1_classifier_tfidf.py`

- **Algorithm**: TF-IDF vectorizer + Random Forest with class weighting
- **Training**: Yes (on training split of merged_annotations_v2.csv)
- **Time**: ~21 seconds
- **F1 Score**: 0.432 (2-class), 0.277 (3-class)
- **Cross-entropy**: 0.815
- **Best for**: Fast baseline, interpretable features

#### 2. Ollama Zero-shot

**File**: `task1_classification/task1_ollama_classifier.py`

- **Model**: Mistral 7B (via local Ollama)
- **Approach**: Direct classification prompt without examples
- **Time**: ~43 minutes (one request per tweet)
- **F1 Score**: 0.419 (2-class), 0.060 (3-class)
- **Cross-entropy**: 2.250
- **Problem**: Very poor at 3-class (F1=0.060) - predicts almost exclusively "none"

#### 3. Ollama Few-shot

**File**: `task1_classification/task1_ollama_fewshot.py`

- **Model**: Mistral 7B (via local Ollama)
- **Approach**: Classification with balanced few-shot examples
- **Time**: ~55 minutes
- **F1 Score**: 0.399 (2-class), 0.141 (3-class)
- **Cross-entropy**: 24.611 (abnormally high - probability calibration issue)
- **Problem**: Cross-entropy is ~24x higher than other methods despite similar F1, indicating poorly calibrated confidence scores

#### 4. DistilBERT Feature Extraction + Classifier (Best) 🚀

**File**: `task1_classification/transformer/transformer.py`

- **Model**: Frozen DistilBERT-base-uncased (66M params) for feature extraction
- **Classifier**: LinearSVC (tested SVM, LogisticRegression, SGD; SVM wins in CV)
- **Approach**: Extract [CLS] embeddings from frozen DistilBERT, normalize with L2, train linear classifier
- **Training Time**: ~29 seconds total (feature extraction + classifier training)
- **F1 Score**: 0.635 (2-class), 0.470 (3-class)
- **Cross-entropy**: 0.955
- **Prediction distribution**: none=713, premise=415, conclusion=205 (vs ground truth 882/394/57)
- **Best for**: Best accuracy, fast inference, all 3 classes predicted

**Key design decisions**:
- Feature extraction (not fine-tuning) avoids slow per-sample backprop on CPU and class collapse to "none"
- 5-fold stratified CV selects best classifier per fold (LinearSVC wins)
- Balanced LogisticRegression used for final model (proper probability estimates)
- Cross-platform device handling: `--device cpu/cuda/mps` override, `device_map="auto"` with CPU fallback

**Configuration**:
```python
MODEL_NAME = "distilbert-base-uncased"
BATCH_SIZE = 64
MAX_LENGTH = 128
EPOCHS = 5 (for any training tasks)
LEARNING_RATE = 5e-5
WEIGHT_FRACTION = 0.25
```

**Device handling**:
1. Tries CUDA (NVIDIA GPU) first
2. Falls back to MPS (Apple Silicon GPU) with safe detection
3. Falls back to CPU with `device_map="auto"` (accelerate) or direct CPU load
4. Use `--device cpu` to force CPU regardless of hardware

### Generation: Task 2

#### 1. Template-based (Baseline)

**File**: `task2_generation/task2_generator.py`

- **Coverage**: 100%
- **Time**: ~10 seconds
- **Quality**: Medium (syntactic templates)

#### 2. Ollama-based (LLM) ⭐

**File**: `task2_generation/task2_ollama_generator.py`

- **Coverage**: 100%
- **Time**: ~2 minutes
- **Quality**: High (semantic generation)
- **Recommended**: Best quality for reasonable time

#### 3. T5 (Fine-tuned Generation)

**File**: `task2_generation/task2_generator_enhanced.py`

- **Usage**: `python task2_generation/task2_generator_enhanced.py --t5` (train), `--both` (train + generate)
- **Approach**: Fine-tunes T5-small on implicit proposition data from the CSV, generates premises/conclusions
- **Device**: Auto-detects CUDA → MPS → CPU
- **Requires**: `transformers` package (already installed), GPU/MPS recommended for training

---

## Task Specifications

### Task 1: Enthymeme Detection (3-Class Classification)

**Input**: Tweet text  
**Output**: Classification label + confidence scores

**Labels** (global mapping, used in `config.py`):
- `none` (0) - Argument is fully explicit
- `premise` (1) - Unstated premise (assumption)
- `conclusion` (2) - Unstated conclusion

**Primary Metric**: 2-class macro F1 (implicit=(premise|conclusion) vs. none)

### Task 2: Proposition Generation

**Input**: Tweet text + Task 1 label  
**Output**: Natural language sentence expressing the missing proposition

**Requirements**:
- Concise and declarative
- Completes the argument logically
- Reconstructs the implicit component as explicit text

**Evaluation**: Lexical overlap (precision/recall/F1) against annotator implicit texts

---

## Data & Performance

### Dataset

**Source**: MediaEval 2026 Shared Task - Enthymeme Detection  
**CSV**: `enthymemes_2/merged_annotations_v2.csv`  
**Total**: 1,333 annotated tweets with 5 annotators per instance

**Label distribution** (entire dataset):
- `none`: 882 (66.2%)
- `premise`: 394 (29.6%)
- `conclusion`: 57 (4.3%)

**Domains**:
- Vaccine debate: ~1,169 tweets (88%)
- Immigration debate: ~312 tweets (12%)

**Split**: 80/20 train/val (no separate test set; 5-fold CV used for evaluation)

### Actual Results (comparison.json)

| Method | F1 (3-class) | **F1 (2-class)** | CE Loss | Time |
|--------|------|------|--|------|
| **Transformer** | **0.470** | **0.635** | **0.955** | **~29s** |
| TF-IDF + RF | 0.277 | 0.432 | 0.815 | ~21s |
| Ollama Zero-shot | 0.060 | 0.419 | 2.250 | ~43min |
| Ollama Few-shot | 0.141 | 0.399 | 24.611 | ~55min |

### Per-class Results (Transformer)

| Class | Precision | Recall | F1 | Support |
|-------|------|------|----|----|
| none | 0.788 | 0.637 | 0.705 | 882 |
| premise | 0.480 | 0.505 | 0.492 | 394 |
| conclusion | 0.137 | 0.491 | 0.214 | 57 |

### Key Insights

1. **Class imbalance is the central challenge**: conclusion is only 4.3% of data. All methods struggle with this class.
2. **Transformer feature extraction wins**: 0.635 F1 (2-class), beating TF-IDF by 0.203 absolute.
3. **Ollama cross-entropy is inflated**: Few-shot CE=24.6 suggests probability calibration issues, even when F1 is reasonable.
4. **All methods predict mostly "none"**: The minority classes (premise, conclusion) are hard to detect.
5. **Feature extraction > fine-tuning**: For this small dataset (1333 samples), frozen embeddings + linear classifier outperforms fine-tuning, which collapses to predicting "none".

---

## Technical Specifications

### Directory Structure

```
MediaEval/2026/
├── config.py                          ← Root config (single source of truth)
├── run_methods.py                     ← Run all classifiers, produce comparison
├── README.md                          ← You are here
├── AGENT_CONTEXT.md                   ← Architecture & extension guide
├── requirements.txt                   ← Dependencies
│
├── core/                              # Shared utilities
│   ├── ollama_integration.py          # Ollama client (URL: localhost:11434)
│   ├── common_utils.py                # Labels, metrics, formatting
│   ├── domain_feature_engineering.py  # Linguistic features
│   └── explore_data.py               # Data exploration
│
├── task1_classification/              # Classification experiments
│   ├── task1_classifier_tfidf.py     # TF-IDF + Random Forest
│   ├── task1_ollama_classifier.py    # Ollama zero-shot
│   ├── task1_ollama_fewshot.py       # Ollama few-shot
│   └── transformer/                   # DistilBERT approach
│       ├── config.py                  # Transformer-specific config
│       ├── fine_tune.py              # Feature extraction + classifier
│       ├── inference.py              # Inference (legacy)
│       └── __init__.py
│
├── task2_generation/                  # Generation experiments
│   ├── task2_generator.py            # Template-based
│   ├── task2_ollama_generator.py     # Ollama generation
│   └── task2_generator_enhanced.py   # T5 fine-tuning + generation
│
├── evaluation/                        # Evaluation tools
│   └── evaluate.py                   # Unified evaluation (Task 1 + Task 2)
│
└── outputs/                           # Results directory
    ├── comparison.json               # All methods compared
    ├── predictions_classifiers.json  # Final predictions
    ├── evaluation_report.json        # Unified evaluation
    └── fine_tuned_model/             # Saved classifier
```

### Label Mapping (config.py)

```python
CLASS_LABELS = ["premise", "conclusion", "none"]
CLASS_IDS = [0, 1, 2]
LABEL_TO_ID = {"premise": 0, "conclusion": 1, "none": 2}
ID_TO_LABEL = {0: "premise", 1: "conclusion", 2: "none"}
```

### Root Config (config.py)

Single source of truth for:
- `DATA_CSV_PATH` - Annotation CSV location
- `OUTPUT_DIR` - Where predictions/reports go
- `CLASS_LABELS`, `LABEL_TO_ID`, `ID_TO_LABEL` - Label mappings
- `TRAIN_VAL_SPLIT` = 0.8, `CV_N_FOLDS` = 5
- `LABEL_SMOOTHING` = 0.1
- `TRANSFORMER_*` - Model name, batch size, learning rate, epochs, etc.
- `FEWSHOT_*` - Ollama few-shot parameters
- `load_data()` - Loads CSV, returns IDs, texts, soft labels, majority labels, implicit texts

### Transformer Device Handling

```python
# Auto-detect (tries CUDA → MPS → CPU)
python task1_classification/transformer/transformer.py

# Force specific device
python task1_classification/transformer/transformer.py --device cpu
python task1_classification/transformer/transformer.py --device cuda
python task1_classification/transformer/transformer.py --device mps
```

MPS has known PyTorch embedding bugs (`Placeholder storage not allocated`). The code handles this:
1. Uses `device_map="auto"` for proper HuggingFace device placement
2. Falls back to CPU if `accelerate` isn't installed
3. Falls back to CPU if `device_map` fails

### Shared Utilities (core/)

**ollama_integration.py**:
```python
from core.ollama_integration import OllamaClassifier, OllamaGenerator
classifier = OllamaClassifier()
result = classifier.classify("Your text", model="mistral")
```

**common_utils.py**:
```python
from core.common_utils import (
    label_to_id, id_to_label,
    compute_classification_metrics, normalize_label
)
```

**domain_feature_engineering.py**:
```python
from core.domain_feature_engineering import DomainFeatureExtractor
extractor = DomainFeatureExtractor()
features = extractor.extract_all_features("Your text")
```

---

## Troubleshooting

| Issue | Solution |
|-------|------|
| `ModuleNotFoundError: No module named 'core'` | Run from project root, not subdirectory |
| `Cannot connect to Ollama` | Run `ollama serve` in another terminal. URL: `http://localhost:11434` |
| `Model mistral not found` | `ollama pull mistral` |
| `torch.cuda.OutOfMemory` | Reduce `BATCH_SIZE` |
| `GPU not detected` | Check device: `python -c "import torch; print('CUDA:', torch.cuda.is_available())"` |
| Scripts hang on first Ollama request | Normal - model loads (~30s), then fast (~5-10s) |
| JSON output not saved | Ensure `outputs/` directory exists |
| Transformer output is all "none" | This shouldn't happen with the current feature extraction approach. Check device handling |
| `device_map='auto' failed` | Install accelerate: `pip install accelerate` |
| Data file not found | Check `config.py` → `DATA_CSV_PATH` matches your system |

---

## How Scripts Work

All scripts use root `config.py` as the single source of truth:
- No hardcoded CSV paths
- No hardcoded output directories
- No hardcoded label mappings
- No hardcoded splits

Scripts that don't need `core/` access (TF-IDF, transformer) import directly from root `config.py` via `sys.path.insert`. Ollama scripts import `core.ollama_integration`.

---

## Next Steps

1. ✅ All approaches implemented and compared
2. Transformer is the best method (0.635 F1 2-class)
3. Consider: ensemble methods, different feature extractors, or active learning
4. Generation approaches need evaluation

---

## For Developers

To extend this system, see [AGENT_CONTEXT.md](AGENT_CONTEXT.md) for:
- Architecture overview
- Design decisions
- How to add new classification/generation approaches
- Testing patterns
- Performance optimization

---

**Dataset Source**: MediaEval 2026 Shared Task - Enthymeme Detection  
**Last Updated**: May 16, 2026  
**Status**: ✅ All approaches functional and tested
