# AGENT_CONTEXT.md - Architecture & Extension Guide

**Audience**: Developers, AI agents, automation tools enhancing this codebase
**Last Updated**: Jun 12, 2026


## DONE: Completed Work

### Code Structure Improvements (Completed)

#### 1. Shared CV/Evaluation Infrastructure
Created `core/evaluator.py` and `core/pipeline.py`:
- One data loading path: All classifiers use `config.load_data()`
- One evaluation engine: `run_model_config()` with dict-based model iteration
- Uniform result format across all models

#### 2. Unified Training Pipeline
Each classifier module now has a single `_run_pipeline()` function that handles both:
- Standalone mode (80/20 split with CV)
- Ensemble mode (train on ALL data)

Files updated:
- `task1_classifier_tfidf.py`: Uses `_run_pipeline(use_full_data=True/False)`
- `new_classifiers.py`: Each classifier has one implementation

#### 3. XGBoost Fixes
- Removed deprecated `use_label_encoder=False` parameter (removed in XGBoost 3.x)
- No longer imports unused `cross_encoder` module

**macOS ARM64 Segmentation Fault Issue**
- **Problem**: XGBoost on Apple Silicon M1/M2/M3 crashes with segmentation fault when training TF-IDF + XGBoost models
- **Root Cause**: BLAS library conflicts in XGBoost's compiled binaries on macOS ARM64
- **Status**: NOT FULLY RESOLVED - The code has attempted fixes but the issue persists

  **Workarounds available:**
  1. Install XGBoost via conda-forge instead of pip:
     ```bash
     conda install -c conda-forge xgboost
     ```
  2. Use an older stable version (XGBoost 2.x):
     ```bash
     pip install 'xgboost<3.0'
     ```
  3. Disable XGBoost by removing "tfidf_xgb" from `core/pipeline.py` MODELS dict

#### 4. Transformer device_map Fix
- Changed `AutoModel.from_pretrained(..., device_map="auto")` to explicit `.to(device)`
- On macOS MPS, `device_map="auto"` can fail; now explicitly loads on CPU first


### New Models Added (Completed)

| Model | File | Status |
|-------|------|--------|
| tfidf_lr (LogisticRegression) | `core/pipeline.py` | Added 2026-06-11 |
| tfidf_ridge (RidgeClassifier) | `core/pipeline.py` | Added 2026-06-11 |
| tfidf_knn (KNeighborsClassifier) | `core/pipeline.py` | Added 2026-06-11 |
| tfidf_complement_nb (ComplementNB) | `core/pipeline.py` | Added 2026-06-11 |
| nested_cv_ensemble | `core/ensemble_nested_cv.py` | Implemented |

### Files Created

| File | Purpose |
|------|---------|
| `core/evaluator.py` | Unified CV/evaluation engine with dict-based model iteration |
| `core/pipeline.py` | Model registry and orchestrators (run_standalone, run_ensemble) |
| `core/ensemble_nested_cv.py` | Nested cross-validation ensemble implementation |

### Files Modified

| File | Changes |
|------|---------|
| `task1_classifier_tfidf.py` | Unified `_run_pipeline()` for both modes |
| `task1_classification/new_classifiers.py` | Fixed XGBoost, removed dead code, Apple Silicon compatibility fixes |
| `task1_classification/transformer/transformer.py` | Fixed device_map issue |
| `core/pipeline.py` | Added new models (Ridge, kNN, ComplementNB), fixed XGBoost device param |
| `docs/AGENT_CONTEXT.md` | Updated with current state |
| `README.md` | Updated with new model registry and macOS notes |


## FUTURE WORK

### 1. Complete Model Coverage

**Models to Add:**
- SGD (log_loss) on TF-IDF - Standard baseline for large-scale text
- NearestCentroid - Simple but effective baseline

**Already Added:**
- LogisticRegression, Ridge, kNN, ComplementNB (completed above)

### 2. Nested Cross-Validation for Ensemble Optimization

The current implementation uses a simplified approach:
- Train base models with their CV on training data
- Use CV F1 scores to compute ensemble weights

**Target Approach (methodologically correct):**
1. Outer loop: 5-fold CV on Dtr + Dval
2. Inner loop for each outer fold:
   - Tune base model hyperparameters using nested CV
   - Train optimized models on outer training set
   - Use held-out fold to optimize ensemble weights
3. Final: Retrain on ALL data with optimal hyperparameters

**Implementation:** `core/ensemble_nested_cv.py` provides a framework but needs verification.

### 3. Cross-Encoder Integration (Not Currently Available)

The `cross_encoder` (reranker-MiniLM) was originally planned but:
- Model availability issues
- Requires gated model access or network connectivity

To add when available:
```python
def extract_cross_encoder_features(train_texts, test_texts):
    from sentence_transformers import CrossEncoder
    model = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L6-v2")
    # Return features...
```

### 4. Eliminate Redundant Files (Dead Code)

Delete these files once confirmed no longer needed:
1. `core/ollama_integration.py` - Ollama client (replaced by direct calls)
2. `task2_generation/task2_generator.py_not_used`
3. `task2_generation/task2_ollama_generator.py_not_used`
4. `task2_generation/task2_generator_enhanced.py_not_used`

### 5. Data-Driven Model Selection

Add automated model selection based on validation metrics:
```python
def select_best_model(validation_results):
    """Return list of models ranked by F1(3-class) with recommended ensemble weights."""
    # Sort by f1_macro_3class, compute normalized weights
    pass
```

### 6. Performance Optimization

- Use joblib memory caching for expensive feature extraction
- Implement incremental learning for large datasets
- Consider mixed precision training for transformer models


## XGBoost Segmentation Fault on macOS ARM64 - Technical Details

### The Issue
When running `python submit.py --run-all` on Apple Silicon (M1/M2/M3):
```
[XGB] Training TF-IDF + XGBoost...
zsh: segmentation fault  python submit.py --run-all
```

### Diagnosis Steps

```bash
# Check XGBoost version
python -c "import xgboost as xgb; print(xgb.__version__)"

# Test basic XGBoost functionality
python -c "
import xgboost as xgb
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=100, n_features=20, random_state=42)
clf = xgb.XGBClassifier(n_estimators=5, tree_method='hist')
clf.fit(X[:50], y[:50])
print('XGBoost basic test:', 'PASS' if clf else 'FAIL')
"

# Check BLAS backend
python -c "
import numpy as np
print('NumPy backend info:')
print(f'  BLAS: {np.__config__.show().get(\"BLAS\", \"unknown\")}')
"
```

### Potential Solutions

1. **Conda-forge installation (recommended):**
   ```bash
   conda install -c conda-forge xgboost
   ```

2. **Downgrade to XGBoost 2.x:**
   ```bash
   pip uninstall xgboost
   pip install 'xgboost<3.0'
   ```

3. **Disable XGBoost in code:**
   Edit `core/pipeline.py`:
   ```python
   if XGB_AVAILABLE:
       def make_xgb():
           return None  # Disable XGBoost
   ```
