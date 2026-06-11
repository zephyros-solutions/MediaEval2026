# AGENT_CONTEXT.md - Architecture & Extension Guide

**Audience**: Developers, AI agents, automation tools enhancing this codebase
**Last Updated**: Jun 12, 2026


Current State
Critical problems:

submit.py has 5 undefined variable errors — I deleted train_and_predict_tfidf_for_ensemble, train_and_predict_transformer_for_ensemble, train_and_predict_svm_for_ensemble, train_and_predict_xgboost_for_ensemble, and train_and_predict_sbert_for_ensemble (they were dead functions that no one called), but I left 5 calls to them in submit_task1() at lines 340, 344, 348, 352, 359. The code is currently broken and won't run.

_build_classifier was broken — lr_balanced and lr_weighted had identical implementations (both used "balanced" class weights). The distinction meant nothing. I also wrote sgd_log_weighted to accept {"balanced": None} which is a nonsensical dict, not a valid class_weight value. These were not typos; the entire classifier selection logic was wrong.

XGBoost import of unused cross_encoder — new_classifiers.py imports cross_encoder (lines 356 and 444) but neither is used. This causes Pylance reportMissingImports and would crash if imported. The XGBoost model also uses the deprecated use_label_encoder=False parameter removed in XGBoost 3.x, which causes segfaults on Apple Silicon.

T5+LoRA generation produced nonsense — T5 generated topically related but semantically disconnected phrases (e.g., discussing vaccines when the tweet was about immigration). We switched to Ollama because T5's outputs were unreliable.

Model/Approach Choices — What Was Wrong and What Wasn't Tried
Classification models: wrong choices, insufficient exploration
What we have (in ensemble):

DistilBERT features + Balanced LR
SBERT embeddings + LR
TF-IDF + XGBoost
TF-IDF + Random Forest
TF-IDF + LinearSVC
What was missed:

LogisticRegression on TF-IDF — this is arguably the strongest possible statistical baseline for text classification, but it wasn't tried. My code has no path from LR to TF-IDF features at all.
Ridge Classifier on TF-IDF — well-suited for high-dimensional sparse text data; never tested
kNN — a natural comparison against tree-based methods on TF-IDF
Complement Naive Bayes — specifically designed for imbalanced text classification
Log-loss SGD (linear) — standard baseline for large-scale text
NearestCentroid — another simple baseline that should be in the comparison
The RF choice was unjustified. Random Forest is a sensible sklearn default but not inherently better than LR for text. I selected RF because it was a familiar pattern, not because domain reasoning pointed to it. Linear classifiers are typically stronger on TF-IDF because tree-based methods fragment the high-dimensional sparse feature space unnecessarily.

The ensemble of 5 models is a patchwork, not a principled set where each fills a distinct gap. We can't answer whether LR-on-TF-IDF would have beaten RF, because we never tried it. The comparison framework itself is broken by this omission.

Task 2: T5+LoRA vs Ollama
T5+LoRA: Fine-tuned on 364 training pairs with LoRA (r=8, α=16). Produced coherent but sometimes semantically disconnected outputs — the model learned dominant topics rather than argumentative structure. Output quality was inconsistent and unreliable.

Ollama (gemma4): Simple prompting, no fine-tuning. Outperformed T5+LoRA on qualitative assessment despite being a simpler approach. No data-driven comparison was done before choosing — we just observed that Ollama produced better outputs.

What should have been tried but wasn't
LR-on-TF-IDF as the primary statistical baseline (should be tried first)
Ridge, kNN, ComplementNB, SGD(log_loss) on TF-IDF features (fair comparison within the same feature space)
Data-driven T5 vs Ollama evaluation — we should have computed ROSCOSS similarity for both on validation data before choosing
Structural Problems
1. No shared CV/evaluation infrastructure
Each classifier module (task1_classifier_tfidf.py, transformer/transformer.py, new_classifiers.py) implements its own:

_run_pipeline() with different signatures
run_cv() with different parameter grids and CV strategies
Own metric computation, sometimes via compute_metrics(), sometimes manually
There is no guarantee that models are compared on the same data split, with the same CV folds, or with the same metrics. This means any comparison between methods is meaningless because the evaluation itself differs.

2. Submit.py's wrapper function sprawl
submit.py contains 10+ thin wrapper functions (train_and_predict_*_for_ensemble) that exist only to call upstream code through one additional layer of indirection with no added value. They were never called directly — I deleted the ones that weren't called and now left dangling calls in submit_task1(). The function should have had one canonical path: one function that takes a model config, trains it on all data, and returns artifacts.

3. No dict-based loop architecture
The ideal design would be:

A single evaluator function that iterates over a dict of model configurations
Each model runs through the same CV/fold/evaluation pipeline
Results collected into a uniform format for comparison
Instead, each classifier has its own entry point, its own parameter grid definition, and sometimes its own metric computation. There's no way to add a new model without writing a completely independent training loop.

Target Architecture (Desired State)
The design goals:
One data loading path — all classifiers use the same load_data() call and the same train/test split logic
One CV/evaluation path — a single evaluator that runs models through identical folds, metrics, and scoring
Dict-driven model iteration — define models as config entries (name, feature extractor, classifier, params), run them all in a loop, collect results uniformly
Two top-level functions:
run_standalone_models(models_dict) — runs all standalone classifiers through CV on 80%, returns ranked results
run_ensemble_models(models_dict) — builds ensembles from the same models with configurable strategies
The code should have:
No duplicate parameter grids or CV logic between modules
No thin wrapper functions (no more _train_and_predict_*_for_ensemble)
One run_model(config, mode) function that takes a config dict and runs it in either standalone (80% CV) or ensemble (all data) mode
Uniform result format: {name, f1_macro_3class, f1_macro_2class, per_class, cross_entropy, elapsed}
Files needed:
core/evaluator.py — the single evaluation engine with dict-based model iteration
core/pipeline.py — thin orchestrators that call evaluator for standalone vs ensemble modes
Classifier modules simplified to just: feature extractors and model definitions (no training loops, no CV code)
What to remove:
All _run_pipeline, run_*_for_ensemble, train_and_predict_* wrapper functions from all files
Duplicate parameter grid definitions across classifier modules
cross_encoder import from new_classifiers.py (unused)
The use_label_encoder=False parameter from XGBoost models
Dead/dead code paths in submit.py that reference deleted functions

README should document:
The target architecture with the dict-based loop as the design pattern
Which models are currently tested and their results
What additional model families need to be tried (LR, Ridge, kNN, ComplementNB, SGD on TF-IDF)
That adding a new classifier requires only a config entry, not a new training loop
