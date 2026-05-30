"""
MediaEval 2026 Enthymeme Detection -- Shared Configuration

All scripts import from here instead of hardcoding paths,
splits, or class labels. Run scripts from the project root.
"""

import os
import numpy as np
import pandas as pd

# ============== DATA PATHS ============

DATA_CSV_PATH = "/Users/SB/LocalProjects/DataSets/Medieeval/2026/enthymemes_2/merged_annotations_v2.csv"
TEST_CSV_PATH = "/Users/SB/LocalProjects/DataSets/Medieeval/2026/enthymemes_2/test_v2.csv"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")


# ============== CLASS DEFINITIONS ============

CLASS_LABELS = ["premise", "conclusion", "none"]
CLASS_IDS = [0, 1, 2]
LABEL_TO_ID = dict(zip(CLASS_LABELS, CLASS_IDS))
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


# ============== DATA SPLIT HYPERPARAMETERS ============

CV_N_FOLDS = 5
TRAIN_VAL_SPLIT = 0.8
RANDOM_STATE = 42

# ============== OLLAMA FEW-SHOT PARAMETERS ============

FEWSHOT_EXAMPLES_PER_CLASS = 10
FEWSHOT_MULTI_SHOT_AVG = True
FEWSHOT_NUM_ROUNDS = 5
FEWSHOT_TEMPERATURE = 0.1


# ============== TRANSFORMER HYPERPARAMETERS ============

LABEL_SMOOTHING = 0.1
TRANSFORMER_MODEL_NAME = "distilbert-base-uncased"
TRANSFORMER_NUM_LABELS = 3
TRANSFORMER_BATCH_SIZE = 16
TRANSFORMER_LEARNING_RATE = 2e-5
TRANSFORMER_EPOCHS = 3
TRANSFORMER_MAX_LENGTH = 128
TRANSFORMER_WEIGHT_DECAY = 0.01
TRANSFORMER_WARMUP_STEPS = 0


# ============== T5 GENERATION HYPERPARAMETERS ============

T5_MODEL_NAME = "google/flan-t5-base"  # better instruction-following than t5-small; train with LoRA on small dataset
T5_BATCH_SIZE = 16
T5_MAX_LENGTH = 512
T5_LEARNING_RATE = 3e-4
T5_NUM_EPOCHS = 3
T5_NUM_BEAMS = 4
T5_TEMPERATURE = 0.7
T5_TOP_P = 0.9
T5_GENERATION_MAX_LENGTH = 100


# ============== TF-IDF / RANDOM FOREST DEFAULTS ============

TFIDF_DEFAULTS = {
    "max_features": 5000,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.8,
}
RF_DEFAULTS = {
    "n_estimators": [100, 200],
    "max_depth": [15, 25, None],
    "min_samples_leaf": [1, 3],
}


# ============== HELPERS ============


def get_device():
    """Detect best available device (CUDA > MPS > CPU).

    Returns (device, name) -- e.g. (device('mps'), 'MPS (Apple Silicon GPU)')
    """
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda"), "CUDA (GPU)"
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps"), "MPS (Apple Silicon GPU)"
    except Exception:
        pass
    return torch.device("cpu"), "CPU (default)"


def _soft_label_from_row(row):
    """Compute per-instance soft label from 5 annotators.

    Returns a dict mapping class_id -> probability.
    E.g. {0: 0.4, 1: 0.4, 2: 0.2} when 2 annotators say 'premise',
    2 say 'conclusion', and 1 says 'none'.
    """
    counts = np.zeros(3)
    for col in ("ann1_label", "ann2_label", "ann3_label", "ann4_label", "ann5_label"):
        val = getattr(row, col)
        if val in LABEL_TO_ID:
            counts[LABEL_TO_ID[val]] += 1
    return dict(enumerate(counts / 5))


def load_data(csv_path=None):
    """Load the annotation CSV and return a dict with:
        - ids:            list[int]
        - texts:          list[str]
        - ann_labels:     list[dict]  (soft label per instance)
        - majority_labels: np.ndarray[int]
        - implicit_texts: list[str|None]  (annotator implicit texts per instance)
        - df:             pd.DataFrame (full CSV)
    """
    if csv_path is None:
        csv_path = DATA_CSV_PATH
    df = pd.read_csv(csv_path)

    ids = df["id"].tolist()
    texts = df["tweet_text"].tolist()
    ann_labels = [_soft_label_from_row(r) for r in df.itertuples()]
    majority_labels = np.array([LABEL_TO_ID[getattr(row, "majority_label")] for row in df.itertuples()])

    # Collect implicit texts from annotators (may be NaN/empty)
    implicit_cols = [c for c in df.columns if c.endswith("_implicit")]
    implicit_texts = []
    for _, row in df.iterrows():
        implitics = [str(row[c]) for c in implicit_cols if pd.notna(row.get(c))]
        implicit_texts.append(implitics if implitics else None)

    return {
        "ids": ids,
        "texts": texts,
        "ann_labels": ann_labels,
        "majority_labels": majority_labels,
        "implicit_texts": implicit_texts,
        "df": df,
    }


def load_training_data(csv_path=None):
    """Load training data and return the DataFrame with convenience columns.

    This is a thin wrapper around load_data() that returns the df with
    extra columns ('label', 'text', 'soft_label') for easy access by
    all classifier methods. Prints label distribution.
    """
    data = load_data(csv_path)
    df = data["df"].copy()
    df["label"] = data["majority_labels"]
    df["text"] = data["texts"]
    df["soft_label"] = data["ann_labels"]
    print(f"Loaded {len(df)} samples")
    print(df["label"].value_counts().sort_index().to_string(index=False))
    print()
    return df


# ============== TRAIN/TEST SPLIT HELPER ============


def get_train_test_indices(n_samples, test_size=1.0 - TRAIN_VAL_SPLIT, random_state=RANDOM_STATE):
    """Return (train_idx, test_idx) arrays using a random stratified split.

    All scripts should use this function to get identical train/test indices,
    ensuring fair comparison across classifiers.
    """
    from sklearn.model_selection import train_test_split
    train_idx, test_idx = train_test_split(
        np.arange(n_samples),
        test_size=test_size,
        random_state=random_state,
    )
    return np.sort(train_idx), np.sort(test_idx)
