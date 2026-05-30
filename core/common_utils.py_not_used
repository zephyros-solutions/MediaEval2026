"""
Common Utilities for Classification Approaches

Shared functionality for data loading, evaluation, and result management.
Used by both Ollama and Transformer approaches.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Union
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

# ============================================================================
# LABEL MANAGEMENT (delegated to project root config)
# ============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LABEL_TO_ID as LABEL_MAPPING, ID_TO_LABEL as REVERSE_LABEL_MAPPING


def normalize_label(label):
    """Normalize label string to standard format."""
    if isinstance(label, int):
        return REVERSE_LABEL_MAPPING.get(label, "unknown")

    label_str = str(label).lower().strip()

    # Handle variations
    if label_str in ["none", "no premise", "no_premise"]:
        return "none"
    elif label_str in ["premise", "p"]:
        return "premise"
    elif label_str in ["conclusion", "c"]:
        return "conclusion"

    return label_str


def label_to_id(label):
    """Convert label string to ID."""
    normalized = normalize_label(label)
    return LABEL_TO_ID.get(normalized, -1)


def id_to_label(label_id):
    """Convert label ID to string."""
    return ID_TO_LABEL.get(label_id, "unknown")


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_json_data(filepath: str) -> List[Dict]:
    """
    Load JSON data file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        List of data items (each a dictionary)
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    return data if isinstance(data, list) else [data]


def load_csv_data(filepath: str) -> pd.DataFrame:
    """Load CSV data file."""
    return pd.read_csv(filepath)


def extract_text_and_label(data_item: Dict) -> Tuple[str, Union[str, None]]:
    """
    Extract text and label from data item.
    
    Handles multiple column name variations (text/tweet/sentence, label/class/annotation)
    
    Args:
        data_item: Dictionary with data
        
    Returns:
        Tuple of (text, label)
    """
    # Find text field
    text = None
    for key in ["text", "tweet", "sentence", "content", "input"]:
        if key in data_item:
            text = data_item[key]
            break
    
    # Find label field
    label = None
    for key in ["label", "class", "annotation", "tag", "category"]:
        if key in data_item:
            label = data_item[key]
            break
    
    return text, label


def prepare_data_for_evaluation(
    data: Union[List[Dict], pd.DataFrame],
    predictions: Union[List[Dict], np.ndarray, List[str]]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare true labels and predictions for evaluation.
    
    Args:
        data: Original data items
        predictions: Predictions (list of dicts with 'label' key, or direct labels)
        
    Returns:
        Tuple of (true_labels_ids, pred_labels_ids) as numpy arrays
    """
    true_labels = []
    pred_labels = []
    
    # Convert data to list if DataFrame
    if isinstance(data, pd.DataFrame):
        data = data.to_dict('records')
    
    # Convert predictions to list of labels if needed
    if isinstance(predictions, np.ndarray):
        predictions = predictions.tolist()
    
    # Extract labels
    for i, item in enumerate(data):
        # Get true label
        _, true_label = extract_text_and_label(item)
        if true_label:
            true_labels.append(label_to_id(true_label))
        
        # Get predicted label
        pred = predictions[i]
        if isinstance(pred, dict):
            pred_label = pred.get("label", pred.get("predicted_label", "none"))
        else:
            pred_label = pred
        
        pred_labels.append(label_to_id(pred_label))
    
    return np.array(true_labels), np.array(pred_labels)


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_classification_metrics(true_labels: np.ndarray, pred_labels: np.ndarray) -> Dict:
    """
    Compute comprehensive classification metrics.
    
    Args:
        true_labels: True label IDs
        pred_labels: Predicted label IDs
        
    Returns:
        Dictionary with metrics
    """
    metrics = {
        "accuracy": accuracy_score(true_labels, pred_labels),
        "f1_macro": f1_score(true_labels, pred_labels, average='macro', zero_division=0),
        "f1_weighted": f1_score(true_labels, pred_labels, average='weighted', zero_division=0),
        "precision_macro": precision_score(true_labels, pred_labels, average='macro', zero_division=0),
        "recall_macro": recall_score(true_labels, pred_labels, average='macro', zero_division=0),
    }
    
    return metrics


def get_classification_report(true_labels: np.ndarray, pred_labels: np.ndarray) -> str:
    """
    Get detailed classification report.
    
    Args:
        true_labels: True label IDs
        pred_labels: Predicted label IDs
        
    Returns:
        Classification report as string
    """
    target_names = [id_to_label(i) for i in range(len(LABEL_MAPPING))]
    
    return classification_report(
        true_labels, pred_labels,
        target_names=target_names,
        zero_division=0
    )


def get_confusion_matrix(true_labels: np.ndarray, pred_labels: np.ndarray) -> Dict:
    """
    Get confusion matrix.
    
    Args:
        true_labels: True label IDs
        pred_labels: Predicted label IDs
        
    Returns:
        Dictionary with confusion matrix and labels
    """
    cm = confusion_matrix(true_labels, pred_labels)
    labels = [id_to_label(i) for i in range(len(LABEL_MAPPING))]
    
    return {
        "matrix": cm.tolist(),
        "labels": labels
    }


# ============================================================================
# PREDICTION FORMATTING
# ============================================================================

def format_prediction(text: str, predicted_label: str, confidence: float = None,
                     true_label: str = None) -> Dict:
    """
    Format a single prediction for output.
    
    Args:
        text: Input text
        predicted_label: Predicted label
        confidence: Confidence score (optional)
        true_label: True label for evaluation (optional)
        
    Returns:
        Formatted prediction dictionary
    """
    pred = {
        "text": text,
        "predicted_label": normalize_label(predicted_label),
        "predicted_label_id": label_to_id(predicted_label)
    }
    
    if confidence is not None:
        pred["confidence"] = float(confidence)
    
    if true_label is not None:
        pred["true_label"] = normalize_label(true_label)
        pred["true_label_id"] = label_to_id(true_label)
        pred["correct"] = pred["predicted_label"] == pred["true_label"]
    
    return pred


def save_predictions(predictions: List[Dict], output_path: str) -> None:
    """Save predictions to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"✅ Saved {len(predictions)} predictions to {output_path}")


# ============================================================================
# RESULT COMPARISON
# ============================================================================

def compare_approaches(
    results: Dict[str, Dict],
    metric: str = "f1_macro"
) -> pd.DataFrame:
    """
    Compare multiple classification approaches.
    
    Args:
        results: Dictionary with approach names and their metrics
        metric: Metric to sort by
        
    Returns:
        DataFrame with comparison
    """
    df = pd.DataFrame(results).T
    df = df.sort_values(metric, ascending=False)
    return df


def print_evaluation_summary(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    approach_name: str = "Classification",
    details: bool = True
) -> Dict:
    """
    Print comprehensive evaluation summary.
    
    Args:
        true_labels: True label IDs
        pred_labels: Predicted label IDs
        approach_name: Name of approach being evaluated
        details: Include detailed classification report
        
    Returns:
        Metrics dictionary
    """
    metrics = compute_classification_metrics(true_labels, pred_labels)
    
    print("\n" + "="*80)
    print(f"EVALUATION: {approach_name}")
    print("="*80)
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"F1 (Macro):      {metrics['f1_macro']:.4f}")
    print(f"F1 (Weighted):   {metrics['f1_weighted']:.4f}")
    print(f"Precision:       {metrics['precision_macro']:.4f}")
    print(f"Recall:          {metrics['recall_macro']:.4f}")
    
    if details:
        print(f"\n📋 Classification Report:")
        print(get_classification_report(true_labels, pred_labels))
    
    return metrics


# ============================================================================
# ANALYSIS UTILITIES
# ============================================================================

def get_mispredictions(
    data: List[Dict],
    pred_labels: np.ndarray,
    true_labels: np.ndarray,
    top_n: int = 10
) -> List[Dict]:
    """
    Get most interesting mispredictions for analysis.
    
    Args:
        data: Original data items
        pred_labels: Predicted label IDs
        true_labels: True label IDs
        top_n: Number of mispredictions to return
        
    Returns:
        List of misprediction items
    """
    mispredictions = []
    
    for i, item in enumerate(data):
        if pred_labels[i] != true_labels[i]:
            text, _ = extract_text_and_label(item)
            mispredictions.append({
                "text": text,
                "true_label": id_to_label(true_labels[i]),
                "predicted_label": id_to_label(pred_labels[i])
            })
    
    return mispredictions[:top_n]


def get_label_distribution(labels: np.ndarray) -> Dict[str, int]:
    """Get distribution of labels."""
    unique, counts = np.unique(labels, return_counts=True)
    return {id_to_label(uid): int(count) for uid, count in zip(unique, counts)}


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example: Evaluate predictions
    true_labels = np.array([0, 1, 1, 2, 0])
    pred_labels = np.array([0, 1, 2, 2, 0])
    
    metrics = compute_classification_metrics(true_labels, pred_labels)
    print("Metrics:", metrics)
    print("\nClassification Report:")
    print(get_classification_report(true_labels, pred_labels))
    
    # Example: Format predictions
    sample_pred = format_prediction(
        "Sample text",
        "premise",
        confidence=0.95,
        true_label="premise"
    )
    print("\nFormatted prediction:", sample_pred)
