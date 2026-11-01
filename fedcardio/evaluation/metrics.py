"""
FedCardio Clinical Evaluation Metrics
Computes ROC-AUC, PR-AUC, Sensitivity, Specificity, Brier calibration, and F1 score.
"""
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, brier_score_loss, f1_score

def compute_clinical_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """
    Computes comprehensive clinical cardiovascular predictive metrics.
    """
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob).ravel()
    y_pred = (y_prob >= threshold).astype(int)
    
    # Handle single class edge cases gracefully
    if len(np.unique(y_true)) < 2:
        roc_auc = 0.5
        pr_auc = float(np.mean(y_true))
    else:
        roc_auc = float(roc_auc_score(y_true, y_prob))
        pr_auc = float(average_precision_score(y_true, y_prob))
        
    # Confusion Matrix: tn, fp, fn, tp
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    # Clinical Sensitivity (Recall): TP / (TP + FN) -> Crucial: do not miss sick patients!
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    
    # Clinical Specificity: TN / (TN + FP) -> Avoid unnecessary invasive catheterizations
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    # Brier Score: Mean squared error of probabilities (lower is better, 0.0 = perfect calibration)
    brier = float(brier_score_loss(y_true, y_prob))
    
    # F1 score & Accuracy
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    accuracy = float((tp + tn) / len(y_true))
    
    return {
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "brier_score": round(brier, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn)
    }
