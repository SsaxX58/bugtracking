"""Regression and classification metrics."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    median_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    root_mean_squared_error,
)


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Metrics in days, plus MAE/RMSE in log1p space where the model is trained."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    lt, lp = np.log1p(y_true), np.log1p(y_pred)
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "mdae": median_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "mae_log": mean_absolute_error(lt, lp),
        "rmse_log": root_mean_squared_error(lt, lp),
        "r2_log": r2_score(lt, lp),
    }


def classification_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    """Positive class is 1 = slow. AUCs are undefined when the test set has a single class."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    two_classes = len(np.unique(y_true)) == 2
    return {
        "mcc": matthews_corrcoef(y_true, y_pred),
        "pr_auc": average_precision_score(y_true, y_score) if two_classes else float("nan"),
        "roc_auc": roc_auc_score(y_true, y_score) if two_classes else float("nan"),
        "f1_slow": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_slow": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_slow": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "slow_rate": float(y_true.mean()),
    }
