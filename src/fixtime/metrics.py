"""Regression and classification metrics."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from pydantic import BaseModel, ConfigDict
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


class RegressionMetrics(BaseModel):
    """Errors in days, plus the same errors in log1p space where the model is trained."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mae: float
    mdae: float
    rmse: float
    r2: float
    mae_log: float
    rmse_log: float
    r2_log: float


class ClassificationMetrics(BaseModel):
    """Positive class is 1 = slow. AUCs are None when the test set has a single class."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    mcc: float
    pr_auc: float | None
    roc_auc: float | None
    f1_slow: float
    f1_macro: float
    balanced_accuracy: float
    accuracy: float
    precision_slow: float
    recall_slow: float
    tn: int
    fp: int
    fn: int
    tp: int
    slow_rate: float


def regression_metrics(y_true: ArrayLike, y_pred: ArrayLike) -> RegressionMetrics:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    lt, lp = np.log1p(yt), np.log1p(yp)
    return RegressionMetrics(
        mae=mean_absolute_error(yt, yp),
        mdae=median_absolute_error(yt, yp),
        rmse=root_mean_squared_error(yt, yp),
        r2=r2_score(yt, yp),
        mae_log=mean_absolute_error(lt, lp),
        rmse_log=root_mean_squared_error(lt, lp),
        r2_log=r2_score(lt, lp),
    )


def _finite_or_none(value: float) -> float | None:
    return None if math.isnan(value) else float(value)


def classification_metrics(
    y_true: ArrayLike, y_pred: ArrayLike, y_score: ArrayLike
) -> ClassificationMetrics:
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)
    tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
    two_classes = len(np.unique(yt)) == 2
    return ClassificationMetrics(
        mcc=matthews_corrcoef(yt, yp),
        pr_auc=_finite_or_none(average_precision_score(yt, y_score)) if two_classes else None,
        roc_auc=_finite_or_none(roc_auc_score(yt, y_score)) if two_classes else None,
        f1_slow=f1_score(yt, yp, pos_label=1, zero_division=0),
        f1_macro=f1_score(yt, yp, average="macro", zero_division=0),
        balanced_accuracy=balanced_accuracy_score(yt, yp),
        accuracy=accuracy_score(yt, yp),
        precision_slow=precision_score(yt, yp, pos_label=1, zero_division=0),
        recall_slow=recall_score(yt, yp, pos_label=1, zero_division=0),
        tn=int(tn),
        fp=int(fp),
        fn=int(fn),
        tp=int(tp),
        slow_rate=float(yt.mean()),
    )
