"""Model registry. Every model is a sklearn Pipeline with median imputation.

The estimator is chosen by the type of its validated hyperparameters (see config.py), so a
configuration that type-checks always maps to exactly one estimator.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from numpy.typing import ArrayLike
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from fixtime.config import (
    ClassifierParams,
    DummyClassifierParams,
    DummyRegressorParams,
    LightGBMParams,
    LogisticRegressionParams,
    RandomForestParams,
    RegressorParams,
    RidgeParams,
    XGBoostParams,
)


def _pipeline(estimator: Any, scale: bool) -> Pipeline:
    steps: list[tuple[str, Any]] = [("impute", SimpleImputer(strategy="median"))]
    if scale:  # only the linear models need standardized features
        steps.append(("scale", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def build_regressor(params: RegressorParams, seed: int) -> TransformedTargetRegressor:
    """Regressors learn log1p(y); predictions are mapped back to days with expm1."""
    kw = params.kwargs()
    match params:
        case DummyRegressorParams():
            est: Any = DummyRegressor(**kw)
        case RidgeParams():
            est = Ridge(**kw)
        case RandomForestParams():
            est = RandomForestRegressor(random_state=seed, n_jobs=1, **kw)
        case XGBoostParams():
            est = XGBRegressor(random_state=seed, n_jobs=1, **kw)
        case LightGBMParams():
            est = LGBMRegressor(random_state=seed, n_jobs=1, verbose=-1, **kw)
    return TransformedTargetRegressor(
        regressor=_pipeline(est, scale=isinstance(params, RidgeParams)),
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def build_classifier(
    params: ClassifierParams, seed: int, balanced: bool, y_train: ArrayLike
) -> Pipeline:
    """Balanced variants reweight classes inversely to their training frequency."""
    kw = params.kwargs()
    weight = "balanced" if balanced else None
    match params:
        case DummyClassifierParams():
            # a prior-based dummy ignores class weights, it is the same in both variants
            est: Any = DummyClassifier(**kw)
        case LogisticRegressionParams():
            est = LogisticRegression(class_weight=weight, random_state=seed, **kw)
        case RandomForestParams():
            est = RandomForestClassifier(class_weight=weight, random_state=seed, n_jobs=1, **kw)
        case XGBoostParams():
            labels = np.asarray(y_train)
            pos = max(int(labels.sum()), 1)
            spw = (len(labels) - pos) / pos if balanced else 1.0  # n_fast / n_slow
            est = XGBClassifier(scale_pos_weight=spw, random_state=seed, n_jobs=1, **kw)
        case LightGBMParams():
            est = LGBMClassifier(class_weight=weight, random_state=seed, n_jobs=1, verbose=-1, **kw)
    return _pipeline(est, scale=isinstance(params, LogisticRegressionParams))
