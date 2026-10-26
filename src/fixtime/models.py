"""Model registry. Every model is a sklearn Pipeline with median imputation."""

from __future__ import annotations

from typing import Any

import numpy as np
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

SCALED = {"ridge", "logreg"}


def _pipeline(name: str, estimator) -> Pipeline:
    steps = [("impute", SimpleImputer(strategy="median"))]
    if name in SCALED:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def build_regressor(name: str, params: dict[str, Any], seed: int):
    """Regressors learn log1p(y); predictions are mapped back to days with expm1."""
    params = dict(params)
    if name == "dummy":
        est = DummyRegressor(**params)
    elif name == "ridge":
        est = Ridge(**params)
    elif name == "random_forest":
        est = RandomForestRegressor(random_state=seed, n_jobs=1, **params)
    elif name == "xgboost":
        est = XGBRegressor(random_state=seed, n_jobs=1, **params)
    elif name == "lightgbm":
        est = LGBMRegressor(random_state=seed, n_jobs=1, verbose=-1, **params)
    else:
        raise ValueError(f"unknown regressor {name!r}")
    return TransformedTargetRegressor(
        regressor=_pipeline(name, est), func=np.log1p, inverse_func=np.expm1, check_inverse=False
    )


def build_classifier(name: str, params: dict[str, Any], seed: int, balanced: bool, y_train):
    """Balanced variants reweight classes inversely to their training frequency."""
    params = dict(params)
    weight = "balanced" if balanced else None
    if name == "dummy":
        # a prior-based dummy ignores class weights, it is the same in both variants
        est = DummyClassifier(**params)
    elif name == "logreg":
        est = LogisticRegression(class_weight=weight, random_state=seed, **params)
    elif name == "random_forest":
        est = RandomForestClassifier(class_weight=weight, random_state=seed, n_jobs=1, **params)
    elif name == "xgboost":
        y_train = np.asarray(y_train)
        pos = max(int(y_train.sum()), 1)
        spw = (len(y_train) - pos) / pos if balanced else 1.0
        est = XGBClassifier(scale_pos_weight=spw, random_state=seed, n_jobs=1, **params)
    elif name == "lightgbm":
        est = LGBMClassifier(class_weight=weight, random_state=seed, n_jobs=1, verbose=-1, **params)
    else:
        raise ValueError(f"unknown classifier {name!r}")
    return _pipeline(name, est)
