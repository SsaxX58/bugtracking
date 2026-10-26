import json

import joblib
import pandas as pd

from fixtime.config import AGGREGATED
from fixtime.runner import run


def _run(cfg, **kw):
    logs = []
    failures = run(cfg, log=logs.append, **kw)
    assert failures == []
    return logs


def test_full_run_writes_all_outputs(synthetic_cfg):
    _run(synthetic_cfg, scopes=["A"], models=["dummy", "lightgbm"])
    run_dir = synthetic_cfg.run_dir
    assert (run_dir / "splits" / "A.json").exists()

    cv_json = list((run_dir / "cv" / "A" / "D6").rglob("fold*.json"))
    # (2 regressors + 3 thresholds * 2 balance * 2 classifiers) * 5 folds
    assert len(cv_json) == (2 + 12) * 5
    assert len(list((run_dir / "cv").rglob("fold*.parquet"))) == len(cv_json)
    assert len(list((run_dir / "holdout").rglob("model.joblib"))) == 2 + 12


def test_classification_unit_contents(synthetic_cfg):
    _run(synthetic_cfg, scopes=["A"], stages=["holdout"], tasks=["classification"],
         models=["lightgbm"])
    d = synthetic_cfg.run_dir / "holdout" / "A" / "D6" / "classification" / "p75" / "balanced"
    meta = json.loads((d / "lightgbm" / "metrics.json").read_text())
    assert meta["n_train"] + meta["n_test"] == 300
    assert 0.2 < meta["train_slow_rate"] < 0.3
    assert {"mcc", "pr_auc", "roc_auc", "f1_slow"} <= set(meta["metrics"])

    preds = pd.read_parquet(d / "lightgbm" / "predictions.parquet")
    split = json.loads((synthetic_cfg.run_dir / "splits" / "A.json").read_text())
    assert sorted(preds["id"]) == sorted(split["holdout"])
    assert ((preds["y_true"] >= meta["theta"]).astype(int) == preds["label_true"]).all()

    # the saved model is the interface for SHAP: it must predict on raw features
    model = joblib.load(d / "lightgbm" / "model.joblib")
    X = pd.DataFrame({"x1": [0.0], "x2": [1.0]})[meta["feature_names"]]
    assert model.predict_proba(X).shape == (1, 2)


def test_caching_and_partial_rerun(synthetic_cfg):
    kw = {"scopes": ["A"], "stages": ["cv"], "tasks": ["regression"], "models": ["ridge"]}
    _run(synthetic_cfg, **kw)
    assert "0 to run" in _run(synthetic_cfg, **kw)[0]

    (synthetic_cfg.run_dir / "cv" / "A" / "D6" / "regression" / "ridge" / "fold2.json").unlink()
    assert "1 to run" in _run(synthetic_cfg, **kw)[0]

    synthetic_cfg.models["regression"]["ridge"]["alpha"] = 10.0
    assert "5 to run" in _run(synthetic_cfg, **kw)[0]


def test_aggregated_scope(synthetic_cfg):
    _run(synthetic_cfg, scopes=[AGGREGATED], stages=["holdout"], tasks=["regression"],
         models=["dummy"])
    preds = pd.read_parquet(
        synthetic_cfg.run_dir / "holdout" / AGGREGATED / "D6" / "regression" / "dummy"
        / "predictions.parquet"
    )
    assert len(preds) == 100
    assert preds["id"].str.contains(":").all()
