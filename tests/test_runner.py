import json

import joblib
import pandas as pd
import pytest

from fixtime.config import AGGREGATED
from fixtime.data import load_dataset
from fixtime.runner import run


def _run(cfg, **kw):
    logs = []
    failures = run(cfg, log=logs.append, **kw)
    assert failures == []
    return logs


def test_full_run_writes_all_outputs(synthetic_cfg):
    _run(synthetic_cfg, scopes=["A"], datasets=["T0"], models=["dummy", "lightgbm"])
    run_dir = synthetic_cfg.run_dir
    assert (run_dir / "splits" / "A.json").exists()

    cv_json = list((run_dir / "cv" / "A" / "T0").rglob("fold*.json"))
    # (2 regressors + 3 thresholds * 2 balance * 2 classifiers) * 5 folds
    assert len(cv_json) == (2 + 12) * 5
    assert len(list((run_dir / "cv").rglob("fold*.parquet"))) == len(cv_json)
    assert len(list((run_dir / "holdout").rglob("model.joblib"))) == 2 + 12


def test_classification_unit_contents(synthetic_cfg):
    _run(synthetic_cfg, scopes=["A"], datasets=["T0"], stages=["holdout"],
         tasks=["classification"], models=["lightgbm"])
    d = synthetic_cfg.run_dir / "holdout" / "A" / "T0" / "classification" / "p75" / "balanced"
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
    kw = {"scopes": ["A"], "datasets": ["T0"], "stages": ["cv"], "tasks": ["regression"],
          "models": ["ridge"]}
    _run(synthetic_cfg, **kw)
    assert "0 to run" in _run(synthetic_cfg, **kw)[0]

    (synthetic_cfg.run_dir / "cv" / "A" / "T0" / "regression" / "ridge" / "fold2.json").unlink()
    assert "1 to run" in _run(synthetic_cfg, **kw)[0]

    synthetic_cfg.models["regression"]["ridge"]["alpha"] = 10.0
    assert "5 to run" in _run(synthetic_cfg, **kw)[0]


def test_aggregated_scope(synthetic_cfg):
    _run(synthetic_cfg, scopes=[AGGREGATED], datasets=["T0"], stages=["holdout"],
         tasks=["regression"], models=["dummy"])
    preds = pd.read_parquet(
        synthetic_cfg.run_dir / "holdout" / AGGREGATED / "T0" / "regression" / "dummy"
        / "predictions.parquet"
    )
    assert len(preds) == 100
    assert preds["id"].str.contains(":").all()


def test_snapshot_reuses_reference_split(synthetic_cfg):
    _run(synthetic_cfg, scopes=["A"], stages=["holdout"], tasks=["regression"],
         models=["dummy"])
    split = json.loads((synthetic_cfg.run_dir / "splits" / "A.json").read_text())
    t24 = load_dataset(synthetic_cfg, "A", "T24")
    preds = pd.read_parquet(
        synthetic_cfg.run_dir / "holdout" / "A" / "T24" / "regression" / "dummy"
        / "predictions.parquet"
    )
    # the T24 holdout is exactly the T0 holdout restricted to the bugs still open at 24h
    assert sorted(preds["id"]) == sorted(set(split["holdout"]) & set(t24.y.index))
    meta = json.loads(
        (synthetic_cfg.run_dir / "holdout" / "A" / "T24" / "regression" / "dummy"
         / "metrics.json").read_text()
    )
    assert meta["n_train"] + meta["n_test"] == len(t24.y)


def test_ids_outside_reference_fail(synthetic_cfg):
    path = synthetic_cfg.dataset_path("A", "T24")
    df = pd.read_csv(path)
    df.loc[len(df)] = df.iloc[0].to_numpy()
    df.loc[len(df) - 1, "bug_id"] = 99999
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="not in the reference dataset"):
        run(synthetic_cfg, scopes=["A"], datasets=["T24"], log=lambda _: None)
