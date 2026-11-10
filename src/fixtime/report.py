"""Collect the per-unit metrics JSON files into CSV tables."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from fixtime.config import Config, Task
from fixtime.records import UNIT_RESULT

KEYS: dict[Task, list[str]] = {
    "regression": ["scope", "dataset", "model"],
    "classification": ["scope", "dataset", "theta_percentile", "balanced", "model"],
}
INFO = ["fold", "n_train", "n_test", "fit_seconds", "theta", "train_slow_rate"]


def _load(files: Iterable[Path]) -> pd.DataFrame:
    """One row per saved result: unit keys, fit info and every metric as a column."""
    rows: list[dict[str, Any]] = []
    for f in files:
        result = UNIT_RESULT.validate_json(f.read_bytes())
        # regression results have no theta / train_slow_rate: those cells stay empty
        row = {k: getattr(result, k, None) for k in ["task", *KEYS["classification"], *INFO]}
        row.update(result.metrics.model_dump())
        rows.append(row)
    return pd.DataFrame(rows)


def _clean(df: pd.DataFrame, task: Task) -> pd.DataFrame:
    df = df[df["task"] == task].drop(columns="task").dropna(axis=1, how="all")
    return df.sort_values(KEYS[task] + (["fold"] if "fold" in df else [])).reset_index(drop=True)


def summarize_cv(folds: pd.DataFrame, task: Task) -> pd.DataFrame:
    """Mean and std over CV folds for every metric."""
    keys = KEYS[task]
    metrics = [c for c in folds.columns if c not in keys + INFO]
    grouped = folds.groupby(keys, dropna=False)
    summary = grouped[metrics].agg(["mean", "std"])
    summary.columns = ["_".join(col) for col in summary.columns.to_flat_index()]  # mae_mean...
    summary.insert(0, "n_folds", grouped.size())
    if "theta" in folds:
        summary.insert(1, "theta_mean", grouped["theta"].mean())
    return summary.reset_index()


def build_report(cfg: Config) -> list[Path]:
    run_dir = cfg.run_dir
    out = run_dir / "results"
    out.mkdir(parents=True, exist_ok=True)
    written = []

    cv = _load(sorted((run_dir / "cv").rglob("fold*.json")))
    holdout = _load(sorted((run_dir / "holdout").rglob("metrics.json")))
    for task in KEYS:
        if not cv.empty and (cv["task"] == task).any():
            folds = _clean(cv, task)
            written.append(out / f"cv_{task}_folds.csv")
            folds.to_csv(written[-1], index=False)
            written.append(out / f"cv_{task}_summary.csv")
            summarize_cv(folds, task).to_csv(written[-1], index=False)
        if not holdout.empty and (holdout["task"] == task).any():
            written.append(out / f"holdout_{task}.csv")
            _clean(holdout, task).drop(columns="fold").to_csv(written[-1], index=False)
    return written
