"""Enumerate work units, skip the ones already done, run the rest in parallel.

A unit is one model fit: (scope, dataset, fold, task, model[, θ percentile, balanced]).
Each unit writes its metrics JSON last, so an existing JSON with a matching hash means
the unit is complete and is skipped on the next run.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from fixtime.config import Config
from fixtime.data import Dataset, load_dataset
from fixtime.metrics import classification_metrics, regression_metrics
from fixtime.models import build_classifier, build_regressor
from fixtime.splits import HOLDOUT, Split, load_or_create_split
from fixtime.thresholds import compute_theta, make_labels

REGRESSION = "regression"
CLASSIFICATION = "classification"
TASKS = (REGRESSION, CLASSIFICATION)
STAGES = ("cv", HOLDOUT)


@dataclass(frozen=True)
class Unit:
    scope: str
    dataset: str
    fold: int | str  # CV fold index or HOLDOUT
    task: str
    model: str
    theta_percentile: int | None = None
    balanced: bool | None = None

    @property
    def stage(self) -> str:
        return HOLDOUT if self.fold == HOLDOUT else "cv"

    def directory(self, run_dir: Path) -> Path:
        d = run_dir / self.stage / self.scope / self.dataset / self.task
        if self.task == CLASSIFICATION:
            d = d / f"p{self.theta_percentile}" / ("balanced" if self.balanced else "unbalanced")
        return d / self.model

    def paths(self, run_dir: Path) -> dict[str, Path]:
        d = self.directory(run_dir)
        if self.stage == HOLDOUT:
            return {
                "metrics": d / "metrics.json",
                "predictions": d / "predictions.parquet",
                "model": d / "model.joblib",
            }
        return {"metrics": d / f"fold{self.fold}.json", "predictions": d / f"fold{self.fold}.parquet"}


def enumerate_units(
    cfg: Config, scope: str, dataset: str, split: Split, stages, tasks, models
) -> list[Unit]:
    folds = [f for f in split.fold_names if (HOLDOUT if f == HOLDOUT else "cv") in stages]
    units = []
    for fold in folds:
        if REGRESSION in tasks:
            for name in cfg.models[REGRESSION]:
                if models is None or name in models:
                    units.append(Unit(scope, dataset, fold, REGRESSION, name))
        if CLASSIFICATION in tasks:
            for p in cfg.thresholds:
                for balanced in (False, True):
                    for name in cfg.models[CLASSIFICATION]:
                        if models is None or name in models:
                            units.append(
                                Unit(scope, dataset, fold, CLASSIFICATION, name, p, balanced)
                            )
    return units


def unit_hash(cfg: Config, unit: Unit, ds: Dataset, split: Split) -> str:
    """Everything that, if changed, must invalidate the saved result."""
    payload = {
        "unit": asdict(unit),
        "params": cfg.models[unit.task][unit.model],
        "seed": cfg.seed,
        "target": cfg.target,
        "data": ds.file_hash,
        "features": ds.feature_names,
        "split": split.ids_hash,
        "folds": cfg.split,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def is_done(path: Path, expected_hash: str) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get("hash") == expected_hash
    except json.JSONDecodeError:
        return False


def _write_json(path: Path, data: dict) -> None:
    # write then rename, so a crash never leaves a half-written "done" marker
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, default=float))
    os.replace(tmp, path)


def execute_unit(
    unit: Unit,
    params: dict,
    seed: int,
    run_dir: Path,
    unit_hash_: str,
    X: np.ndarray,
    y: np.ndarray,
    ids: np.ndarray,
    feature_names: list[str],
    train_pos: np.ndarray,
    test_pos: np.ndarray,
) -> tuple[Unit, str | None]:
    """Fit one model and save its predictions and metrics. Returns an error message or None."""
    try:
        paths = unit.paths(run_dir)
        paths["metrics"].parent.mkdir(parents=True, exist_ok=True)
        X_train = pd.DataFrame(X[train_pos], columns=feature_names)
        X_test = pd.DataFrame(X[test_pos], columns=feature_names)
        y_train, y_test = y[train_pos], y[test_pos]
        info = {"n_train": len(train_pos), "n_test": len(test_pos)}

        start = time.perf_counter()
        if unit.task == REGRESSION:
            model = build_regressor(unit.model, params, seed)
            model.fit(X_train, y_train)
            info["fit_seconds"] = time.perf_counter() - start
            pred = model.predict(X_test)
            metrics = regression_metrics(y_test, pred)
            predictions = pd.DataFrame({"id": ids[test_pos], "y_true": y_test, "y_pred": pred})
        else:
            theta = compute_theta(y_train, unit.theta_percentile)
            l_train, l_test = make_labels(y_train, theta), make_labels(y_test, theta)
            if len(np.unique(l_train)) < 2:
                raise ValueError(f"θ={theta} leaves a single class in the training set")
            model = build_classifier(unit.model, params, seed, unit.balanced, l_train)
            model.fit(X_train, l_train)
            info["fit_seconds"] = time.perf_counter() - start
            proba = model.predict_proba(X_test)[:, 1]
            pred = (proba >= 0.5).astype(int)
            metrics = classification_metrics(l_test, pred, proba)
            info.update(theta=theta, train_slow_rate=float(l_train.mean()))
            predictions = pd.DataFrame(
                {
                    "id": ids[test_pos],
                    "y_true": y_test,
                    "label_true": l_test,
                    "proba_slow": proba,
                    "label_pred": pred,
                }
            )

        predictions.to_parquet(paths["predictions"], index=False)
        if "model" in paths:
            joblib.dump(model, paths["model"], compress=3)
            info["feature_names"] = feature_names
        _write_json(
            paths["metrics"],
            {**asdict(unit), "hash": unit_hash_, "params": params, **info, "metrics": metrics},
        )
        return unit, None
    except Exception:  # noqa: BLE001  (one failed unit must not stop the whole run)
        return unit, traceback.format_exc()


def _positions(ds: Dataset, ids: list) -> np.ndarray:
    pos = ds.y.index.get_indexer(ids)
    if (pos < 0).any():
        raise ValueError(f"{ds.scope}/{ds.name}: split ids not found in dataset")
    return pos


def write_manifest(cfg: Config, argv: list[str]) -> Path:
    def git(*args):
        try:
            return subprocess.run(
                ["git", *args], cwd=cfg.base_dir, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    packages = ["fixtime", "numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "joblib"]
    manifest = {
        "timestamp": datetime.now(UTC).isoformat(),
        "argv": argv,
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "python": platform.python_version(),
        "packages": {p: metadata.version(p) for p in packages},
        "config": cfg.raw,
    }
    d = cfg.run_dir / "manifests"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(manifest, indent=2, default=str))
    return path


def run(
    cfg: Config,
    scopes=None,
    datasets=None,
    stages=STAGES,
    tasks=TASKS,
    models=None,
    force: bool = False,
    limit: int | None = None,
    log=print,
) -> list[tuple[Unit, str]]:
    """Run all pending units. Returns the failed units with their tracebacks."""
    failures = []
    for scope in scopes or cfg.scopes:
        for dataset in datasets or list(cfg.datasets):
            ds = load_dataset(cfg, scope, dataset, limit=limit)
            split = load_or_create_split(cfg, ds)
            units = enumerate_units(cfg, scope, dataset, split, stages, tasks, models)
            hashes = {u: unit_hash(cfg, u, ds, split) for u in units}
            pending = [
                u for u in units if force or not is_done(u.paths(cfg.run_dir)["metrics"], hashes[u])
            ]
            log(f"[{scope}/{dataset}] {len(units)} units, {len(units) - len(pending)} cached, "
                f"{len(pending)} to run")
            if not pending:
                continue

            X = ds.X.to_numpy()
            y = ds.y.to_numpy()
            ids = np.asarray(ds.y.index)
            positions = {}
            for fold in {u.fold for u in pending}:
                train, test = split.train_test(fold)
                positions[fold] = (_positions(ds, train), _positions(ds, test))

            jobs = (
                delayed(execute_unit)(
                    u, cfg.models[u.task][u.model], cfg.seed, cfg.run_dir, hashes[u],
                    X, y, ids, ds.feature_names, *positions[u.fold],
                )
                for u in pending
            )
            start = time.perf_counter()
            results = Parallel(n_jobs=cfg.n_jobs, return_as="generator_unordered")(jobs)
            for i, (unit, error) in enumerate(results, 1):
                if error:
                    failures.append((unit, error))
                    log(f"  FAILED {unit}\n{error}")
                if i % 25 == 0 or i == len(pending):
                    log(f"  {i}/{len(pending)} done ({time.perf_counter() - start:.0f}s)")
    return failures
