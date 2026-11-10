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
from collections.abc import Callable, Collection, Sequence
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import cast

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pydantic import ValidationError

from fixtime.config import ClassifierParams, Config, RegressorParams, Task
from fixtime.data import Dataset, load_dataset
from fixtime.metrics import classification_metrics, regression_metrics
from fixtime.models import build_classifier, build_regressor
from fixtime.records import (
    CLASSIFICATION,
    REGRESSION,
    STAGES,
    TASKS,
    UNIT_RESULT,
    ClassificationResult,
    Manifest,
    RegressionResult,
    Stage,
    Unit,
)
from fixtime.splits import HOLDOUT, BugId, Split, load_or_create_split
from fixtime.thresholds import compute_theta, make_labels

# Bump when a code change alters results, so that cached units are recomputed.
PIPELINE_VERSION = 2


def enumerate_units(
    cfg: Config,
    scope: str,
    dataset: str,
    split: Split,
    stages: Collection[Stage],
    tasks: Collection[Task],
    models: Collection[str] | None,
) -> list[Unit]:
    folds = [f for f in split.fold_names if (HOLDOUT if f == HOLDOUT else "cv") in stages]
    regressors = [n for n in cfg.models.regression.enabled() if models is None or n in models]
    classifiers = [n for n in cfg.models.classification.enabled() if models is None or n in models]
    units = []
    for fold in folds:
        if REGRESSION in tasks:
            for name in regressors:
                units.append(
                    Unit(scope=scope, dataset=dataset, fold=fold, task=REGRESSION, model=name)
                )
        if CLASSIFICATION in tasks:
            for p in cfg.thresholds:
                for balanced in (False, True):
                    for name in classifiers:
                        units.append(
                            Unit(
                                scope=scope, dataset=dataset, fold=fold, task=CLASSIFICATION,
                                model=name, theta_percentile=p, balanced=balanced,
                            )
                        )
    return units


def unit_hash(cfg: Config, unit: Unit, ds: Dataset, split: Split) -> str:
    """Everything that, if changed, must invalidate the saved result."""
    payload = {
        "version": PIPELINE_VERSION,
        "unit": unit.model_dump(),
        "params": cfg.model_params(unit.task, unit.model).kwargs(),
        "seed": cfg.seed,
        "target": cfg.target,
        "data": ds.file_hash,
        "features": ds.feature_names,
        "split": split.ids_hash,
        "folds": cfg.split.model_dump(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def is_done(path: Path, expected_hash: str) -> bool:
    """A unit is done when its result file exists, is valid and has the expected hash."""
    if not path.exists():
        return False
    try:
        return UNIT_RESULT.validate_json(path.read_bytes()).hash == expected_hash
    except ValidationError:
        return False


def _write_atomic(path: Path, text: str) -> None:
    # write then rename, so a crash never leaves a half-written "done" marker
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def execute_unit(
    unit: Unit,
    params: RegressorParams | ClassifierParams,
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
        paths.metrics.parent.mkdir(parents=True, exist_ok=True)
        X_train = pd.DataFrame(X[train_pos], columns=feature_names)
        X_test = pd.DataFrame(X[test_pos], columns=feature_names)
        y_train, y_test = y[train_pos], y[test_pos]
        common = {
            **unit.model_dump(),
            "hash": unit_hash_,
            "params": params.kwargs(),
            "n_train": len(train_pos),
            "n_test": len(test_pos),
            "feature_names": feature_names if paths.model else None,
        }

        start = time.perf_counter()
        result: RegressionResult | ClassificationResult
        if unit.task == REGRESSION:
            model = build_regressor(cast(RegressorParams, params), seed)
            model.fit(X_train, y_train)
            fit_seconds = time.perf_counter() - start
            # linear models can extrapolate far outside the target range in log space,
            # and expm1 turns that into absurd values: clip to the training range
            pred = np.clip(model.predict(X_test), 0, y_train.max())
            result = RegressionResult(
                **common,
                fit_seconds=fit_seconds,
                pred_clip_max=float(y_train.max()),
                metrics=regression_metrics(y_test, pred),
            )
            predictions = pd.DataFrame({"id": ids[test_pos], "y_true": y_test, "y_pred": pred})
        else:
            assert unit.theta_percentile is not None and unit.balanced is not None
            theta = compute_theta(y_train, unit.theta_percentile)
            l_train, l_test = make_labels(y_train, theta), make_labels(y_test, theta)
            if len(np.unique(l_train)) < 2:
                raise ValueError(f"θ={theta} leaves a single class in the training set")
            model = build_classifier(cast(ClassifierParams, params), seed, unit.balanced, l_train)
            model.fit(X_train, l_train)
            fit_seconds = time.perf_counter() - start
            proba = model.predict_proba(X_test)[:, 1]
            label_pred = (proba >= 0.5).astype(int)
            result = ClassificationResult(
                **common,
                fit_seconds=fit_seconds,
                theta=theta,
                train_slow_rate=float(l_train.mean()),
                metrics=classification_metrics(l_test, label_pred, proba),
            )
            predictions = pd.DataFrame(
                {
                    "id": ids[test_pos],
                    "y_true": y_test,
                    "label_true": l_test,
                    "proba_slow": proba,
                    "label_pred": label_pred,
                }
            )

        predictions.to_parquet(paths.predictions, index=False)
        if paths.model:
            joblib.dump(model, paths.model, compress=3)
        _write_atomic(paths.metrics, result.model_dump_json(indent=2))
        return unit, None
    except Exception:  # noqa: BLE001  (one failed unit must not stop the whole run)
        return unit, traceback.format_exc()


def _positions(ds: Dataset, ids: list[BugId]) -> np.ndarray:
    """Row positions of the split ids present in this dataset.

    Snapshot datasets (e.g. bugs still open after 24h) contain only a subset of the
    reference ids, so ids missing from the dataset are simply skipped.
    """
    pos = ds.y.index.get_indexer(pd.Index(ids))
    return pos[pos >= 0]


def write_manifest(cfg: Config, argv: list[str]) -> Path:
    def git(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=cfg.base_dir, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    packages = [
        "fixtime", "numpy", "pandas", "scikit-learn", "xgboost", "lightgbm", "joblib", "pydantic",
    ]
    manifest = Manifest(
        timestamp=datetime.now(UTC),
        argv=argv,
        git_commit=git("rev-parse", "HEAD"),
        git_dirty=bool(git("status", "--porcelain", "--untracked-files=no")),
        python=platform.python_version(),
        packages={p: metadata.version(p) for p in packages},
        config=cfg,
    )
    d = cfg.run_dir / "manifests"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(manifest.model_dump_json(indent=2))
    return path


def _print_flush(msg: str) -> None:
    print(msg, flush=True)


def run(
    cfg: Config,
    scopes: Sequence[str] | None = None,
    datasets: Sequence[str] | None = None,
    stages: Collection[Stage] = STAGES,
    tasks: Collection[Task] = TASKS,
    models: Collection[str] | None = None,
    force: bool = False,
    limit: int | None = None,
    log: Callable[[str], None] = _print_flush,
) -> list[tuple[Unit, str]]:
    """Run all pending units. Returns the failed units with their tracebacks."""
    failures: list[tuple[Unit, str]] = []
    for scope in scopes or cfg.scopes:
        # the split is defined once per scope on the reference dataset, which must contain
        # every bug; the other datasets of the scope reuse it on their subset of bugs
        reference = load_dataset(cfg, scope, cfg.split_reference, limit=limit)
        split = load_or_create_split(cfg, reference)
        universe = set(split.all_ids)
        for dataset in datasets or list(cfg.datasets):
            if dataset == cfg.split_reference:
                ds = reference
            else:
                ds = load_dataset(cfg, scope, dataset, keep_ids=universe if limit else None)
            outside = set(ds.y.index) - universe
            if outside:
                raise ValueError(
                    f"{scope}/{dataset}: {len(outside)} ids are not in the reference dataset "
                    f"{cfg.split_reference!r}, e.g. {sorted(outside, key=str)[:3]}"
                )
            units = enumerate_units(cfg, scope, dataset, split, stages, tasks, models)
            hashes = {u: unit_hash(cfg, u, ds, split) for u in units}
            pending = [
                u for u in units if force or not is_done(u.paths(cfg.run_dir).metrics, hashes[u])
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
                    u, cfg.model_params(u.task, u.model), cfg.seed, cfg.run_dir, hashes[u],
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
