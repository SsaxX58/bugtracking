"""Typed records of the pipeline: work units, their saved results and the run manifest.

These models define the exact shape of the JSON files under outputs/<run>/, which are
written with `model_dump_json` and read back with `model_validate_json`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from fixtime.config import Config, Task
from fixtime.metrics import ClassificationMetrics, RegressionMetrics
from fixtime.splits import HOLDOUT, Fold

REGRESSION: Task = "regression"
CLASSIFICATION: Task = "classification"
TASKS: tuple[Task, ...] = (REGRESSION, CLASSIFICATION)
Stage = Literal["cv", "holdout"]
STAGES: tuple[Stage, ...] = ("cv", HOLDOUT)


class UnitPaths(BaseModel):
    metrics: Path
    predictions: Path
    model: Path | None  # saved only for the holdout stage


class Unit(BaseModel):
    """One model fit: (scope, dataset, fold, task, model[, θ percentile, balanced])."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scope: str
    dataset: str
    fold: Fold
    task: Task
    model: str
    theta_percentile: int | None = None
    balanced: bool | None = None

    @model_validator(mode="after")
    def _classification_fields(self) -> Unit:
        has_fields = self.theta_percentile is not None and self.balanced is not None
        no_fields = self.theta_percentile is None and self.balanced is None
        if self.task == CLASSIFICATION and not has_fields:
            raise ValueError("a classification unit needs theta_percentile and balanced")
        if self.task == REGRESSION and not no_fields:
            raise ValueError("a regression unit has no theta_percentile or balanced")
        return self

    @property
    def stage(self) -> Stage:
        return HOLDOUT if self.fold == HOLDOUT else "cv"

    def directory(self, run_dir: Path) -> Path:
        d = run_dir / self.stage / self.scope / self.dataset / self.task
        if self.task == CLASSIFICATION:
            d = d / f"p{self.theta_percentile}" / ("balanced" if self.balanced else "unbalanced")
        return d / self.model

    def paths(self, run_dir: Path) -> UnitPaths:
        d = self.directory(run_dir)
        if self.stage == HOLDOUT:
            return UnitPaths(
                metrics=d / "metrics.json",
                predictions=d / "predictions.parquet",
                model=d / "model.joblib",
            )
        return UnitPaths(
            metrics=d / f"fold{self.fold}.json",
            predictions=d / f"fold{self.fold}.parquet",
            model=None,
        )


class _UnitResult(Unit):
    """Fields shared by every saved result; the unit fields come first, flat in the JSON."""

    hash: str  # see runner.unit_hash: a different hash means the result is stale
    params: dict[str, Any]  # hyperparameters actually passed to the estimator
    n_train: int
    n_test: int
    fit_seconds: float
    feature_names: list[str] | None = None  # holdout only: column order expected by the model


class RegressionResult(_UnitResult):
    task: Literal["regression"] = "regression"
    pred_clip_max: float  # predictions are clipped to [0, max(y_train)]
    metrics: RegressionMetrics


class ClassificationResult(_UnitResult):
    task: Literal["classification"] = "classification"
    theta: float  # θ in days, computed on the training part
    train_slow_rate: float
    metrics: ClassificationMetrics


UnitResult = Annotated[RegressionResult | ClassificationResult, Field(discriminator="task")]
UNIT_RESULT: TypeAdapter[RegressionResult | ClassificationResult] = TypeAdapter(UnitResult)


class Manifest(BaseModel):
    """What produced a run: written to manifests/<timestamp>.json at every launch."""

    timestamp: datetime
    argv: list[str]
    git_commit: str | None
    git_dirty: bool
    python: str
    packages: dict[str, str]
    config: Config
