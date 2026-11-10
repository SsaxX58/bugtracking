"""Typed configuration: load the YAML file and validate it with pydantic.

Every key is checked when the file is loaded, so a typo (an unknown key, a misspelled
hyperparameter, a threshold outside 1..99) fails immediately instead of after hours of fits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

AGGREGATED = "aggregated"
Task = Literal["regression", "classification"]


class StrictModel(BaseModel):
    """Base for configuration sections: unknown keys are errors, assignments are validated."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class DatasetSource(StrictModel):
    project_path: str  # `{project}` is replaced with the project name
    aggregated_path: str | None = None  # None: pool the project files


class IdColumns(StrictModel):
    project: str
    aggregated: str


class SplitSettings(StrictModel):
    holdout_size: float = Field(gt=0, lt=1)
    n_folds: int = Field(ge=2)
    n_bins: int = Field(ge=2)  # quantile bins of the target used for stratification


# Hyperparameters. A field left unset (None) is not passed, so the library default applies
# and only the values written in the YAML reach the estimator.


class ModelParams(StrictModel):
    def kwargs(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class DummyRegressorParams(ModelParams):
    strategy: Literal["mean", "median"] | None = None


class DummyClassifierParams(ModelParams):
    strategy: Literal["prior", "most_frequent", "stratified", "uniform"] | None = None


class RidgeParams(ModelParams):
    alpha: float | None = Field(default=None, ge=0)


class LogisticRegressionParams(ModelParams):
    C: float | None = Field(default=None, gt=0)
    max_iter: int | None = Field(default=None, gt=0)


class RandomForestParams(ModelParams):
    n_estimators: int | None = Field(default=None, gt=0)
    min_samples_leaf: int | None = Field(default=None, gt=0)
    max_features: Literal["sqrt", "log2"] | int | float | None = None
    max_depth: int | None = Field(default=None, gt=0)


class XGBoostParams(ModelParams):
    n_estimators: int | None = Field(default=None, gt=0)
    learning_rate: float | None = Field(default=None, gt=0)
    max_depth: int | None = Field(default=None, gt=0)
    subsample: float | None = Field(default=None, gt=0, le=1)
    colsample_bytree: float | None = Field(default=None, gt=0, le=1)
    tree_method: Literal["hist", "approx", "exact"] | None = None


class LightGBMParams(ModelParams):
    n_estimators: int | None = Field(default=None, gt=0)
    learning_rate: float | None = Field(default=None, gt=0)
    num_leaves: int | None = Field(default=None, gt=1)
    max_depth: int | None = None
    subsample: float | None = Field(default=None, gt=0, le=1)
    subsample_freq: int | None = Field(default=None, ge=0)
    colsample_bytree: float | None = Field(default=None, gt=0, le=1)


RegressorParams = (
    DummyRegressorParams | RidgeParams | RandomForestParams | XGBoostParams | LightGBMParams
)
ClassifierParams = (
    DummyClassifierParams
    | LogisticRegressionParams
    | RandomForestParams
    | XGBoostParams
    | LightGBMParams
)


class RegressionModels(StrictModel):
    """Regressors to run: a model is enabled when its key is present in the YAML."""

    dummy: DummyRegressorParams | None = None
    ridge: RidgeParams | None = None
    random_forest: RandomForestParams | None = None
    xgboost: XGBoostParams | None = None
    lightgbm: LightGBMParams | None = None

    def enabled(self) -> dict[str, RegressorParams]:
        return {n: p for n in type(self).model_fields if (p := getattr(self, n)) is not None}


class ClassificationModels(StrictModel):
    """Classifiers to run: a model is enabled when its key is present in the YAML."""

    dummy: DummyClassifierParams | None = None
    logreg: LogisticRegressionParams | None = None
    random_forest: RandomForestParams | None = None
    xgboost: XGBoostParams | None = None
    lightgbm: LightGBMParams | None = None

    def enabled(self) -> dict[str, ClassifierParams]:
        return {n: p for n in type(self).model_fields if (p := getattr(self, n)) is not None}


class Models(StrictModel):
    regression: RegressionModels
    classification: ClassificationModels


Percentile = Annotated[int, Field(gt=0, lt=100)]


class Config(StrictModel):
    base_dir: Path  # absolute; relative paths in the file are resolved by load_config
    output_dir: Path
    run_name: str = Field(min_length=1)
    seed: int
    target: str
    projects: list[str] = Field(min_length=1)
    aggregated: bool = False
    id_column: IdColumns
    project_column: str
    drop_columns: list[str] = []
    datasets: dict[str, DatasetSource] = Field(min_length=1)
    split_reference: str  # dataset whose ids define the split of each scope
    thresholds: list[Percentile] = Field(min_length=1)
    split: SplitSettings
    n_jobs: int = 1
    models: Models

    @model_validator(mode="before")
    @classmethod
    def _default_split_reference(cls, data: Any) -> Any:
        # without an explicit reference, the first dataset defines the split
        if isinstance(data, dict) and data.get("datasets") and not data.get("split_reference"):
            data = {**data, "split_reference": next(iter(data["datasets"]))}
        return data

    @model_validator(mode="after")
    def _check_consistency(self) -> Config:
        if self.split_reference not in self.datasets:
            raise ValueError(f"split_reference {self.split_reference!r} is not in datasets")
        if AGGREGATED in self.projects:
            raise ValueError(f"{AGGREGATED!r} is reserved for the pooled scope")
        if len(set(self.thresholds)) != len(self.thresholds):
            raise ValueError("thresholds must be unique")
        return self

    @property
    def scopes(self) -> list[str]:
        return self.projects + ([AGGREGATED] if self.aggregated else [])

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.run_name

    def id_col(self, scope: str) -> str:
        return self.id_column.aggregated if scope == AGGREGATED else self.id_column.project

    def dataset_path(self, scope: str, dataset: str) -> Path | None:
        """Path of the CSV, or None for an aggregated scope built by concatenating projects."""
        source = self.datasets[dataset]
        if scope == AGGREGATED:
            return self.base_dir / source.aggregated_path if source.aggregated_path else None
        return self.base_dir / source.project_path.format(project=scope)

    def model_params(self, task: Task, name: str) -> RegressorParams | ClassifierParams:
        models = self.models.regression if task == "regression" else self.models.classification
        return models.enabled()[name]


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    base_dir = (path.parent / raw.pop("base_dir", ".")).resolve()
    output_dir = base_dir / raw.pop("output_dir")
    return Config.model_validate({**raw, "base_dir": base_dir, "output_dir": output_dir})
