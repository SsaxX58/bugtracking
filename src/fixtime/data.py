"""Load one (scope, dataset) table into features, target and ids."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from functools import cache
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator

from fixtime.config import AGGREGATED, Config


class Dataset(BaseModel):
    """Features, target and ids of one (scope, dataset), all indexed by bug id."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    scope: str
    name: str
    X: pd.DataFrame
    y: pd.Series
    project: pd.Series | None  # project of each row, only for the aggregated scope
    file_hash: str

    @model_validator(mode="after")
    def _aligned(self) -> Dataset:
        if not self.X.index.equals(self.y.index):
            raise ValueError(f"{self.scope}/{self.name}: features and target ids differ")
        if not self.X.index.is_unique:
            raise ValueError(f"{self.scope}/{self.name}: duplicated ids")
        return self

    @property
    def feature_names(self) -> list[str]:
        return list(self.X.columns)


@cache
def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_table(cfg: Config, scope: str, dataset: str) -> tuple[pd.DataFrame, str, str]:
    """Return the raw table, a label for error messages and a hash of the source file(s)."""
    path = cfg.dataset_path(scope, dataset)
    if path is not None:
        if not path.exists():
            raise FileNotFoundError(f"{scope}/{dataset}: file not found: {path}")
        return pd.read_csv(path), str(path), file_sha256(path)

    # no aggregated file: pool the per-project tables, with ids made unique as "Project:bug_id"
    parts, hashes = [], []
    for project in cfg.projects:
        df, _, h = _read_table(cfg, project, dataset)
        df[cfg.project_column] = project
        df[cfg.id_col(AGGREGATED)] = project + ":" + df[cfg.id_col(project)].astype(str)
        parts.append(df)
        hashes.append(h)
    columns = [set(df.columns) for df in parts]
    if any(c != columns[0] for c in columns):
        raise ValueError(f"{scope}/{dataset}: projects have different columns, cannot aggregate")
    combined = hashlib.sha256("".join(hashes).encode()).hexdigest()
    return pd.concat(parts, ignore_index=True), f"{scope}/{dataset} (concatenated)", combined


def load_dataset(
    cfg: Config,
    scope: str,
    dataset: str,
    limit: int | None = None,
    keep_ids: Iterable[int | str] | None = None,
) -> Dataset:
    """Load features, target and ids. `limit` subsamples rows; `keep_ids` keeps only those ids."""
    df, source, file_hash = _read_table(cfg, scope, dataset)

    id_col = cfg.id_col(scope)
    for col in (id_col, cfg.target):
        if col not in df.columns:
            raise ValueError(f"{source}: missing column {col!r}")
    if df[id_col].duplicated().any():
        raise ValueError(f"{source}: duplicated ids in {id_col!r}")
    if df[cfg.target].isna().any() or (df[cfg.target] < 0).any():
        raise ValueError(f"{source}: target {cfg.target!r} has missing or negative values")

    if keep_ids is not None:
        df = df[df[id_col].isin(set(keep_ids))]
    elif limit is not None and limit < len(df):
        df = df.sample(n=limit, random_state=cfg.seed)

    df = df.set_index(id_col, drop=False)
    df.index.name = "id"
    project = df[cfg.project_column] if scope == AGGREGATED else None

    excluded = set(cfg.drop_columns) | {cfg.target, id_col}
    X = df[[c for c in df.columns if c not in excluded]]
    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        raise ValueError(f"{source}: non-numeric feature columns {non_numeric}")

    return Dataset(
        scope=scope,
        name=dataset,
        X=X.astype("float64"),
        y=df[cfg.target].astype("float64"),
        project=project,
        file_hash=file_hash,
    )
