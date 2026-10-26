"""Load one (scope, dataset) table into features, target and ids."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pandas as pd

from fixtime.config import AGGREGATED, Config


@dataclass
class Dataset:
    scope: str
    name: str
    X: pd.DataFrame  # indexed by id
    y: pd.Series  # indexed by id
    project: pd.Series | None  # project of each row, only for the aggregated scope
    file_hash: str

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


def load_dataset(cfg: Config, scope: str, dataset: str, limit: int | None = None) -> Dataset:
    path = cfg.dataset_path(scope, dataset)
    if not path.exists():
        raise FileNotFoundError(f"{scope}/{dataset}: file not found: {path}")
    df = pd.read_csv(path)

    id_col = cfg.id_col(scope)
    for col in (id_col, cfg.target):
        if col not in df.columns:
            raise ValueError(f"{path}: missing column {col!r}")
    if df[id_col].duplicated().any():
        raise ValueError(f"{path}: duplicated ids in {id_col!r}")
    if df[cfg.target].isna().any() or (df[cfg.target] < 0).any():
        raise ValueError(f"{path}: target {cfg.target!r} has missing or negative values")

    if limit is not None and limit < len(df):
        df = df.sample(n=limit, random_state=cfg.seed)

    df = df.set_index(id_col, drop=False)
    df.index.name = "id"
    project = df[cfg.project_column] if scope == AGGREGATED else None

    excluded = set(cfg.drop_columns) | {cfg.target, id_col}
    X = df[[c for c in df.columns if c not in excluded]]
    non_numeric = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    if non_numeric:
        raise ValueError(f"{path}: non-numeric feature columns {non_numeric}")

    return Dataset(
        scope=scope,
        name=dataset,
        X=X.astype("float64"),
        y=df[cfg.target].astype("float64"),
        project=project,
        file_hash=file_sha256(path),
    )
