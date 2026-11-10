"""Holdout split and CV folds, computed once per scope and shared by all its datasets."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from fixtime.config import Config
from fixtime.data import Dataset

HOLDOUT = "holdout"


@dataclass
class Split:
    scope: str
    holdout: list
    folds: list[list]  # test ids of each CV fold, all drawn from the non-holdout part
    ids_hash: str

    @property
    def all_ids(self) -> list:
        return self.holdout + self.development

    @property
    def development(self) -> list:
        return [i for fold in self.folds for i in fold]

    @property
    def fold_names(self) -> list:
        return list(range(len(self.folds))) + [HOLDOUT]

    def train_test(self, fold) -> tuple[list, list]:
        """Train and test ids for a CV fold index or for the holdout stage."""
        if fold == HOLDOUT:
            return self.development, self.holdout
        test = self.folds[fold]
        train = [i for k, f in enumerate(self.folds) if k != fold for i in f]
        return train, test

    def to_json(self) -> dict:
        return {
            "scope": self.scope,
            "ids_hash": self.ids_hash,
            "holdout": self.holdout,
            "folds": self.folds,
        }


def ids_hash(ids) -> str:
    joined = "\n".join(sorted(str(i) for i in ids))
    return hashlib.sha256(joined.encode()).hexdigest()


def stratification_labels(ds: Dataset, n_bins: int) -> pd.Series:
    # rank first so that the many tied values (e.g. zeros) still give equal-sized bins
    bins = pd.qcut(ds.y.rank(method="first"), n_bins, labels=False).astype(str)
    if ds.project is not None:
        bins = ds.project.astype(str) + "|" + bins
    return bins


def make_split(cfg: Config, ds: Dataset) -> Split:
    strata = stratification_labels(ds, cfg.split["n_bins"])
    ids = np.asarray(ds.y.index)
    dev, holdout = train_test_split(
        ids,
        test_size=cfg.split["holdout_size"],
        stratify=strata.to_numpy(),
        random_state=cfg.seed,
    )
    skf = StratifiedKFold(n_splits=cfg.split["n_folds"], shuffle=True, random_state=cfg.seed)
    folds = [dev[test].tolist() for _, test in skf.split(dev, strata.loc[dev].to_numpy())]
    return Split(
        scope=ds.scope,
        holdout=sorted(holdout.tolist(), key=str),
        folds=[sorted(f, key=str) for f in folds],
        ids_hash=ids_hash(ids),
    )


def split_path(cfg: Config, scope: str) -> Path:
    return cfg.run_dir / "splits" / f"{scope}.json"


def load_or_create_split(cfg: Config, ds: Dataset) -> Split:
    """Reuse the saved split of the scope; fail loudly if the reference ids have changed."""
    path = split_path(cfg, ds.scope)
    if path.exists():
        data = json.loads(path.read_text())
        split = Split(data["scope"], data["holdout"], data["folds"], data["ids_hash"])
        if split.ids_hash != ids_hash(ds.y.index):
            raise ValueError(
                f"{ds.scope}/{ds.name}: ids differ from the saved split {path}. "
                "The reference dataset of a scope must not change within a run."
            )
        return split
    split = make_split(cfg, ds)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(split.to_json()))
    return split
