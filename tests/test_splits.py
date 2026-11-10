import json

import numpy as np
import pytest

from fixtime.config import AGGREGATED
from fixtime.data import load_dataset
from fixtime.splits import HOLDOUT, load_or_create_split, make_split, split_path
from fixtime.thresholds import compute_theta, make_labels


def test_split_partitions_ids(synthetic_cfg):
    ds = load_dataset(synthetic_cfg, "A", "T0")
    split = make_split(synthetic_cfg, ds)
    all_ids = split.holdout + split.development
    assert sorted(all_ids) == sorted(ds.y.index)
    assert len(set(all_ids)) == len(all_ids)
    assert len(split.holdout) == pytest.approx(0.2 * len(ds.y), abs=1)
    assert len(split.folds) == 5


def test_train_test_are_disjoint(synthetic_cfg):
    ds = load_dataset(synthetic_cfg, "A", "T0")
    split = make_split(synthetic_cfg, ds)
    for fold in split.fold_names:
        train, test = split.train_test(fold)
        assert not set(train) & set(test)
        assert not set(train) & set(split.holdout)
    train, test = split.train_test(HOLDOUT)
    assert test == split.holdout


def test_split_is_reproducible(synthetic_cfg):
    ds = load_dataset(synthetic_cfg, "A", "T0")
    assert make_split(synthetic_cfg, ds).to_json() == make_split(synthetic_cfg, ds).to_json()


def test_aggregated_split_is_stratified_by_project(synthetic_cfg):
    ds = load_dataset(synthetic_cfg, AGGREGATED, "T0")
    split = make_split(synthetic_cfg, ds)
    share = ds.project.loc[split.holdout].value_counts(normalize=True)
    overall = ds.project.value_counts(normalize=True)
    assert share["A"] == pytest.approx(overall["A"], abs=0.02)


def test_saved_split_is_reused_and_checked(synthetic_cfg):
    ds = load_dataset(synthetic_cfg, "A", "T0")
    first = load_or_create_split(synthetic_cfg, ds)
    assert split_path(synthetic_cfg, "A").exists()
    assert load_or_create_split(synthetic_cfg, ds).to_json() == first.to_json()

    data = json.loads(split_path(synthetic_cfg, "A").read_text())
    data["ids_hash"] = "changed"
    split_path(synthetic_cfg, "A").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="ids differ"):
        load_or_create_split(synthetic_cfg, ds)


def test_theta_and_labels():
    y_train = np.arange(100)
    theta = compute_theta(y_train, 75)
    assert theta == pytest.approx(74.25)
    labels = make_labels([0, 74, 75, 1000], theta)
    assert labels.tolist() == [0, 0, 1, 1]
