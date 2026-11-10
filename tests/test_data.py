import pytest

from fixtime.config import AGGREGATED
from fixtime.data import load_dataset

LEAKY = {"resolved_date", "fixing_time_days", "fixing_time_log", "residual_fixing_time_days"}


def test_real_v2_snapshots(real_cfg):
    t0 = load_dataset(real_cfg, "W3C", "T0")
    assert len(t0.X) == 408
    assert t0.X.shape[1] == 28
    for name in real_cfg.datasets:
        ds = load_dataset(real_cfg, "W3C", name)
        assert set(ds.y.index) <= set(t0.y.index)
        assert not LEAKY & set(ds.X.columns)
        assert (ds.y == t0.y.loc[ds.y.index]).all()
    assert len(load_dataset(real_cfg, "W3C", "T24").X) < 408


def test_real_v2_aggregated_is_concatenated(real_cfg):
    ds = load_dataset(real_cfg, AGGREGATED, "T0")
    assert set(ds.project.unique()) == set(real_cfg.projects)
    assert ds.X.index.is_unique
    assert ds.X.index[0].startswith(ds.project.iloc[0] + ":")
    assert not {"project", "row_uid", "bug_id"} & set(ds.X.columns)


def test_real_v1_aggregated_file(real_v1_cfg):
    ds = load_dataset(real_v1_cfg, AGGREGATED, "D6")
    assert ds.X.shape[1] == 45
    assert not {"source_row", "project", "row_uid", "bug_id"} & set(ds.X.columns)
    assert set(ds.project.unique()) == set(real_v1_cfg.projects)


def test_synthetic_limit_is_deterministic(synthetic_cfg):
    a = load_dataset(synthetic_cfg, "A", "T0", limit=50)
    b = load_dataset(synthetic_cfg, "A", "T0", limit=50)
    assert len(a.X) == 50
    assert list(a.X.index) == list(b.X.index)


def test_keep_ids(synthetic_cfg):
    ds = load_dataset(synthetic_cfg, "A", "T24", keep_ids=[0, 1, 2, 3])
    assert set(ds.y.index) <= {0, 1, 2, 3}


def test_synthetic_aggregated_concatenation(synthetic_cfg):
    ds = load_dataset(synthetic_cfg, AGGREGATED, "T24")
    a = load_dataset(synthetic_cfg, "A", "T24")
    b = load_dataset(synthetic_cfg, "B", "T24")
    assert len(ds.X) == len(a.X) + len(b.X)
    assert list(ds.X.columns) == ["x1", "x2", "x3"]


def test_missing_file_raises(synthetic_cfg):
    with pytest.raises(FileNotFoundError):
        load_dataset(synthetic_cfg, "C", "T0")
