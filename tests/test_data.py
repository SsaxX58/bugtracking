import pytest

from fixtime.config import AGGREGATED
from fixtime.data import load_dataset


def test_real_project_dataset(real_cfg):
    ds = load_dataset(real_cfg, "W3C", "D6")
    assert len(ds.X) == 408
    assert ds.X.shape[1] == 45
    assert "bug_id" not in ds.X.columns and "fixing_time" not in ds.X.columns
    assert ds.X.index.is_unique
    assert ds.project is None


def test_real_aggregated_dataset(real_cfg):
    ds = load_dataset(real_cfg, AGGREGATED, "D6")
    assert ds.X.shape[1] == 45
    assert not {"source_row", "project", "row_uid", "bug_id"} & set(ds.X.columns)
    assert ds.X.index.is_unique
    assert set(ds.project.unique()) == set(real_cfg.projects)


def test_synthetic_limit_is_deterministic(synthetic_cfg):
    a = load_dataset(synthetic_cfg, "A", "D6", limit=50)
    b = load_dataset(synthetic_cfg, "A", "D6", limit=50)
    assert len(a.X) == 50
    assert list(a.X.index) == list(b.X.index)


def test_missing_file_raises(synthetic_cfg):
    with pytest.raises(FileNotFoundError):
        load_dataset(synthetic_cfg, "C", "D6")
