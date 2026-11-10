from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fixtime.config import load_config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
V1_CONFIG = ROOT / "configs" / "v1.yaml"


def _real(path, dataset):
    cfg = load_config(path)
    if not cfg.dataset_path("W3C", dataset).exists():
        pytest.skip("real datasets not available")
    return cfg


@pytest.fixture
def real_cfg():
    return _real(DEFAULT_CONFIG, "T0")


@pytest.fixture
def real_v1_cfg():
    return _real(V1_CONFIG, "D6")


def _synthetic_table(rng, n):
    x1 = rng.normal(size=n)
    x2 = rng.exponential(size=n)
    y = np.round(np.expm1(np.clip(2 + 1.5 * x1 + 0.5 * x2 + rng.normal(scale=0.5, size=n), 0, None)))
    return pd.DataFrame({"bug_id": np.arange(n), "fixing_time": y, "x1": x1, "x2": x2})


@pytest.fixture
def synthetic_cfg(tmp_path):
    """Two tiny projects with two snapshots each, and a config pointing to them.

    T0 has every bug; T24 only the bugs with fixing time >= 1 and an extra feature, like
    the real snapshot datasets. The aggregated scope is built by concatenation.
    """
    rng = np.random.default_rng(0)
    data = tmp_path / "data"
    data.mkdir()
    for project, n in [("A", 300), ("B", 200)]:
        df = _synthetic_table(rng, n)
        df.to_csv(data / f"{project}_T0.csv", index=False)
        later = df[df["fixing_time"] >= 1].assign(x3=lambda d: d["x1"] * 2)
        later.to_csv(data / f"{project}_T24.csv", index=False)

    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    raw.update(
        base_dir=".",
        output_dir="outputs",
        run_name="test",
        projects=["A", "B"],
        datasets={
            "T0": {"project_path": "data/{project}_T0.csv"},
            "T24": {"project_path": "data/{project}_T24.csv"},
        },
        split_reference="T0",
        n_jobs=1,
    )
    # keep the models small so tests stay fast
    for task in raw["models"].values():
        for params in task.values():
            if "n_estimators" in params:
                params["n_estimators"] = 20
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return load_config(path)
