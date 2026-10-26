from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from fixtime.config import load_config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


@pytest.fixture
def real_cfg():
    cfg = load_config(DEFAULT_CONFIG)
    if not cfg.dataset_path("W3C", "D6").exists():
        pytest.skip("real datasets not available")
    return cfg


def _synthetic_table(rng, n, project=None):
    x1 = rng.normal(size=n)
    x2 = rng.exponential(size=n)
    y = np.round(np.expm1(np.clip(2 + 1.5 * x1 + 0.5 * x2 + rng.normal(scale=0.5, size=n), 0, None)))
    df = pd.DataFrame({"bug_id": np.arange(n), "fixing_time": y, "x1": x1, "x2": x2})
    if project is not None:
        df["project"] = project
        df["row_uid"] = [f"{project}:{i}" for i in range(n)]
    return df


@pytest.fixture
def synthetic_cfg(tmp_path):
    """Two tiny projects plus their aggregation, with a config pointing to them."""
    rng = np.random.default_rng(0)
    data = tmp_path / "data"
    data.mkdir()
    parts = []
    for project, n in [("A", 300), ("B", 200)]:
        df = _synthetic_table(rng, n, project)
        df.drop(columns=["project", "row_uid"]).to_csv(data / f"features{project}.csv", index=False)
        parts.append(df)
    pd.concat(parts).to_csv(data / "aggregated.csv", index=False)

    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    raw.update(
        base_dir=".",
        output_dir="outputs",
        run_name="test",
        projects=["A", "B"],
        datasets={
            "D6": {
                "project_path": "data/features{project}.csv",
                "aggregated_path": "data/aggregated.csv",
            }
        },
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
