"""Load the YAML configuration and resolve scopes and dataset paths."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

AGGREGATED = "aggregated"


@dataclass
class Config:
    base_dir: Path
    output_dir: Path
    run_name: str
    seed: int
    target: str
    projects: list[str]
    aggregated: bool
    id_column: dict[str, str]
    project_column: str
    drop_columns: list[str]
    datasets: dict[str, dict[str, str]]
    thresholds: list[int]
    split: dict[str, Any]
    n_jobs: int
    models: dict[str, dict[str, dict[str, Any]]]
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def scopes(self) -> list[str]:
        return self.projects + ([AGGREGATED] if self.aggregated else [])

    @property
    def run_dir(self) -> Path:
        return self.output_dir / self.run_name

    def id_col(self, scope: str) -> str:
        return self.id_column[AGGREGATED if scope == AGGREGATED else "project"]

    def dataset_path(self, scope: str, dataset: str) -> Path:
        entry = self.datasets[dataset]
        if scope == AGGREGATED:
            rel = entry["aggregated_path"]
        else:
            rel = entry["project_path"].format(project=scope)
        return self.base_dir / rel


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    base_dir = (path.parent / raw.get("base_dir", ".")).resolve()
    return Config(
        base_dir=base_dir,
        output_dir=base_dir / raw["output_dir"],
        run_name=raw["run_name"],
        seed=int(raw["seed"]),
        target=raw["target"],
        projects=list(raw["projects"]),
        aggregated=bool(raw.get("aggregated", False)),
        id_column=dict(raw["id_column"]),
        project_column=raw["project_column"],
        drop_columns=list(raw.get("drop_columns", [])),
        datasets=dict(raw["datasets"]),
        thresholds=[int(p) for p in raw["thresholds"]],
        split=dict(raw["split"]),
        n_jobs=int(raw.get("n_jobs", 1)),
        models=dict(raw["models"]),
        raw=raw,
    )
