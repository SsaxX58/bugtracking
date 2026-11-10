from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from fixtime.config import (
    LightGBMParams,
    RandomForestParams,
    RidgeParams,
    XGBoostParams,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"
V1_CONFIG = ROOT / "configs" / "v1.yaml"


def _load_with(tmp_path, **changes):
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    raw["base_dir"] = str(DEFAULT_CONFIG.parent.parent)
    for dotted, value in changes.items():
        node = raw
        *parents, last = dotted.split("__")
        for key in parents:
            node = node[key]
        if value is None:
            del node[last]
        else:
            node[last] = value
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False))  # keep the dataset order
    return load_config(path)


def test_shipped_configs_are_valid():
    for path in (DEFAULT_CONFIG, V1_CONFIG):
        cfg = load_config(path)
        assert cfg.base_dir.is_absolute() and cfg.output_dir.is_absolute()
        assert cfg.split_reference in cfg.datasets


def test_typed_sections():
    cfg = load_config(DEFAULT_CONFIG)
    assert cfg.split.n_folds == 5
    assert isinstance(cfg.models.regression.ridge, RidgeParams)
    assert isinstance(cfg.model_params("classification", "random_forest"), RandomForestParams)
    assert isinstance(cfg.model_params("regression", "xgboost"), XGBoostParams)
    assert set(cfg.models.classification.enabled()) == {
        "dummy", "logreg", "random_forest", "xgboost", "lightgbm"
    }


def test_kwargs_only_contain_values_from_the_file():
    assert LightGBMParams(n_estimators=10).kwargs() == {"n_estimators": 10}


def test_split_reference_defaults_to_first_dataset(tmp_path):
    cfg = _load_with(tmp_path, split_reference=None)
    assert cfg.split_reference == "T0"


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"models__regression__ridge": {"alpah": 1.0}}, "alpah"),           # typo in a param
        ({"models__classification__svm": {}}, "svm"),                       # unknown model
        ({"thresholds": [50, 150]}, "less than 100"),                       # bad percentile
        ({"split__holdout_size": 1.5}, "holdout_size"),
        ({"split_reference": "T99"}, "split_reference"),
        ({"projects": ["Eclipse", "aggregated"]}, "reserved"),
        ({"models__regression__xgboost": {"subsample": 2}}, "subsample"),
        ({"seeed": 1}, "seeed"),                                            # unknown top key
    ],
)
def test_invalid_config_fails_at_load(tmp_path, changes, message):
    with pytest.raises(ValidationError, match=message):
        _load_with(tmp_path, **changes)


def test_assignment_is_validated():
    cfg = load_config(DEFAULT_CONFIG)
    with pytest.raises(ValidationError):
        cfg.n_jobs = "many"
