import pytest
from pydantic import ValidationError

from fixtime.metrics import classification_metrics, regression_metrics
from fixtime.records import UNIT_RESULT, ClassificationResult, RegressionResult, Unit


def test_unit_validation():
    Unit(scope="A", dataset="T0", fold=0, task="regression", model="ridge")
    Unit(scope="A", dataset="T0", fold="holdout", task="classification", model="ridge",
         theta_percentile=75, balanced=True)
    with pytest.raises(ValidationError, match="needs theta_percentile"):
        Unit(scope="A", dataset="T0", fold=0, task="classification", model="ridge")
    with pytest.raises(ValidationError, match="has no theta_percentile"):
        Unit(scope="A", dataset="T0", fold=0, task="regression", model="ridge", balanced=True)
    with pytest.raises(ValidationError):
        Unit(scope="A", dataset="T0", fold="test", task="regression", model="ridge")
    with pytest.raises(ValidationError):
        Unit(scope="A", dataset="T0", fold=0, task="clustering", model="ridge")


def test_unit_is_hashable_and_frozen():
    u = Unit(scope="A", dataset="T0", fold=0, task="regression", model="ridge")
    assert {u: 1}[Unit(scope="A", dataset="T0", fold=0, task="regression", model="ridge")] == 1
    with pytest.raises(ValidationError):
        u.model = "lightgbm"


def test_results_round_trip_through_json():
    base = {"scope": "A", "dataset": "T0", "fold": 1, "model": "ridge", "hash": "h",
            "params": {"alpha": 1.0}, "n_train": 8, "n_test": 2, "fit_seconds": 0.1}
    reg = RegressionResult(**base, task="regression", pred_clip_max=10.0,
                           metrics=regression_metrics([1.0, 2.0], [1.0, 3.0]))
    clf = ClassificationResult(**{**base, "model": "logreg"}, task="classification",
                               theta_percentile=75, balanced=False, theta=3.0,
                               train_slow_rate=0.25,
                               metrics=classification_metrics([0, 1], [0, 1], [0.2, 0.9]))
    for result in (reg, clf):
        parsed = UNIT_RESULT.validate_json(result.model_dump_json())
        assert type(parsed) is type(result)
        assert parsed == result
