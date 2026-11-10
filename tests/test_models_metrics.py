import numpy as np
import pytest

from fixtime.metrics import classification_metrics, regression_metrics
from fixtime.models import build_classifier, build_regressor

REGRESSORS = ["dummy", "ridge", "random_forest", "xgboost", "lightgbm"]
CLASSIFIERS = ["dummy", "logreg", "random_forest", "xgboost", "lightgbm"]


@pytest.fixture
def xy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 3))
    X[::17, 1] = np.nan
    y = np.expm1(np.clip(2 + 1.5 * X[:, 0] + rng.normal(scale=0.3, size=400), 0, None))
    return X, y


def small(params):
    if "n_estimators" in type(params).model_fields:
        return params.model_copy(update={"n_estimators": 20})
    return params


@pytest.mark.parametrize("name", REGRESSORS)
def test_regressors_predict_in_days(synthetic_cfg, xy, name):
    X, y = xy
    model = build_regressor(small(synthetic_cfg.model_params("regression", name)), seed=0)
    model.fit(X, y)
    pred = model.predict(X)
    assert pred.shape == y.shape
    assert np.isfinite(pred).all()
    if name == "dummy":
        assert np.allclose(pred, np.expm1(np.median(np.log1p(y))))


@pytest.mark.parametrize("name", CLASSIFIERS)
@pytest.mark.parametrize("balanced", [False, True])
def test_classifiers_fit(synthetic_cfg, xy, name, balanced):
    X, y = xy
    labels = (y >= np.percentile(y, 90)).astype(int)
    params = small(synthetic_cfg.model_params("classification", name))
    model = build_classifier(params, seed=0, balanced=balanced, y_train=labels)
    model.fit(X, labels)
    proba = model.predict_proba(X)[:, 1]
    assert proba.shape == labels.shape


def test_balanced_predicts_more_minority(synthetic_cfg, xy):
    X, y = xy
    labels = (y >= np.percentile(y, 90)).astype(int)
    params = synthetic_cfg.models.classification.logreg
    preds = {}
    for balanced in (False, True):
        m = build_classifier(params, seed=0, balanced=balanced, y_train=labels)
        preds[balanced] = m.fit(X, labels).predict(X).sum()
    assert preds[True] > preds[False]


def test_xgboost_scale_pos_weight(synthetic_cfg):
    labels = np.array([1] * 10 + [0] * 90)
    params = synthetic_cfg.models.classification.xgboost
    m = build_classifier(params, seed=0, balanced=True, y_train=labels)
    assert m.named_steps["model"].get_params()["scale_pos_weight"] == pytest.approx(9.0)


def test_regression_metrics_perfect():
    y = np.array([0.0, 1.0, 10.0, 100.0])
    m = regression_metrics(y, y)
    assert m.mae == 0 and m.rmse == 0 and m.r2 == 1 and m.mae_log == 0


def test_classification_metrics_dummy():
    y = np.array([0] * 90 + [1] * 10)
    m = classification_metrics(y, np.zeros(100), np.full(100, 0.1))
    assert m.mcc == 0
    assert m.roc_auc == 0.5
    assert m.tp == 0 and m.fn == 10
    assert m.slow_rate == pytest.approx(0.1)


@pytest.mark.filterwarnings("ignore:A single label was found")
def test_auc_undefined_with_one_class():
    m = classification_metrics(np.zeros(5), np.zeros(5), np.full(5, 0.3))
    assert m.roc_auc is None and m.pr_auc is None
    assert '"roc_auc":null' in m.model_dump_json()
