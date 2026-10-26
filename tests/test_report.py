import pandas as pd

from fixtime.report import build_report
from fixtime.runner import run


def test_report_tables(synthetic_cfg):
    assert run(synthetic_cfg, scopes=["A", "B"], models=["dummy", "ridge", "logreg"],
               log=lambda _: None) == []
    names = {p.name for p in build_report(synthetic_cfg)}
    assert names == {
        "cv_regression_folds.csv", "cv_regression_summary.csv", "holdout_regression.csv",
        "cv_classification_folds.csv", "cv_classification_summary.csv",
        "holdout_classification.csv",
    }
    results = synthetic_cfg.run_dir / "results"

    reg = pd.read_csv(results / "cv_regression_summary.csv")
    assert len(reg) == 2 * 2  # scopes x regressors
    assert (reg["n_folds"] == 5).all()
    ridge = reg[(reg["scope"] == "A") & (reg["model"] == "ridge")].iloc[0]
    dummy = reg[(reg["scope"] == "A") & (reg["model"] == "dummy")].iloc[0]
    assert ridge["mae_log_mean"] < dummy["mae_log_mean"]

    clf = pd.read_csv(results / "cv_classification_summary.csv")
    assert len(clf) == 2 * 3 * 2 * 2  # scopes x thresholds x balance x classifiers
    logreg = clf[(clf["model"] == "logreg")]
    assert (logreg["mcc_mean"] > 0).all()

    hold = pd.read_csv(results / "holdout_classification.csv")
    assert len(hold) == len(clf)
