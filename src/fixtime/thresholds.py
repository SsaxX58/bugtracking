"""Fast/slow threshold θ and binary labels."""

from __future__ import annotations

import numpy as np


def compute_theta(y_train, percentile: float) -> float:
    """θ is the given percentile of the training target only, to avoid leakage."""
    return float(np.percentile(np.asarray(y_train, dtype=float), percentile))


def make_labels(y, theta: float) -> np.ndarray:
    """1 = slow (fixing time at or above θ), 0 = fast (below θ)."""
    return (np.asarray(y, dtype=float) >= theta).astype(int)
