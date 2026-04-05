"""
Uplift evaluation metrics and Optuna utilities.
"""

from __future__ import annotations

import time

import numpy as np


# ---------------------------------------------------------------------------
# Uplift metrics
# ---------------------------------------------------------------------------

def qini_coefficient(y, w, uplift_pred) -> float:
    """Normalized Qini coefficient (vectorized).

    Ranks customers by predicted uplift (descending), computes the area
    under the Qini curve, and subtracts the random-model baseline.
    Higher is better; used as the primary Optuna objective.
    """
    order = np.argsort(-np.asarray(uplift_pred).ravel())
    y_s = np.asarray(y, dtype=float)[order]
    w_s = np.asarray(w, dtype=float)[order]

    n_t = w_s.sum()
    n_c = len(w_s) - n_t
    if n_t == 0 or n_c == 0:
        return 0.0

    cum_t_y = np.cumsum(y_s * w_s) / n_t
    cum_c_y = np.cumsum(y_s * (1 - w_s)) / n_c
    qini = cum_t_y - cum_c_y

    auqc = np.trapz(qini) / len(qini)
    return float(auqc - qini[-1] / 2)  # subtract random baseline


def uplift_at_k(y, w, uplift_pred, k: float = 0.3) -> float:
    """Actual ATE in the top-k fraction ranked by predicted uplift."""
    pred = np.asarray(uplift_pred).ravel()
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    threshold = np.quantile(pred, 1 - k)
    top = pred >= threshold
    t_top = y[top & (w == 1)]
    c_top = y[top & (w == 0)]
    if len(t_top) == 0 or len(c_top) == 0:
        return 0.0
    return float(t_top.mean() - c_top.mean())


# ---------------------------------------------------------------------------
# Optuna callback
# ---------------------------------------------------------------------------

class TrialLogger:
    """Prints progress every *every* trials with elapsed time.

    Compatible with Optuna's callback interface:
        study.optimize(..., callbacks=[TrialLogger("MyModel")])
    """

    def __init__(self, name: str, every: int = 5):
        self.name = name
        self.every = every
        self.t0 = time.time()

    def __call__(self, study, trial) -> None:
        n = trial.number + 1
        if n == 1 or n % self.every == 0:
            dt = time.time() - self.t0
            print(
                f"  [{self.name}] trial {n:>3d} | "
                f"val={trial.value:+.6f} | best={study.best_value:+.6f} | "
                f"{dt:.0f}s elapsed"
            )
