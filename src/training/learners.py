"""
Uplift meta-learner wrappers with a uniform CATE interface.

Every wrapper exposes:
    predict(X) -> np.ndarray of shape (N, 1)   (CATE per customer)

X may be a numpy array or a pandas DataFrame; wrappers handle both.
"""

from __future__ import annotations

import numpy as np


class SLearnerWrapper:
    """Single model with treatment as a feature.

    CATE = f(X, T=1) - f(X, T=0)
    """

    def __init__(self, base_model, feature_cols: list[str]):
        self.base_model = base_model
        self.feature_cols = feature_cols

    def predict(self, X) -> np.ndarray:
        X_arr = X if isinstance(X, np.ndarray) else X[self.feature_cols].values
        X1 = np.column_stack([X_arr, np.ones(len(X_arr))])
        X0 = np.column_stack([X_arr, np.zeros(len(X_arr))])
        return (self.base_model.predict(X1) - self.base_model.predict(X0)).reshape(-1, 1)


class XLearnerWrapper:
    """Two-stage X-Learner.

    Stage 1 — separate outcome models on treatment / control splits.
    Stage 2 — CATE models on imputed treatment effects.
    Final    — propensity-weighted combination of the two CATE models.
    """

    def __init__(
        self,
        model_t,
        model_c,
        cate_model_t,
        cate_model_c,
        feature_cols: list[str],
        propensity: float,
    ):
        self.model_t = model_t
        self.model_c = model_c
        self.cate_model_t = cate_model_t
        self.cate_model_c = cate_model_c
        self.feature_cols = feature_cols
        self.propensity = propensity

    def predict(self, X) -> np.ndarray:
        X_arr = X if isinstance(X, np.ndarray) else X[self.feature_cols].values
        tau_t = self.cate_model_t.predict(X_arr)
        tau_c = self.cate_model_c.predict(X_arr)
        g = self.propensity
        return (g * tau_c + (1 - g) * tau_t).reshape(-1, 1)


class RLearnerWrapper:
    """R-Learner (Robinson decomposition).

    Trains on pseudo-outcome = Y_residual / W_residual with
    sample_weight = W_residual^2 to estimate CATE directly.
    """

    def __init__(self, cate_model, feature_cols: list[str]):
        self.cate_model = cate_model
        self.feature_cols = feature_cols

    def predict(self, X) -> np.ndarray:
        X_arr = X if isinstance(X, np.ndarray) else X[self.feature_cols].values
        return self.cate_model.predict(X_arr).reshape(-1, 1)


class UpliftModelWrapper:
    """Adapts causalml tree / forest models to the uniform CATE interface.

    causalml output conventions
    ---------------------------
    UpliftTreeClassifier         shape (N, 2): [P(Y|ctrl), P(Y|trt)]  →  CATE = col1 - col0
    UpliftRandomForestClassifier shape (N, 1): CATE directly           →  CATE = col0
    """

    def __init__(self, model, feature_cols: list[str]):
        self.model = model
        self.feature_cols = feature_cols

    def predict(self, X) -> np.ndarray:
        X_arr = X if isinstance(X, np.ndarray) else X[self.feature_cols].values
        preds = self.model.predict(X_arr)
        if preds.ndim == 1:
            return preds.reshape(-1, 1)
        if preds.shape[1] == 2:
            return (preds[:, 1] - preds[:, 0]).reshape(-1, 1)
        return preds[:, :1]
