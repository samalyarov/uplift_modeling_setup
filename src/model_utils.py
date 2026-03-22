"""
Model wrapper: stores a trained model together with its expected feature columns
An abstract wrapper around the models - allowing us to change them more easily
"""

from __future__ import annotations

import pickle
from typing import Any

import pandas as pd

class ModelKeeper:
    """Wraps a trained uplift model and its feature column list"""

    def __init__(self, model: Any, column_set: list[str]):
        self.model = model
        self.column_set = column_set

    def predict(self, data: pd.DataFrame) -> pd.Series:
        X = data[self.column_set].values
        preds = self.model.predict(X)
        # causalml predict() returnes ndarray of shape (n, n_treatments)
        # for binary treatment there is only one treatment column
        if preds.ndim == 2:
            preds = preds[:, 0]
        return pd.Series(preds, index=data.index)
    
    def dump(self, path: str) -> None:
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "column_set": self.column_set}, f)

    def log_to_mlflow(self, artifact_path: str = "uplift_model") -> None:
        """Log the model bundle to the active MLflow run"""
        import mlflow
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model_keeper.pickle")
            self.dump(path)
            mlflow.log_artifact(path, artifact_path=artifact_path)

    @classmethod
    def load(cls, path: str) -> "ModelKeeper":
        with open(path, "rb") as f:
            obj = pickle.load(f) # noqa: S301
        return cls(model=obj["model"], column_set=obj["column_set"])