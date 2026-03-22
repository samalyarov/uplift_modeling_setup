"""
Transformers: sklearn-compatible transformers for the feature pipeline
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd
import sklearn.base as skbase
import sklearn.preprocessing as skpreprocessing

class LocationEncoder(skbase.BaseEstimator, skbase.TransformerMixin):
    """One-hot encode the 'location' column"""
    
    def __init__(self, prefix: str = "loc"):
        self.prefix = prefix
        self.encoder_: skpreprocessing.OneHotEncoder | None = None

    def fit(self, data: pd.DataFrame, y=None):
        self.encoder_ = skpreprocessing.OneHotEncoder(
            sparse_output=False, handle_unknown="ignore"
        )
        self.encoder_.fit(data["location"])
        return self
    
    def transform(self, data: pd.DataFrame, y=None) -> pd.DataFrame:
        encoded = self.encoder_.transform(data[["location"]])
        cols = [
            f"{self.prefix}__{v}" for v in self.encoder_.categories_[0]
        ]
        df_enc = pd.DataFrame(encoded, column=cols, index=data.index)
        result = pd.concat([data, df_enc], axis=1)
        result = result.drop(column=["location"], errors="ignore")
        return result
    
class FillNaTransformer(skbase.BaseEstimator, skbase.TransformerMixin):
    """Fill NaN values with a given constant (default 0)"""

    def __init__(self, fill_value: float = 0.0):
        self.fill_value = fill_value

    def fit(self, data: pd.DataFrame, y=None):
        return self
    
    def transform(self, data: pd.DataFrame, y=None) -> pd.DataFrame:
        return data.fillna(self.fill_value)