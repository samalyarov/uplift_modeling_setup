"""
Feature extraction: compute features from raw tables

Each calcer is a class with a 'compute(engine) -> pd.DataFrame' method.
All calcers produce a DataFrame keyed by 'customer_id'
"""

from __future__ import annotations

import datetime
from typing import List

import numpy as np
import pandas as pd

from . import Engine

# ---------------------------------------------------------------------------
# Receipt-based features
# ---------------------------------------------------------------------------

class ReceiptsAggCalcer:
    """Aggregate purchase statistics over the last 'delta' days before 'date_to'"""

    name = "receipts_agg"

    def __init__(self, delta: int, date_to: datetime.date):
        self.delta = delta
        self.date_to = date_to

    def compute(self, engine: Engine) -> pd.DataFrame:
        receipts = engine.get_table("receipts").copy()
        receipts["date"] = pd.to_datetime(receipts["date"])

        dt_to = pd.Timestamp(self.date_to)
        dt_from = dt_to - pd.Timedelta(days=self.delta)
        mask = (receipts["date"] >= dt_from) & (receipts["date"] < dt_to)
        df = receipts.loc[mask]

        suffix = f"__{self.delta}d"

        agg = df.groupby("customer_id").agg(
            txn_count=("date", "count"),
            purchase_amt_sum=("purchase_amt", "sum"),
            purchase_amt_mean=("purchase_amt", "mean"),
            purchase_amt_max=("purchase_amt", "max"),
            purchase_amt_min=("purchase_amt", "min"),
            purchase_amt_std=("purchase_amt", "std"),
            purchase_sum_sum=("purchase_sum", "sum"),
            purchase_sum_mean=("purchase_sum", "mean"),
            purchase_sum_max=("purchase_sum", "max"),
            purchase_sum_min=("purchase_sum", "min"),
            purchase_sum_std=("purchase_sum", "std"),
            date_min=("date", "min"),
            date_max=("date", "max"),
        ).reset_index()

        # Derived time features
        agg["mean_time_interval" + suffix] = np.where(
            agg["txn_count"] > 1,
            (agg["date_max"] - agg["date_min"]).dt.total_seconds()
            / (24 * 3600)
            / (agg["txn_count"] - 1),
            np.nan
        )

        # Rename columns with suffix
        rename = {
            c: c + suffix
            for c in agg.columns
            if c not in ("customer_id", "date_min", "date_max")
               and not c.endswith(suffix)
        }
        agg = agg.rename(columns=rename)
        agg = agg.drop(columns=["date_min", "date_max"])
        return agg
    
class RecencyCalcer:
    """Time (in days) since the very last purchase before 'date_to'"""

    name = "recency_global"

    def __init__(self, date_to: datetime.date):
        self.date_to = date_to

    def compute(self, engine: Engine) -> pd.DataFrame:
        receipts = engine.get_table("receipts").copy()
        receipts["date"] = pd.to_datetime(receipts["date"])
        dt_to = pd.Timestamp(self.date_to)
        receipts = receipts.loc[receipts["date"] < dt_to]

        last_date = receipts.groupby("customer_id")["date"].max().reset_index()
        last_date["days_since_last_purchase"] = (
            (dt_to - last_date["date"]).dt.total_seconds() / (24 * 3600)
        )

        return last_date[["customer_id", "days_since_last_purchase"]]
    
class PurchaseTrendCalcer:
    """Ratio of recent spend to older spend - captures trajectory"""

    name = "purchase_trend"

    def __init__(self, delta_short: int, delta_long: int, date_to: datetime.date):
        self.delta_short = delta_short
        self.delta_long = delta_long
        self.date_to = date_to

    def compute(self, engine: Engine) -> pd.DataFrame:
        receipts = engine.get_table("receipts").copy()
        receipts["date"] = pd.to_datetime(receipts["date"])
        dt_to = pd.Timestamp(self.date_to)

        def _sum_in_window(delta):
            dt_from = dt_to - pd.Timedelta(days=delta)
            mask = (receipts["date"] >= dt_from) & (receipts["date"] < dt_to)
            return receipts.loc[mask].groupby("customer_id")["purchase_sum"].sum()
        
        short = _sum_in_window(self.delta_short).rename("spend_short")
        long = _sum_in_window(self.delta_long).rename("spend_long")
        merged = pd.merge(short, long, on="customer_id", how="outer").fillna(0)
        merged["spend_trend_ratio"] = np.where(
            merged["spend_long"] > 0,
            merged["spend_short"] / merged["spend_long"],
            0.0,
        )
        return merged[["spend_trend_ratio"]].reset_index()
    

# ---------------------------------------------------------------------------
# Demographics
# ---------------------------------------------------------------------------

class DemographicsCalcer:
    """Age + location from the customers table"""

    name = "demographics"

    def compute(self, engine: Engine) -> pd.DataFrame:
        return engine.get_table("customers")[["customer_id", "age", "location"]]
    
# ---------------------------------------------------------------------------
# Campaign history features
# ---------------------------------------------------------------------------

class CampaignHistoryCalcer:
    """
    Features derived from past campaign participation
    Binary flas: was the customer ever targeted before?
    """

    name = "campaign_history"

    def __init__(self, date_to: datetime.date):
        self.date_to = date_to

    def compute(self, engine: Engine) -> pd.DataFrame:
        campaigns = engine.get_table("campaigns").copy()
        campaigns["date"] = pd.to_datetime(campaigns["date"])
        dt_to = pd.Timestamp(self.date_to)
        past = campaigns.loc[campaigns["date"] < dt_to]

        agg = past.groupby("customer_id").agg(
            n_past_campaigns=("date", "count"),
            was_in_target=("target_group_flag", "max")
        ).reset_index()
        return agg
    
# ---------------------------------------------------------------------------
# Registry & pipeline
# ---------------------------------------------------------------------------

CALCER_REGISTRY: dict[str, type] = {
    "receipts_agg": ReceiptsAggCalcer,
    "recency_global": RecencyCalcer,
    "purchase_trend": PurchaseTrendCalcer,
    "demographics": DemographicsCalcer,
    "campaign_history": CampaignHistoryCalcer,
}

def extract_features(engine: Engine, config: list[dict]) -> pd.DataFrame:
    """
    Run all calcers from config and outer-join on customer_id to compile a final table

    Config format::

        [
            {"name": "receipts_agg", "args": {"delta": 60, "date_to": "2019-03-19"}},
            {"name": "demographics", "args": {}},
            ...
        ]
    """
    frames: list[pd.DataFrame] = []
    for calcer_cfg in config:
        cls = CALCER_REGISTRY[calcer_cfg["name"]]
        args = _parse_args(calcer_cfg.get("args", {}))
        calcer = cls(**args)
        frames.append(calcer.compute(engine))

    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on="customer_id", how="outer")
    return result

def _parse_args(args: dict) -> dict:
    """Auto-convert date strings to datetime.date objects"""
    out = {}
    for k, v in args.items():
        if isinstance(v, str) and k.startswith("date"):
            out[k] = datetime.datetime.strptime(v, "%Y-%m-%d").date()
        else:
            out[k] = v
    return out
