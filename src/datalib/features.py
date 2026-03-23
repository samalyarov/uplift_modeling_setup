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
        agg["recency" + suffix] = (
            (dt_to - agg["date_max"]).dt.total_seconds() / (24 * 3600)
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
    Features derived from past campaign participation.
    Binary flag: was the customer ever targeted before?
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
# Day-of-week features
# ---------------------------------------------------------------------------

class DayOfWeekCalcer:
    """Distribution of purchases across days of the week before 'date_to'."""

    name = "day_of_week"

    def __init__(self, date_to: datetime.date):
        self.date_to = date_to

    def compute(self, engine: Engine) -> pd.DataFrame:
        receipts = engine.get_table("receipts").copy()
        receipts["date"] = pd.to_datetime(receipts["date"])
        dt_to = pd.Timestamp(self.date_to)
        receipts = receipts.loc[receipts["date"] < dt_to]

        receipts["dow"] = receipts["date"].dt.dayofweek  # 0=Mon, 6=Sun

        # Share of purchases per day of week
        dow_counts = (
            receipts.groupby(["customer_id", "dow"])
            .size()
            .unstack(fill_value=0)
        )
        total = dow_counts.sum(axis=1)
        dow_share = dow_counts.div(total, axis=0)
        dow_share.columns = [f"dow_share_{int(c)}" for c in dow_share.columns]

        # Most frequent purchase day
        mode_dow = receipts.groupby("customer_id")["dow"].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else -1
        ).rename("mode_dow")

        # Weekend share (Sat=5, Sun=6)
        receipts["is_weekend"] = receipts["dow"].isin([5, 6]).astype(int)
        weekend_share = (
            receipts.groupby("customer_id")["is_weekend"]
            .mean()
            .rename("weekend_purchase_share")
        )

        result = dow_share.join(mode_dow).join(weekend_share).reset_index()
        return result


# ---------------------------------------------------------------------------
# Average cheque by city
# ---------------------------------------------------------------------------

class AvgCityChequeCalcer:
    """Average purchase sum (cheque) by city — relative to customer's own average."""

    name = "avg_city_cheque"

    def __init__(self, date_to: datetime.date):
        self.date_to = date_to

    def compute(self, engine: Engine) -> pd.DataFrame:
        receipts = engine.get_table("receipts").copy()
        customers = engine.get_table("customers")[["customer_id", "location"]].copy()
        receipts["date"] = pd.to_datetime(receipts["date"])
        dt_to = pd.Timestamp(self.date_to)
        receipts = receipts.loc[receipts["date"] < dt_to]

        # Merge receipts with customer location
        r = receipts.merge(customers, on="customer_id", how="left")

        # City-level average cheque
        city_avg = r.groupby("location")["purchase_sum"].mean().rename("city_avg_cheque")

        # Customer-level average cheque
        cust_avg = r.groupby("customer_id")["purchase_sum"].mean().rename("customer_avg_cheque")
        cust_loc = r.groupby("customer_id")["location"].first()

        result = pd.DataFrame({"customer_avg_cheque": cust_avg, "location": cust_loc})
        result = result.join(city_avg, on="location")

        # Relative cheque: customer vs city average
        result["cheque_vs_city"] = np.where(
            result["city_avg_cheque"] > 0,
            result["customer_avg_cheque"] / result["city_avg_cheque"],
            1.0,
        )
        result = result.drop(columns=["location"])
        return result.reset_index()


# ---------------------------------------------------------------------------
# Loyalty coefficient
# ---------------------------------------------------------------------------

class LoyaltyCalcer:
    """Loyalty-related features: purchase regularity and engagement depth."""

    name = "loyalty"

    def __init__(self, date_to: datetime.date):
        self.date_to = date_to

    def compute(self, engine: Engine) -> pd.DataFrame:
        receipts = engine.get_table("receipts").copy()
        receipts["date"] = pd.to_datetime(receipts["date"])
        dt_to = pd.Timestamp(self.date_to)
        receipts = receipts.loc[receipts["date"] < dt_to]

        cust = receipts.groupby("customer_id").agg(
            n_txn=("date", "count"),
            n_unique_days=("date", "nunique"),
            first_purchase=("date", "min"),
            last_purchase=("date", "max"),
            total_spend=("purchase_sum", "sum"),
        )

        # Lifespan in days
        cust["lifespan_days"] = (
            (cust["last_purchase"] - cust["first_purchase"]).dt.total_seconds() / 86400
        )

        # Purchase frequency: unique days / lifespan (0-1 scale)
        cust["purchase_frequency"] = np.where(
            cust["lifespan_days"] > 0,
            cust["n_unique_days"] / cust["lifespan_days"],
            0.0,
        )

        # Average spend per transaction day
        cust["spend_per_day"] = np.where(
            cust["n_unique_days"] > 0,
            cust["total_spend"] / cust["n_unique_days"],
            0.0,
        )

        # Loyalty score: composite of frequency and lifespan
        cust["loyalty_score"] = cust["purchase_frequency"] * np.log1p(cust["lifespan_days"])

        result = cust[
            ["n_unique_days", "lifespan_days", "purchase_frequency",
             "spend_per_day", "loyalty_score"]
        ].reset_index()
        return result


# ---------------------------------------------------------------------------
# Registry & pipeline
# ---------------------------------------------------------------------------

CALCER_REGISTRY: dict[str, type] = {
    "receipts_agg": ReceiptsAggCalcer,
    "recency_global": RecencyCalcer,
    "purchase_trend": PurchaseTrendCalcer,
    "demographics": DemographicsCalcer,
    "campaign_history": CampaignHistoryCalcer,
    "day_of_week": DayOfWeekCalcer,
    "avg_city_cheque": AvgCityChequeCalcer,
    "loyalty": LoyaltyCalcer,
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
