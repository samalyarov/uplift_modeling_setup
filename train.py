"""
CLI script for training and registering the uplift model.

Replicates the full notebook pipeline end-to-end:
  data loading → feature extraction → preprocessing → Optuna tuning →
  final model training → artifact saving → MLflow registration.

Usage examples
--------------
# Train all 11 models with default settings:
    python train.py

# Train a lightweight subset (fast iteration / CI):
    python train.py --models slearner-lgb xlearner-lgb rlearner-lgb

# Override trial budget and output dir:
    python train.py --n-trials-fast 30 --n-trials-medium 20 --n-trials-slow 15 \\
                    --artifacts-dir artifacts

# Run on CPU (no GPU required):
    python train.py --device cpu

# Custom data / MLflow:
    python train.py --system-config configs/system.json \\
                    --mlflow-uri http://my-mlflow:5000 \\
                    --experiment smart-reach-training
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import time

import mlflow
import numpy as np
import optuna
import sklearn.pipeline as skpipe
from sklearn.model_selection import cross_val_predict, train_test_split

import catboost as cb
import lightgbm as lgb
import xgboost as xgb
from causalml.inference.tree import UpliftRandomForestClassifier, UpliftTreeClassifier

from src.datalib import build_engine
from src.datalib.features import extract_features
from src.datalib.transforms import FillNaTransformer, LocationEncoder
from src.model_utils import ModelKeeper
from src.training.learners import (
    RLearnerWrapper,
    SLearnerWrapper,
    UpliftModelWrapper,
    XLearnerWrapper,
)
from src.training.metrics import qini_coefficient, uplift_at_k

# ---------------------------------------------------------------------------
# Constants (must match campaign_flow.py and the notebook)
# ---------------------------------------------------------------------------

PRICE_PER_GRAM = 80
COST_PER_GRAM = 52
MARGIN_PER_GRAM = PRICE_PER_GRAM - COST_PER_GRAM   # 28 dollars/gram
CONTACT_COST = 1                                    # dollars per targeted customer
OFFER_DAYS = 7                                      # post-campaign window length

ALL_MODEL_IDS = [
    "slearner-lgb",
    "slearner-xgb",
    "slearner-cb",
    "uplift-tree",
    "uplift-rf",
    "xlearner-lgb",
    "xlearner-xgb",
    "xlearner-cb",
    "rlearner-lgb",
    "rlearner-xgb",
    "rlearner-cb",
]

SERVING_EXTRACT_CONFIG = [
    {"name": "receipts_agg",    "args": {"delta": 7}},
    {"name": "receipts_agg",    "args": {"delta": 15}},
    {"name": "receipts_agg",    "args": {"delta": 30}},
    {"name": "receipts_agg",    "args": {"delta": 60}},
    {"name": "receipts_agg",    "args": {"delta": 90}},
    {"name": "receipts_agg",    "args": {"delta": 180}},
    {"name": "receipts_agg",    "args": {"delta": 365}},
    {"name": "recency_global",  "args": {}},
    {"name": "purchase_trend",  "args": {"delta_short": 7,  "delta_long": 30}},
    {"name": "purchase_trend",  "args": {"delta_short": 15, "delta_long": 60}},
    {"name": "purchase_trend",  "args": {"delta_short": 30, "delta_long": 120}},
    {"name": "purchase_trend",  "args": {"delta_short": 30, "delta_long": 365}},
    {"name": "purchase_trend",  "args": {"delta_short": 90, "delta_long": 365}},
    {"name": "demographics",    "args": {}},
    {"name": "campaign_history","args": {}},
    {"name": "day_of_week",     "args": {}},
    {"name": "avg_city_cheque", "args": {}},
    {"name": "loyalty",         "args": {}},
]

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: device strings per backend
# ---------------------------------------------------------------------------

def _lgb_device(device: str) -> str:
    return "gpu" if device == "gpu" else "cpu"


def _xgb_device(device: str) -> str:
    return "cuda" if device == "gpu" else "cpu"


def _cb_task_type(device: str) -> str:
    return "GPU" if device == "gpu" else "CPU"


# ---------------------------------------------------------------------------
# Helper: resolve max_features for Uplift RF
# ---------------------------------------------------------------------------

def _resolve_max_features(choice: str, n_features: int) -> int | float | str:
    if choice == "sqrt":
        return int(np.sqrt(n_features))
    if choice == "log2":
        return int(np.log2(n_features))
    return n_features   # "all"


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

def _load_and_prepare(data_root: str, random_state: int):
    """Load raw tables, build target variable, extract + transform features.

    Returns
    -------
    data : pd.DataFrame   transformed feature matrix with target/treatment cols
    transform_pipeline    fitted sklearn Pipeline (for serving)
    cols_features         list of feature column names
    col_treatment         name of the treatment column
    col_target            name of the target column
    engine                the Engine (needed for full-dataset scoring later)
    extract_config        feature extraction config used (with dates injected)
    """
    import pandas as pd

    engine = build_engine(data_root)
    customers = engine.get_table("customers")
    receipts  = engine.get_table("receipts")
    campaigns = engine.get_table("campaigns")

    # ---- Target variable ----
    campaign_date = int(campaigns["date"].min())
    window_end    = campaign_date + OFFER_DAYS

    post_receipts = receipts[
        (receipts["date"] >= campaign_date) & (receipts["date"] < window_end)
    ]
    client_revenue = (
        post_receipts
        .groupby("customer_id")
        .agg(
            total_purchased_grams=("purchase_amt",  "sum"),
            total_revenue=        ("purchase_sum",  "sum"),
            total_discount=       ("discount",      "sum"),
        )
        .reset_index()
    )

    campaign_ids = set(campaigns["customer_id"].unique())
    train_data = customers[["customer_id"]].copy()
    train_data["target_group_flag"] = (
        train_data["customer_id"].isin(campaign_ids).astype(int)
    )
    train_data = train_data.merge(client_revenue, on="customer_id", how="left").fillna(0)
    train_data["target_profit"] = (
        train_data["total_purchased_grams"] * MARGIN_PER_GRAM
        - train_data["total_discount"]
        - train_data["target_group_flag"] * CONTACT_COST
    )

    # ---- Feature extraction (no leakage: date < campaign) ----
    feature_date = campaign_date - 1
    extract_config = [
        {"name": "receipts_agg",     "args": {"delta": d, "date_to": feature_date}}
        for d in [7, 15, 30, 60, 90, 180, 365]
    ] + [
        {"name": "recency_global",   "args": {"date_to": feature_date}},
        {"name": "purchase_trend",   "args": {"delta_short": 7,  "delta_long": 30,  "date_to": feature_date}},
        {"name": "purchase_trend",   "args": {"delta_short": 15, "delta_long": 60,  "date_to": feature_date}},
        {"name": "purchase_trend",   "args": {"delta_short": 30, "delta_long": 120, "date_to": feature_date}},
        {"name": "purchase_trend",   "args": {"delta_short": 30, "delta_long": 365, "date_to": feature_date}},
        {"name": "purchase_trend",   "args": {"delta_short": 90, "delta_long": 365, "date_to": feature_date}},
        {"name": "demographics",     "args": {}},
        {"name": "campaign_history", "args": {"date_to": feature_date}},
        {"name": "day_of_week",      "args": {"date_to": feature_date}},
        {"name": "avg_city_cheque",  "args": {"date_to": feature_date}},
        {"name": "loyalty",          "args": {"date_to": feature_date}},
    ]

    raw_features = extract_features(engine, extract_config)
    data = train_data.merge(raw_features, on="customer_id", how="inner")

    transform_pipeline = skpipe.Pipeline([
        ("fill_na",        FillNaTransformer(fill_value=0.0)),
        ("encode_location", LocationEncoder(prefix="loc")),
    ])
    data = transform_pipeline.fit_transform(data)

    non_feature_cols = [
        "customer_id", "target_group_flag",
        "total_purchased_grams", "total_revenue", "total_discount",
        "target_profit",
    ]
    cols_features = [c for c in data.columns if c not in non_feature_cols]

    log.info("Data shape: %s  |  features: %d", data.shape, len(cols_features))
    return data, transform_pipeline, cols_features, "target_group_flag", "target_profit", engine


# ---------------------------------------------------------------------------
# R-Learner residual pre-computation (shared by all three R-Learner variants)
# ---------------------------------------------------------------------------

def _compute_r_residuals(X_train, y_train, w_train, random_state, device):
    """Cross-validated nuisance models for the Robinson decomposition."""
    log.info("  Computing R-Learner residuals (5-fold CV) ...")
    t0 = time.time()

    lgb_device = _lgb_device(device)

    m_y = lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                             random_state=random_state, verbose=-1, device=lgb_device)
    y_hat = cross_val_predict(m_y, X_train, y_train, cv=5)
    y_residual = y_train - y_hat

    m_w = lgb.LGBMRegressor(n_estimators=200, max_depth=4, learning_rate=0.1,
                             random_state=random_state, verbose=-1, device=lgb_device)
    w_hat = cross_val_predict(m_w, X_train, w_train, cv=5)
    w_hat = np.clip(w_hat, 0.01, 0.99)
    w_residual = w_train - w_hat

    pseudo_outcome = y_residual / w_residual
    sample_weight  = w_residual ** 2

    log.info("  Residuals computed in %.0fs", time.time() - t0)
    return pseudo_outcome, sample_weight


# ---------------------------------------------------------------------------
# Optuna study runner
# ---------------------------------------------------------------------------

def _run_study(objective, n_trials: int, study_name: str) -> optuna.Study:
    study = optuna.create_study(direction="maximize", study_name=study_name)
    study.optimize(objective, n_trials=n_trials)
    log.info("  %s  best Qini = %.6f  (%d trials)", study_name, study.best_value, n_trials)
    return study


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    data_root: str,
    artifacts_dir: str,
    mlflow_uri: str,
    experiment_name: str,
    models_to_train: list[str],
    n_trials_fast: int,
    n_trials_medium: int,
    n_trials_slow: int,
    random_state: int,
    device: str,
) -> None:
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ---- Load & prepare data ----
    log.info("Loading and preparing data from %s ...", data_root)
    (data, transform_pipeline, cols_features,
     col_treatment, col_target, engine) = _load_and_prepare(data_root, random_state)

    fit_idx, val_idx = train_test_split(data.index, test_size=0.3,
                                        random_state=random_state)

    X_train_full = data.loc[fit_idx, cols_features].values.astype(np.float32)
    y_train_full = data.loc[fit_idx, col_target].values.astype(np.float32)
    w_train_full = data.loc[fit_idx, col_treatment].values.astype(np.float32)

    X_val = data.loc[val_idx, cols_features].values.astype(np.float32)
    y_val = data.loc[val_idx, col_target].values.astype(np.float32)
    w_val = data.loc[val_idx, col_treatment].values.astype(np.float32)

    X_train_st = np.column_stack([X_train_full, w_train_full])
    X_val_1    = np.column_stack([X_val, np.ones(len(X_val),  dtype=np.float32)])
    X_val_0    = np.column_stack([X_val, np.zeros(len(X_val), dtype=np.float32)])

    propensity_global = float(w_train_full.mean())
    treat_mask = w_train_full == 1
    ctrl_mask  = w_train_full == 0
    X_treat, y_treat = X_train_full[treat_mask], y_train_full[treat_mask]
    X_ctrl,  y_ctrl  = X_train_full[ctrl_mask],  y_train_full[ctrl_mask]

    w_str_full = np.where(w_train_full == 1, "treatment", "control")

    log.info(
        "Split: train=%d  val=%d  treatment_rate=%.2f%%",
        len(fit_idx), len(val_idx), propensity_global * 100,
    )

    # Device strings
    lgb_dev   = _lgb_device(device)
    xgb_dev   = _xgb_device(device)
    cb_ttype  = _cb_task_type(device)

    # ---- MLflow ----
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    # ============================================================
    # Optuna phase
    # ============================================================
    studies: dict[str, optuna.Study] = {}

    # --- S-Learner LightGBM ---
    if "slearner-lgb" in models_to_train:
        log.info("Tuning S-Learner LightGBM ...")
        def _obj_slearner_lgb(trial):
            p = dict(
                n_estimators=trial.suggest_int("n_estimators", 100, 700),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                num_leaves=trial.suggest_int("num_leaves", 15, 127),
                min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            )
            mdl = lgb.LGBMRegressor(**p, random_state=random_state, verbose=-1, device=lgb_dev)
            mdl.fit(X_train_st, y_train_full)
            cate = mdl.predict(X_val_1) - mdl.predict(X_val_0)
            return qini_coefficient(y_val, w_val, cate)
        studies["slearner-lgb"] = _run_study(_obj_slearner_lgb, n_trials_fast, "slearner-lgb")

    # --- S-Learner XGBoost ---
    if "slearner-xgb" in models_to_train:
        log.info("Tuning S-Learner XGBoost ...")
        def _obj_slearner_xgb(trial):
            p = dict(
                n_estimators=trial.suggest_int("n_estimators", 100, 700),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                gamma=trial.suggest_float("gamma", 0, 5.0),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            )
            mdl = xgb.XGBRegressor(**p, random_state=random_state, verbosity=0, device=xgb_dev)
            mdl.fit(X_train_st, y_train_full)
            cate = mdl.predict(X_val_1) - mdl.predict(X_val_0)
            return qini_coefficient(y_val, w_val, cate)
        studies["slearner-xgb"] = _run_study(_obj_slearner_xgb, n_trials_fast, "slearner-xgb")

    # --- S-Learner CatBoost ---
    if "slearner-cb" in models_to_train:
        log.info("Tuning S-Learner CatBoost ...")
        def _obj_slearner_cb(trial):
            p = dict(
                iterations=trial.suggest_int("iterations", 100, 700),
                depth=trial.suggest_int("depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
                random_strength=trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
                border_count=trial.suggest_int("border_count", 32, 255),
            )
            mdl = cb.CatBoostRegressor(**p, random_seed=random_state, verbose=0, task_type=cb_ttype)
            mdl.fit(X_train_st, y_train_full)
            cate = mdl.predict(X_val_1) - mdl.predict(X_val_0)
            return qini_coefficient(y_val, w_val, cate)
        studies["slearner-cb"] = _run_study(_obj_slearner_cb, n_trials_medium, "slearner-cb")

    # --- Uplift Decision Tree ---
    if "uplift-tree" in models_to_train:
        log.info("Tuning Uplift Decision Tree ...")
        def _obj_utree(trial):
            p = dict(
                max_depth=trial.suggest_int("max_depth", 3, 15),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 50, 500),
                min_samples_treatment=trial.suggest_int("min_samples_treatment", 10, 200),
                n_reg=trial.suggest_int("n_reg", 10, 100),
                evaluationFunction=trial.suggest_categorical("evaluationFunction", ["KL", "ED", "Chi"]),
            )
            mdl = UpliftTreeClassifier(**p, control_name="control")
            w_str = np.where(w_train_full == 1, "treatment", "control")
            mdl.fit(X_train_full, w_str, y_train_full)
            preds = mdl.predict(X_val)
            cate = (preds[:, 1] - preds[:, 0]) if preds.ndim == 2 else preds
            return qini_coefficient(y_val, w_val, cate)
        studies["uplift-tree"] = _run_study(_obj_utree, n_trials_medium, "uplift-tree")

    # --- Uplift Random Forest ---
    if "uplift-rf" in models_to_train:
        log.info("Tuning Uplift Random Forest ...")
        def _obj_urf(trial):
            p = dict(
                n_estimators=trial.suggest_int("n_estimators", 30, 100),
                max_depth=trial.suggest_int("max_depth", 3, 8),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 200, 500),
                min_samples_treatment=trial.suggest_int("min_samples_treatment", 50, 200),
                n_reg=trial.suggest_int("n_reg", 10, 100),
                evaluationFunction=trial.suggest_categorical("evaluationFunction", ["KL", "ED", "Chi"]),
            )
            mf_choice = trial.suggest_categorical("max_features_str", ["sqrt", "log2", "all"])
            n_feat = X_train_full.shape[1]
            w_str = np.where(w_train_full == 1, "treatment", "control")
            mdl = UpliftRandomForestClassifier(
                **p,
                max_features=_resolve_max_features(mf_choice, n_feat),
                control_name="control", random_state=random_state, n_jobs=-1,
            )
            mdl.fit(X_train_full, w_str, y_train_full)
            preds = mdl.predict(X_val)
            cate = preds[:, 0] if preds.ndim == 2 else preds
            return qini_coefficient(y_val, w_val, cate)
        studies["uplift-rf"] = _run_study(_obj_urf, n_trials_slow, "uplift-rf")

    # --- X-Learner LightGBM ---
    if "xlearner-lgb" in models_to_train:
        log.info("Tuning X-Learner LightGBM ...")
        def _obj_xl_lgb(trial):
            p = dict(
                n_estimators=trial.suggest_int("n_estimators", 100, 700),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                num_leaves=trial.suggest_int("num_leaves", 15, 127),
                min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            )
            ex = dict(random_state=random_state, verbose=-1, device=lgb_dev)
            m_t = lgb.LGBMRegressor(**p, **ex); m_t.fit(X_treat, y_treat)
            m_c = lgb.LGBMRegressor(**p, **ex); m_c.fit(X_ctrl, y_ctrl)
            d_t = y_treat - m_c.predict(X_treat)
            d_c = m_t.predict(X_ctrl) - y_ctrl
            ct = lgb.LGBMRegressor(**p, **ex); ct.fit(X_treat, d_t)
            cc = lgb.LGBMRegressor(**p, **ex); cc.fit(X_ctrl, d_c)
            cate = propensity_global * cc.predict(X_val) + (1 - propensity_global) * ct.predict(X_val)
            return qini_coefficient(y_val, w_val, cate)
        studies["xlearner-lgb"] = _run_study(_obj_xl_lgb, n_trials_medium, "xlearner-lgb")

    # --- X-Learner XGBoost ---
    if "xlearner-xgb" in models_to_train:
        log.info("Tuning X-Learner XGBoost ...")
        def _obj_xl_xgb(trial):
            p = dict(
                n_estimators=trial.suggest_int("n_estimators", 100, 700),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                gamma=trial.suggest_float("gamma", 0, 5.0),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            )
            ex = dict(random_state=random_state, verbosity=0, device=xgb_dev)
            m_t = xgb.XGBRegressor(**p, **ex); m_t.fit(X_treat, y_treat)
            m_c = xgb.XGBRegressor(**p, **ex); m_c.fit(X_ctrl, y_ctrl)
            d_t = y_treat - m_c.predict(X_treat)
            d_c = m_t.predict(X_ctrl) - y_ctrl
            ct = xgb.XGBRegressor(**p, **ex); ct.fit(X_treat, d_t)
            cc = xgb.XGBRegressor(**p, **ex); cc.fit(X_ctrl, d_c)
            cate = propensity_global * cc.predict(X_val) + (1 - propensity_global) * ct.predict(X_val)
            return qini_coefficient(y_val, w_val, cate)
        studies["xlearner-xgb"] = _run_study(_obj_xl_xgb, n_trials_medium, "xlearner-xgb")

    # --- X-Learner CatBoost ---
    if "xlearner-cb" in models_to_train:
        log.info("Tuning X-Learner CatBoost ...")
        def _obj_xl_cb(trial):
            p = dict(
                iterations=trial.suggest_int("iterations", 100, 700),
                depth=trial.suggest_int("depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
                random_strength=trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
                border_count=trial.suggest_int("border_count", 32, 255),
            )
            ex = dict(random_seed=random_state, verbose=0, task_type=cb_ttype)
            m_t = cb.CatBoostRegressor(**p, **ex); m_t.fit(X_treat, y_treat)
            m_c = cb.CatBoostRegressor(**p, **ex); m_c.fit(X_ctrl, y_ctrl)
            d_t = y_treat - m_c.predict(X_treat)
            d_c = m_t.predict(X_ctrl) - y_ctrl
            ct = cb.CatBoostRegressor(**p, **ex); ct.fit(X_treat, d_t)
            cc = cb.CatBoostRegressor(**p, **ex); cc.fit(X_ctrl, d_c)
            cate = propensity_global * cc.predict(X_val) + (1 - propensity_global) * ct.predict(X_val)
            return qini_coefficient(y_val, w_val, cate)
        studies["xlearner-cb"] = _run_study(_obj_xl_cb, n_trials_medium, "xlearner-cb")

    # --- R-Learner: pre-compute shared residuals once ---
    r_variants = {"rlearner-lgb", "rlearner-xgb", "rlearner-cb"}
    if r_variants & set(models_to_train):
        r_pseudo, r_weight = _compute_r_residuals(
            X_train_full, y_train_full, w_train_full, random_state, device
        )

    # --- R-Learner LightGBM ---
    if "rlearner-lgb" in models_to_train:
        log.info("Tuning R-Learner LightGBM ...")
        def _obj_rl_lgb(trial):
            p = dict(
                n_estimators=trial.suggest_int("n_estimators", 100, 700),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                num_leaves=trial.suggest_int("num_leaves", 15, 127),
                min_child_samples=trial.suggest_int("min_child_samples", 5, 100),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            )
            mdl = lgb.LGBMRegressor(**p, random_state=random_state, verbose=-1, device=lgb_dev)
            mdl.fit(X_train_full, r_pseudo, sample_weight=r_weight)
            return qini_coefficient(y_val, w_val, mdl.predict(X_val))
        studies["rlearner-lgb"] = _run_study(_obj_rl_lgb, n_trials_medium, "rlearner-lgb")

    # --- R-Learner XGBoost ---
    if "rlearner-xgb" in models_to_train:
        log.info("Tuning R-Learner XGBoost ...")
        def _obj_rl_xgb(trial):
            p = dict(
                n_estimators=trial.suggest_int("n_estimators", 100, 700),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
                subsample=trial.suggest_float("subsample", 0.5, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
                gamma=trial.suggest_float("gamma", 0, 5.0),
                reg_alpha=trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            )
            mdl = xgb.XGBRegressor(**p, random_state=random_state, verbosity=0, device=xgb_dev)
            mdl.fit(X_train_full, r_pseudo, sample_weight=r_weight)
            return qini_coefficient(y_val, w_val, mdl.predict(X_val))
        studies["rlearner-xgb"] = _run_study(_obj_rl_xgb, n_trials_medium, "rlearner-xgb")

    # --- R-Learner CatBoost ---
    if "rlearner-cb" in models_to_train:
        log.info("Tuning R-Learner CatBoost ...")
        def _obj_rl_cb(trial):
            p = dict(
                iterations=trial.suggest_int("iterations", 100, 700),
                depth=trial.suggest_int("depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
                bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 1.0),
                random_strength=trial.suggest_float("random_strength", 1e-8, 10.0, log=True),
                border_count=trial.suggest_int("border_count", 32, 255),
            )
            mdl = cb.CatBoostRegressor(**p, random_seed=random_state, verbose=0, task_type=cb_ttype)
            mdl.fit(X_train_full, r_pseudo, sample_weight=r_weight)
            return qini_coefficient(y_val, w_val, mdl.predict(X_val))
        studies["rlearner-cb"] = _run_study(_obj_rl_cb, n_trials_medium, "rlearner-cb")

    # ============================================================
    # Final model training with best hyperparameters
    # ============================================================
    log.info("Training final models on full training set ...")
    trained_models: dict[str, object] = {}

    if "slearner-lgb" in studies:
        t0 = time.time()
        bp = studies["slearner-lgb"].best_params
        m = lgb.LGBMRegressor(**bp, random_state=random_state, verbose=-1, device=lgb_dev)
        m.fit(X_train_st, y_train_full)
        trained_models["S-Learner LightGBM"] = SLearnerWrapper(m, cols_features)
        log.info("  S-Learner LightGBM  %.1fs", time.time() - t0)

    if "slearner-xgb" in studies:
        t0 = time.time()
        bp = studies["slearner-xgb"].best_params
        m = xgb.XGBRegressor(**bp, random_state=random_state, verbosity=0, device=xgb_dev)
        m.fit(X_train_st, y_train_full)
        trained_models["S-Learner XGBoost"] = SLearnerWrapper(m, cols_features)
        log.info("  S-Learner XGBoost   %.1fs", time.time() - t0)

    if "slearner-cb" in studies:
        t0 = time.time()
        bp = studies["slearner-cb"].best_params
        m = cb.CatBoostRegressor(**bp, random_seed=random_state, verbose=0, task_type=cb_ttype)
        m.fit(X_train_st, y_train_full)
        trained_models["S-Learner CatBoost"] = SLearnerWrapper(m, cols_features)
        log.info("  S-Learner CatBoost  %.1fs", time.time() - t0)

    if "uplift-tree" in studies:
        t0 = time.time()
        bp = {k: v for k, v in studies["uplift-tree"].best_params.items()}
        m = UpliftTreeClassifier(**bp, control_name="control")
        m.fit(X_train_full, w_str_full, y_train_full)
        trained_models["Uplift Decision Tree"] = UpliftModelWrapper(m, cols_features)
        log.info("  Uplift Decision Tree %.1fs", time.time() - t0)

    if "uplift-rf" in studies:
        t0 = time.time()
        bp = dict(studies["uplift-rf"].best_params)
        mf_choice = bp.pop("max_features_str", "all")
        m = UpliftRandomForestClassifier(
            **bp,
            max_features=_resolve_max_features(mf_choice, X_train_full.shape[1]),
            control_name="control", random_state=random_state, n_jobs=-1,
        )
        m.fit(X_train_full, w_str_full, y_train_full)
        trained_models["Uplift Random Forest"] = UpliftModelWrapper(m, cols_features)
        log.info("  Uplift Random Forest %.1fs", time.time() - t0)

    def _build_xlearner(bp, cls, extra):
        m_t = cls(**bp, **extra); m_t.fit(X_treat, y_treat)
        m_c = cls(**bp, **extra); m_c.fit(X_ctrl,  y_ctrl)
        d_t = y_treat - m_c.predict(X_treat)
        d_c = m_t.predict(X_ctrl) - y_ctrl
        ct = cls(**bp, **extra); ct.fit(X_treat, d_t)
        cc = cls(**bp, **extra); cc.fit(X_ctrl,  d_c)
        return XLearnerWrapper(m_t, m_c, ct, cc, cols_features, propensity_global)

    if "xlearner-lgb" in studies:
        t0 = time.time()
        ex = dict(random_state=random_state, verbose=-1, device=lgb_dev)
        trained_models["X-Learner LightGBM"] = _build_xlearner(
            studies["xlearner-lgb"].best_params, lgb.LGBMRegressor, ex)
        log.info("  X-Learner LightGBM  %.1fs", time.time() - t0)

    if "xlearner-xgb" in studies:
        t0 = time.time()
        ex = dict(random_state=random_state, verbosity=0, device=xgb_dev)
        trained_models["X-Learner XGBoost"] = _build_xlearner(
            studies["xlearner-xgb"].best_params, xgb.XGBRegressor, ex)
        log.info("  X-Learner XGBoost   %.1fs", time.time() - t0)

    if "xlearner-cb" in studies:
        t0 = time.time()
        ex = dict(random_seed=random_state, verbose=0, task_type=cb_ttype)
        trained_models["X-Learner CatBoost"] = _build_xlearner(
            studies["xlearner-cb"].best_params, cb.CatBoostRegressor, ex)
        log.info("  X-Learner CatBoost  %.1fs", time.time() - t0)

    # R-Learner final: re-compute residuals on full data for clean final model
    if r_variants & set(studies.keys()):
        log.info("  Recomputing R-Learner residuals on full training set ...")
        r_pseudo_full, r_weight_full = _compute_r_residuals(
            X_train_full, y_train_full, w_train_full, random_state, device
        )

    if "rlearner-lgb" in studies:
        t0 = time.time()
        bp = studies["rlearner-lgb"].best_params
        m = lgb.LGBMRegressor(**bp, random_state=random_state, verbose=-1, device=lgb_dev)
        m.fit(X_train_full, r_pseudo_full, sample_weight=r_weight_full)
        trained_models["R-Learner LightGBM"] = RLearnerWrapper(m, cols_features)
        log.info("  R-Learner LightGBM  %.1fs", time.time() - t0)

    if "rlearner-xgb" in studies:
        t0 = time.time()
        bp = studies["rlearner-xgb"].best_params
        m = xgb.XGBRegressor(**bp, random_state=random_state, verbosity=0, device=xgb_dev)
        m.fit(X_train_full, r_pseudo_full, sample_weight=r_weight_full)
        trained_models["R-Learner XGBoost"] = RLearnerWrapper(m, cols_features)
        log.info("  R-Learner XGBoost   %.1fs", time.time() - t0)

    if "rlearner-cb" in studies:
        t0 = time.time()
        bp = studies["rlearner-cb"].best_params
        m = cb.CatBoostRegressor(**bp, random_seed=random_state, verbose=0, task_type=cb_ttype)
        m.fit(X_train_full, r_pseudo_full, sample_weight=r_weight_full)
        trained_models["R-Learner CatBoost"] = RLearnerWrapper(m, cols_features)
        log.info("  R-Learner CatBoost  %.1fs", time.time() - t0)

    # ============================================================
    # Evaluate on validation set + log to MLflow
    # ============================================================
    log.info("Evaluating models and logging to MLflow ...")
    results: dict[str, dict] = {}

    for name, model in trained_models.items():
        model_id = name.lower().replace(" ", "-")
        study = studies.get(model_id.replace("-", "").replace("learner", "-learner")
                              .replace("slearner", "slearner")
                              .replace("uplift", "uplift"))
        # Use the model_id mapping to look up study
        id_map = {
            "s-learner-lightgbm": "slearner-lgb",
            "s-learner-xgboost":  "slearner-xgb",
            "s-learner-catboost": "slearner-cb",
            "uplift-decision-tree": "uplift-tree",
            "uplift-random-forest": "uplift-rf",
            "x-learner-lightgbm": "xlearner-lgb",
            "x-learner-xgboost":  "xlearner-xgb",
            "x-learner-catboost": "xlearner-cb",
            "r-learner-lightgbm": "rlearner-lgb",
            "r-learner-xgboost":  "rlearner-xgb",
            "r-learner-catboost": "rlearner-cb",
        }
        study_key = id_map.get(model_id)
        study = studies.get(study_key) if study_key else None

        with mlflow.start_run(run_name=model_id) as run:
            if study:
                mlflow.log_params(study.best_params)
                mlflow.log_param("optuna_trials", len(study.trials))
                mlflow.log_metric("optuna_best_qini", study.best_value)
            mlflow.log_param("model_type", name)
            mlflow.log_param("random_state", random_state)
            mlflow.log_param("device", device)

            preds = model.predict(X_val)
            if preds.ndim == 2:
                preds = preds[:, 0]

            qini  = qini_coefficient(y_val, w_val, preds)
            u30   = uplift_at_k(y_val, w_val, preds, k=0.3)
            mlflow.log_metrics({
                "qini_coefficient": qini,
                "uplift_at_30pct":  u30,
                "mean_cate":        float(np.mean(preds)),
                "pct_positive_cate": float((preds > 0).mean() * 100),
            })

            mk = ModelKeeper(model=model, column_set=cols_features)
            mk.log_to_mlflow()

            results[name] = {"model": model, "study": study, "qini": qini, "uplift_30": u30,
                             "run_id": run.info.run_id}

        log.info("  %-28s  Qini=%+.6f  U@30%%=%+.2f", name, qini, u30)

    # ============================================================
    # Select best model + save artifacts
    # ============================================================
    best_name  = max(results, key=lambda k: results[k]["qini"])
    best_model = results[best_name]["model"]
    log.info("Best model: %s  (Qini = %+.6f)", best_name, results[best_name]["qini"])

    os.makedirs(artifacts_dir, exist_ok=True)

    pipeline_path = os.path.join(artifacts_dir, "serving_transform_pipeline.pickle")
    with open(pipeline_path, "wb") as f:
        pickle.dump(transform_pipeline, f)

    model_path = os.path.join(artifacts_dir, "uplift_model.pickle")
    mk = ModelKeeper(model=best_model, column_set=cols_features)
    mk.dump(model_path)

    extract_cfg_path = os.path.join(artifacts_dir, "serving_extract_config.json")
    with open(extract_cfg_path, "w") as f:
        json.dump(SERVING_EXTRACT_CONFIG, f, indent=2)

    log.info("Artifacts saved to %s:", artifacts_dir)
    for fn in sorted(os.listdir(artifacts_dir)):
        log.info("  %s", fn)

    # Register best model in MLflow Model Registry
    best_run_id = results[best_name]["run_id"]
    model_uri   = f"runs:/{best_run_id}/model"
    try:
        registered = mlflow.register_model(model_uri, "smart-reach-uplift-model")
        log.info("Registered model version: %s", registered.version)
    except Exception as exc:
        log.warning("Model Registry unavailable (%s) — skipping registration.", exc)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="train",
        description="Train and register the Smart Reach uplift model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--system-config", default="configs/system.json",
        help="Path to system config JSON (data root, MLflow URI, artifacts dir).",
    )
    parser.add_argument(
        "--data-root", default=None,
        help="Override database.root_path from system config.",
    )
    parser.add_argument(
        "--artifacts-dir", default=None,
        help="Override artifacts_root_path from system config.",
    )
    parser.add_argument(
        "--mlflow-uri", default=None,
        help="Override mlflow.tracking_uri from system config.",
    )
    parser.add_argument(
        "--experiment", default=None,
        help="MLflow experiment name (default: smart-reach-training).",
    )
    parser.add_argument(
        "--models", nargs="+", choices=ALL_MODEL_IDS, default=ALL_MODEL_IDS,
        metavar="MODEL",
        help=f"Which models to train. Choices: {', '.join(ALL_MODEL_IDS)}",
    )
    parser.add_argument(
        "--n-trials-fast",   type=int, default=80,
        help="Optuna trials for fast learners (LightGBM / XGBoost S-Learner).",
    )
    parser.add_argument(
        "--n-trials-medium", type=int, default=60,
        help="Optuna trials for medium learners (CatBoost, X-Learner, R-Learner).",
    )
    parser.add_argument(
        "--n-trials-slow",   type=int, default=40,
        help="Optuna trials for slow learners (Uplift Random Forest).",
    )
    parser.add_argument(
        "--random-state", type=int, default=69,
        help="Global random seed.",
    )
    parser.add_argument(
        "--device", choices=["gpu", "cpu"], default="gpu",
        help="Compute device for gradient boosting models.",
    )

    args = parser.parse_args()

    # Load system config
    with open(args.system_config) as f:
        sys_cfg = json.load(f)

    data_root     = args.data_root     or sys_cfg["database"]["root_path"]
    artifacts_dir = args.artifacts_dir or sys_cfg["artifacts_root_path"]
    mlflow_uri    = args.mlflow_uri    or sys_cfg["mlflow"]["tracking_uri"]
    experiment    = args.experiment    or "smart-reach-training"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("Models to train: %s", args.models)
    log.info("Device: %s", args.device)
    log.info("Trials — fast: %d  medium: %d  slow: %d",
             args.n_trials_fast, args.n_trials_medium, args.n_trials_slow)

    train(
        data_root=data_root,
        artifacts_dir=artifacts_dir,
        mlflow_uri=mlflow_uri,
        experiment_name=experiment,
        models_to_train=args.models,
        n_trials_fast=args.n_trials_fast,
        n_trials_medium=args.n_trials_medium,
        n_trials_slow=args.n_trials_slow,
        random_state=args.random_state,
        device=args.device,
    )


if __name__ == "__main__":
    main()
