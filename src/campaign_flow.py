"""
The Marketing Campaign Flow - orchestrated by Prefect 3.

Pipeline stages (Prefect tasks)
──────────────────────────────────────────────────────────────────────
1. load_data            — read raw CSV / parquet files into an Engine
2. extract_features     — compute features from transactional history
3. transform_features   — apply the sklearn preprocessing pipeline
4. score_clients        — apply the trained uplift model
5. select_clients       — pick clients with positive expected net profit
6. export_submission    — write the final CSV for submission

The @flow `run_campaign` orchestrates all tasks and is the main entry point.
"""

from __future__ import annotations

import json
import os
import datetime

import mlflow
import pandas as pd
import sklearn.pipeline as skpipe
from prefect import flow, task, get_run_logger
from prefect.artifacts import create_markdown_artifact

from src.datalib import Engine, build_engine
from src.datalib.features import extract_features
from src.model_utils import ModelKeeper
from src.utils import load_json, load_pickle

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PRICE_PER_GRAM = 80
COST_PER_GRAM = 52
DISCOUNT_USD = 40
CONTACT_COST = 1
MARGIN_PER_GRAM = PRICE_PER_GRAM - COST_PER_GRAM # default - 28

RETRIES_PER_TASK = 1 # 1 retry is cheap, but helps with accidental crashes immensely

# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

@task(name="load-data", retries=RETRIES_PER_TASK)
def load_data(data_root: str) -> Engine:
    """Read raw data files into the Engine"""
    logger = get_run_logger()
    engine = build_engine(data_root)
    for name in ("customers", "receipts", "campaigns"):
        logger.info("  %-12s %d rows", name, len(engine.get_table(name)))
    return engine

@task(name="extract-features", retries=RETRIES_PER_TASK)
def extract_features_task(engine: Engine, extract_config: list[dict]) -> pd.DataFrame:
    """Run all registered feature calculators and join results"""
    logger = get_run_logger()
    df = extract_features(engine, extract_config)
    logger.info("Extracted features shape: %s", df.shape)
    return df

@task(name="transform-features")
def transform_features_task(
    raw_features: pd.DataFrame,
    pipeline: skpipe.Pipeline,
) -> pd.DataFrame:
    """Apply the sklearn transform pipeline (fill NaN, encode location, etc.)"""
    logger = get_run_logger()
    df = pipeline.transform(raw_features)
    logger.info("Transformed features shape: %s", df.shape)
    return df

@task(name="score-clients")
def score_clients_task(
    features: pd.DataFrame,
    model: ModelKeeper,
    score_col: str = "uplift_score",
) -> pd.DataFrame:
    """Apply the uplift model - predict CATE for each client"""
    logger = get_run_logger()
    features = features.copy()
    features[score_col] = model.predict(features)
    logger.info(
        "Score stats: mean=%.4f  median=%.4f  >0=%.1f%%",
        features[score_col].mean(),
        features[score_col].median(),
        (features[score_col] > 0).mean() * 100,
    )
    return features

@task(name="select-clients")
def select_clients_task(
    features: pd.DataFrame,
    score_col: str,
    threshold: float,
) -> pd.DataFrame:
    """Select clients whose predicted CATE exceeds the threshold.

    The model is trained on target_profit (campaign costs already baked in),
    so CATE is in net-profit units (dollars).  A customer is profitable to target
    when CATE > 0 — no additional cost formula is applied here because that
    would double-count the discounts and communication cost.
    """
    logger = get_run_logger()
    cate = features[score_col]
    selected = features.loc[cate > threshold, ["customer_id"]].copy()
    logger.info(
        "Selected %d / %d clients (%.1f%%)  |  Mean CATE (selected) = %.4f dollars",
        len(selected),
        len(features),
        len(selected) / len(features) * 100,
        cate[cate > threshold].mean() if (cate > threshold).any() else 0.0,
    )
    return selected

@task(name="export-submission")
def export_submission_task(selected: pd.DataFrame, output_path: str,) -> str:
    """Write the submission CSV - a format required by the system"""
    logger = get_run_logger()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    selected[["customer_id"]].to_csv(output_path, index=False)
    logger.info("Submission saved to %s  (%d customers)", output_path, len(selected))
    return output_path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _prepare_extract_config(artifact_path: str, date_to: str) -> list[dict]:
    """Load the extract config JSON and inject the date"""
    cfg = load_json(artifact_path)
    for calcer_cfg in cfg:
        args = calcer_cfg.get("args", {})
        for key in list(args.keys()):
            if key.startswith("date"):
                args[key] = date_to
        if "date_to" not in args and calcer_cfg["name"] not in ("demographics",):
            args["date_to"] = date_to
        calcer_cfg["args"] = args
    return cfg

def _build_transform_pipeline(
        transform_config: list[dict],
        artifacts_root: str
) -> tuple[skpipe.Pipeline, ModelKeeper | None]:
    """Build the transform pipeline + model from campaign config"""
    steps = []
    model = None
    for part in transform_config:
        if part["type"] == "pipeline_pickle":
            sub: skpipe.Pipeline = load_pickle(
                os.path.join(artifacts_root, part["path"])
            )
            steps.extend(sub.steps)
        elif part["type"] == "model_apply":
            model = ModelKeeper.load(
                os.path.join(artifacts_root, part["model_path"])
            )
        else:
            raise ValueError(f"Unknown transform type: {part['type']}")
    pipeline = skpipe.Pipeline(steps) if steps else skpipe.Pipeline([("noop", "passthrough")])
    return pipeline, model

# ---------------------------------------------------------------------------
# Main Prefect Flow
# ---------------------------------------------------------------------------

@flow(name="smart-reach-campaign", log_prints=True)
def run_campaign(
    config_path: str = "configs/campaign.json",
    system_config_path: str = "configs/system.json",
    date_to: str | None = None,
    output_path: str = "runs/submission.csv",
) -> pd.DataFrame:
    """
    End-to-end uplift campaign pipeline for Smart Reach.

    Parameters
    ----------
    config_path : str
        Path to the campaign JSON config.
    system_config_path : str
        Path to the system JSON config (data paths, artifact paths).
    date_to : str or None
        Override the cut-off date (YYYY-MM-DD). Uses config default if None.
    output_path : str
        Where to save the submission CSV.
    """
    logger = get_run_logger()

    # ----- Load configs -----
    with open(system_config_path, "r") as f:
        sys_cfg = json.load(f)
    with open(config_path, "r") as f:
        campaign_cfg = json.load(f)

    data_root = sys_cfg["database"]["root_path"]
    artifacts_root = sys_cfg["artifacts_root_path"]
    run_date = date_to or campaign_cfg.get("date_to", "2019-03-20")

    # ----- MLflow setup -----
    mlflow_cfg = sys_cfg.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "mlruns")
    experiment_name = mlflow_cfg.get("experiment_name", "smart-reach-uplift")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    logger.info("=== Smart Reach Campaign ===")
    logger.info("Date: %s", run_date)
    logger.info("MLflow tracking: %s / experiment: %s", tracking_uri, experiment_name)

    # ----- Build pipeline + model from config -----
    extract_cfg = _prepare_extract_config(
        os.path.join(artifacts_root, campaign_cfg["extract"]),
        run_date,
    )
    transform_pipe, model = _build_transform_pipeline(
        campaign_cfg["transform"], artifacts_root
    )
    selection_cfg = campaign_cfg["selection"]

    # ----- Execute pipeline stages (inside MLflow run) -----
    with mlflow.start_run(run_name=f"campaign-{run_date}") as ml_run:
        # log parameters
        mlflow.log_param("date_to", run_date)
        mlflow.log_param("config_path", config_path)
        mlflow.log_param("threshold", selection_cfg["threshold"])
        mlflow.log_param("score_column", selection_cfg["score_column"])
        mlflow.log_param("n_extract_calcers", len(extract_cfg))

        # execute the full pipeline
        engine = load_data(data_root)
        raw_features = extract_features_task(engine, extract_cfg)
        features = transform_features_task(raw_features, transform_pipe)

        if model is not None:
            features = score_clients_task(features, model, selection_cfg["score_column"])

        selected = select_clients_task(
            features,
            score_col=selection_cfg["score_column"],
            threshold=selection_cfg["threshold"]
        )
        saved_path = export_submission_task(selected, output_path)

        # log metrics to MLflow
        # CATE is in net-profit units (dollars) because target_profit has costs baked in.
        score_col = selection_cfg["score_column"]
        if score_col in features.columns:
            cate = features[score_col]
            threshold = selection_cfg["threshold"]
            mlflow.log_metric("mean_cate_dollars", float(cate.mean()))
            mlflow.log_metric("median_cate_dollars", float(cate.median()))
            mlflow.log_metric("std_cate_dollars", float(cate.std()))
            mlflow.log_metric("pct_positive_cate", float((cate > 0).mean() * 100))
            mlflow.log_metric("total_expected_profit_selected",
                              float(cate[cate > threshold].sum()))
        mlflow.log_metric("n_total", len(features))
        mlflow.log_metric("n_selected", len(selected))
        mlflow.log_metric("selection_rate", len(selected) / len(features) * 100)

        # log submission as artifact
        mlflow.log_artifact(saved_path, artifact_path="submissions")

        logger.info("MLflow run ID: %s", ml_run.info.run_id)

    # ----- Create a Prefect Artifact with a summary -----
    summary = (
        f"## Campaign Run Summary\n\n"
        f"- **Date**: {run_date}\n"
        f"- **Total clients scored**: {len(features)}\n"
        f"- **Clients selected**: {len(selected)}\n"
        f"- **Selection rate**: {len(selected)/len(features)*100:.1f}%\n"
        f"- **CATE threshold**: {selection_cfg['threshold']} dollars\n"
        f"- **Output**: `{output_path}`\n"
        f"- **MLflow run**: `{ml_run.info.run_id}`\n"
    )
    create_markdown_artifact(summary, key="campaign-summary")

    return selected
        