"""
CLI entry point for the Smart Reach campaign application.

Usage (one-shot run):
    python run_campaign.py

Usage (serve as a Prefect deployment):
    python run_campaign.py --serve

Usage (custom parameters):
    python run_campaign.py --config configs/campaign.json \\
                           --system-config configs/system.json \\
                           --date-to 2019-03-20 \\
                           -o runs/submission.csv
"""

import argparse
import sys

# MLflow prints emoji characters when finishing a run; force UTF-8 so it
# doesn't blow up on Windows terminals that default to cp1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.campaign_flow import run_campaign


def main():
    parser = argparse.ArgumentParser(
        prog="Smart Reach Campaign",
        description="Run the uplift-targeting campaign pipeline",
    )
    parser.add_argument(
        "--serve", action="store_true",
        help="Register as a Prefect deployment and serve (long-running).",
    )
    parser.add_argument("--config", default="configs/campaign.json")
    parser.add_argument("--system-config", default="configs/system.json")
    parser.add_argument("--date-to", default=None)
    parser.add_argument("-o", "--output", default="runs/submission.csv")
    args = parser.parse_args()

    params = {
        "config_path": args.config,
        "system_config_path": args.system_config,
        "date_to": args.date_to,
        "output_path": args.output,
    }

    if args.serve:
        run_campaign.serve(
            name="smart_reach_deployment",
            parameters=params,
        )
    else:
        run_campaign(**params)

if __name__ == "__main__":
    main()