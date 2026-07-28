"""AI Pipeline Demo — orchestrator.

    python run_pipeline.py                     # full run: ingest → features → train → risk → evaluate
    python run_pipeline.py --stage train       # run one stage (earlier artifacts must exist)
    python run_pipeline.py --sync-dashboard    # mirror progress into AlgoDashboard's pipeline panel

DB (NT/IBKR-style seed) → feature engineering → CNN encoder → PPO agent →
GA risk optimization → Monte Carlo evaluation. Every knob lives in config.yaml.
"""
import argparse
import json
import time

from pipeline.config import load_config
from pipeline.dashboard_sync import DashboardSync

STAGES = ["ingest", "features", "train", "risk", "evaluate"]
DASH_KEY = {"ingest": "ingestion", "features": "features", "train": "training",
            "risk": "testing", "evaluate": "testing", "discover": "testing"}


def main():
    ap = argparse.ArgumentParser(description="MNQ 5-min AI trading pipeline demo")
    ap.add_argument("--stage", choices=STAGES + ["all", "discover"], default="all")
    ap.add_argument("--config", default=None, help="alternate config.yaml")
    ap.add_argument("--sync-dashboard", action="store_true",
                    help="write live progress into AlgoDashboard's SQLite DB")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.sync_dashboard:
        cfg["dashboard"]["enabled"] = True
    dash = DashboardSync(cfg)

    from pipeline import evaluate, features, ingest, search, train
    from pipeline.risk import ga
    runners = {"ingest": ingest.run, "features": features.run, "train": train.run,
               "risk": ga.run, "evaluate": evaluate.run, "discover": search.run}
    stage_names = {"ingest": "Ingestion", "features": "Feature Engineering",
                   "train": "Training (CNN → PPO)", "risk": "Risk GA",
                   "evaluate": "Evaluation + Monte Carlo",
                   "discover": "Strategy Discovery"}

    todo = STAGES if args.stage == "all" else [args.stage]
    if args.stage == "all" and dash.enabled:
        for key in ("ingestion", "features", "training", "testing"):
            dash.update_stage(key, "queued", 0)

    results = {}
    for stage in todo:
        name = stage_names[stage]
        dash.set_meta(name, "2022 – 2025 (MNQ)")
        print(f"\n=== {name} " + "=" * max(0, 60 - len(name)))
        t0 = time.time()
        result = runners[stage](cfg, progress=dash.progress_cb(DASH_KEY[stage]))
        dt = time.time() - t0
        results[stage] = result
        printable = {k: v for k, v in result.items()
                     if not isinstance(v, (dict, list)) or k in ("genome",)}
        print(f"    done in {dt:.1f}s -> {json.dumps(printable, default=str)}")

    dash.set_meta("Idle — run complete", "2022 – 2025 (MNQ)")
    if "evaluate" in results:
        rep = results["evaluate"]
        print("\n=== Out-of-sample report =====================================")
        print(json.dumps({k: rep[k] for k in ("test_contract", "risk_genome", "backtest")}, indent=2))
        print(json.dumps(rep["risk_procedures"], indent=2))
        print(f'\nArtifacts written to {cfg["artifacts_dir"]}')


if __name__ == "__main__":
    main()
