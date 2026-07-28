"""Strategy discovery run — the scheduled entry point.

    python run_discovery.py                  # search, gate, report survivors
    python run_discovery.py --refresh-data   # re-ingest first (new session data)
    python run_discovery.py --sync-dashboard # mirror progress into AlgoDashboard

Designed to be called unattended after the close, e.g. Windows Task Scheduler:
    schtasks /create /tn "AlgoDiscovery" /tr "<repo>\\.venv\\Scripts\\python.exe
             <repo>\\run_discovery.py --refresh-data" /sc daily /st 17:30
or cron:  30 17 * * 1-5  cd <repo> && .venv/bin/python run_discovery.py --refresh-data

It prints a survivors report and writes artifacts/discovery/discovery_report.json.
"""
import argparse
import json

from pipeline import ingest, search
from pipeline.config import load_config
from pipeline.dashboard_sync import DashboardSync


def format_report(report: dict) -> str:
    funnel = report["funnel"]
    lines = [
        "",
        "=== Discovery report =========================================",
        f'  candidates generated : {funnel["candidates_generated"]}',
        f'  cleared every gate   : {funnel["passed_gates"]}',
        f'  gates applied        : {", ".join(report["gates"])}',
        f'  selection policy     : {report["selection_policy"]}',
        "",
    ]
    if not report["survivors"]:
        lines += ["  No candidate cleared every gate this run.",
                  "  (That is a valid outcome — the filter exists to say no.)"]
        return "\n".join(lines)

    lines.append("  Survivors (ranked on validation, confirmed on test):")
    for r in report["survivors"]:
        cand, v, t = r["candidate"], r["validation"]["metrics"], r["test_confirmation"]
        lines += [
            "",
            f'  #{r["id"]}  features={",".join(cand["features"])}',
            f'       lookback={cand["lookback"]}  cnn={cand["cnn_channels"]}  '
            f'ent_coef={cand["ent_coef"]}',
            f'       risk genome: ' + ", ".join(f"{k}={v2}" for k, v2 in r["risk_genome"].items()),
            f'       validation ({r["validation"]["contract"]}): '
            f'return {v["return_pct"]}%  PF {v["profit_factor"]}  '
            f'DD {v["max_drawdown_pct"]}%  Sharpe {v["sharpe"]}  trades {v["n_trades"]}',
            f'       TEST       ({t["contract"]}): '
            f'return {t["metrics"]["return_pct"]}%  PF {t["metrics"]["profit_factor"]}  '
            f'DD {t["metrics"]["max_drawdown_pct"]}%  Sharpe {t["metrics"]["sharpe"]}  '
            f'trades {t["metrics"]["n_trades"]}',
        ]
        mc = t.get("monte_carlo", {})
        if mc:
            ruin = next((f"{k}={v2}%" for k, v2 in mc.items() if k.startswith("prob_ruin")), "n/a")
            lines.append(f'       monte carlo: median terminal '
                         f'${mc.get("terminal_equity_p50", 0):,.0f}  '
                         f'DD p95 {mc.get("max_drawdown_pct_p95", 0)}%  {ruin}')
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Run strategy discovery and report survivors")
    ap.add_argument("--config", default=None)
    ap.add_argument("--refresh-data", action="store_true",
                    help="re-ingest raw exports before searching")
    ap.add_argument("--sync-dashboard", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.sync_dashboard:
        cfg["dashboard"]["enabled"] = True
    dash = DashboardSync(cfg)

    if args.refresh_data:
        dash.set_meta("Ingestion", "MNQ")
        print("=== Ingestion ===================================================")
        print(f"    {json.dumps(ingest.run(cfg, progress=dash.progress_cb('ingestion')))}")

    dash.set_meta("Strategy discovery", "MNQ")
    print("=== Strategy discovery ==========================================")
    summary = search.run(cfg, progress=dash.progress_cb("testing"))
    dash.set_meta("Idle — discovery complete", "MNQ")

    report = json.loads(open(summary["report"], encoding="utf-8").read())
    print(format_report(report))
    print(f'\n  full report: {summary["report"]}')


if __name__ == "__main__":
    main()
