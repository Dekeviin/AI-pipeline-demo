"""Stage 4b — Out-of-sample evaluation + risk procedures.

Runs the frozen policy under the GA-winning risk genome on the test contract
(never seen by training or the GA), then every risk procedure listed in
config.yaml (registry-based — Monte Carlo in the demo). Writes report.json,
trades.csv, equity_curve.csv and charts to artifacts/.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import backtest, dataset, db, train
from .env import ACTION_SPACES
from .risk import RISK_REGISTRY

# Palette: blue = realized series, sequential blue steps = MC envelope,
# neutral ink/grid; validated for the light surface.
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#33322e", "#6f6d66", "#e8e7e3"
BLUE, BLUE_250, BLUE_100 = "#2a78d6", "#86b6ef", "#cde2fb"


def _style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=10)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    for s in ax.spines.values():
        s.set_visible(False)


def plot_equity(result, out_path: Path, contract: str):
    fig, ax = plt.subplots(figsize=(9, 4), dpi=150, facecolor=SURFACE)
    _style(ax, f"Out-of-sample equity — {contract} (5-min, risk overlay applied)")
    ax.plot(result["index"], result["equity_curve"], color=BLUE, linewidth=1.6)
    ax.margins(x=0.01)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def plot_monte_carlo(mc: dict, initial_equity: float, out_path: Path):
    env = mc["_paths_envelope"]
    x = np.arange(1, len(env["p50"]) + 1)
    fig, ax = plt.subplots(figsize=(9, 4), dpi=150, facecolor=SURFACE)
    _style(ax, f'Monte Carlo bootstrap — {mc["n_paths"]:,} resampled trade sequences')
    ax.fill_between(x, env["p5"], env["p95"], color=BLUE_100, label="5–95th pct")
    ax.fill_between(x, env["p25"], env["p75"], color=BLUE_250, label="25–75th pct")
    ax.plot(x, env["p50"], color=BLUE, linewidth=1.6, label="Median")
    ax.axhline(initial_equity, color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Trade #", color=MUTED, fontsize=9)
    leg = ax.legend(loc="upper left", frameon=False, fontsize=8)
    for t in leg.get_texts():
        t.set_color(INK)
    ax.margins(x=0.01)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)


def run(cfg: dict, progress=None) -> dict:
    art = Path(cfg["artifacts_dir"])
    bt = cfg["backtest"]
    contract = cfg["train"]["test_contract"]

    agent, meta = train.load_trained(cfg)
    genome = json.loads((art / "risk_genome.json").read_text())["genome"]

    con = db.connect(cfg["data"]["db_path"])
    test = dataset.load_contract(con, contract, meta["features"])
    con.close()
    dataset.apply_scaler(test, meta["scaler"])

    if progress:
        progress(10, {"Phase": "computing policy signals", "Contract": contract})
    signals = backtest.compute_signals(agent, test, meta["lookback"],
                                       ACTION_SPACES[meta["action_space"]])
    result = backtest.run_overlay(test["bars"], signals, genome, bt)
    m = backtest.metrics(result, bt["initial_equity"])
    if progress:
        progress(50, {"Phase": "risk procedures", "Net return": f'{m["return_pct"]}%'})

    pnls = [t["pnl"] for t in result["trades"]]
    risk_reports = {}
    for name in cfg["risk"]["procedures"]:
        risk_reports[name] = RISK_REGISTRY[name](pnls, bt["initial_equity"], cfg)

    # Charts + tabular artifacts
    plot_equity(result, art / "equity_curve.png", contract)
    mc = risk_reports.get("monte_carlo", {})
    if "_paths_envelope" in mc:
        plot_monte_carlo(mc, bt["initial_equity"], art / "monte_carlo.png")
    pd.DataFrame(result["trades"]).to_csv(art / "trades.csv", index=False)
    pd.DataFrame({"ts": result["index"], "equity": result["equity_curve"]}).to_csv(
        art / "equity_curve.csv", index=False)

    report = {
        "test_contract": contract,
        "risk_genome": genome,
        "backtest": m,
        "risk_procedures": {
            name: {k: v for k, v in rep.items() if not k.startswith("_")}
            for name, rep in risk_reports.items()
        },
    }
    (art / "report.json").write_text(json.dumps(report, indent=2))

    if progress:
        summary = report["risk_procedures"].get("monte_carlo", {})
        progress(100, {
            "Test contract": contract,
            "Net return": f'{m["return_pct"]}%  (${m["net_pnl"]:,.0f})',
            "Max drawdown": f'{m["max_drawdown_pct"]}%',
            "Sharpe": str(m["sharpe"]),
            "MC median terminal": f'${summary.get("terminal_equity_p50", 0):,.0f}',
            "MC P(ruin)": f'{summary.get(f"prob_ruin_dd_over_{cfg["risk"]["monte_carlo"]["ruin_drawdown_pct"]}pct", "n/a")}%',
        })
    return report
