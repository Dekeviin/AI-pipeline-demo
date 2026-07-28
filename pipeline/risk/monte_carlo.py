"""Monte Carlo bootstrap over the strategy's trade distribution.

Resamples the realized trade P&Ls (with replacement) into thousands of
alternate equity paths — same edge, different orderings/draws — to answer:
how bad can a normal losing streak get, and what is the risk of ruin?
"""
import numpy as np

from . import risk_procedure


@risk_procedure("monte_carlo")
def run(trade_pnls: list[float], initial_equity: float, cfg: dict) -> dict:
    mc = cfg["risk"]["monte_carlo"]
    rng = np.random.default_rng(mc["seed"])
    pnls = np.asarray(trade_pnls, dtype=np.float64)
    if len(pnls) < 5:
        return {"error": f"only {len(pnls)} trades — not enough for Monte Carlo"}

    n_paths, n_trades = mc["n_paths"], len(pnls)
    draws = rng.choice(pnls, size=(n_paths, n_trades), replace=True)
    paths = initial_equity + np.cumsum(draws, axis=1)

    peaks = np.maximum.accumulate(np.maximum(paths, initial_equity), axis=1)
    dd_pct = (peaks - paths) / peaks * 100
    max_dd = dd_pct.max(axis=1)
    terminal = paths[:, -1]
    ruin_line = mc["ruin_drawdown_pct"]

    return {
        "n_paths": n_paths,
        "n_trades_resampled": n_trades,
        "terminal_equity_p5": round(float(np.percentile(terminal, 5)), 2),
        "terminal_equity_p50": round(float(np.percentile(terminal, 50)), 2),
        "terminal_equity_p95": round(float(np.percentile(terminal, 95)), 2),
        "max_drawdown_pct_p50": round(float(np.percentile(max_dd, 50)), 2),
        "max_drawdown_pct_p95": round(float(np.percentile(max_dd, 95)), 2),
        "prob_losing_period_pct": round(float((terminal < initial_equity).mean()) * 100, 1),
        f"prob_ruin_dd_over_{ruin_line}pct": round(float((max_dd > ruin_line).mean()) * 100, 2),
        "trade_var_95": round(float(-np.percentile(pnls, 5)), 2),
        # Percentile envelopes across the path — for the fan chart, not the report.
        "_paths_envelope": {
            "p5": np.percentile(paths, 5, axis=0).tolist(),
            "p25": np.percentile(paths, 25, axis=0).tolist(),
            "p50": np.percentile(paths, 50, axis=0).tolist(),
            "p75": np.percentile(paths, 75, axis=0).tolist(),
            "p95": np.percentile(paths, 95, axis=0).tolist(),
        },
    }
