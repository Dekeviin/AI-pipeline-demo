"""Shared backtester: trained policy → signals → risk-overlay simulation.

Used twice — the GA calls `run_overlay` hundreds of times with different
genomes (signals are computed once and reused), and the evaluation stage runs
the winning genome out-of-sample. Fills happen at the NEXT bar's open after a
signal (no lookahead); stops are checked before targets inside a bar
(conservative); commission + slippage on every fill.
"""
import numpy as np
import pandas as pd

from . import dataset


def compute_signals(agent, data: dict, lookback: int, action_map: dict[int, int]) -> np.ndarray:
    """Greedy policy positions per bar, position-feedback exact.

    The policy sees its own position, so greedy actions are precomputed for
    every possible position in batch, then the position chain is walked forward.
    """
    windows = dataset.rolling_windows(data["features"], lookback)  # ends at bar L-1..T-1
    n = len(windows)
    greedy = {}
    for pos in {0, *action_map.values()}:
        acts = []
        for start in range(0, n, 4096):
            batch = windows[start:start + 4096]
            acts.append(agent.act_deterministic(batch, np.full(len(batch), pos, dtype=np.float32)))
        greedy[pos] = np.concatenate(acts)
    signals = np.zeros(len(data["log_returns"]), dtype=np.int8)
    pos = 0
    for i in range(n):
        pos = action_map[int(greedy[pos][i])]
        signals[lookback - 1 + i] = pos
    return signals


def atr(bars: pd.DataFrame, period: int) -> np.ndarray:
    pc = bars["close"].shift(1)
    tr = pd.concat([bars["high"] - bars["low"],
                    (bars["high"] - pc).abs(),
                    (bars["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean().to_numpy()


def run_overlay(bars: pd.DataFrame, signals: np.ndarray, genome: dict, bt: dict) -> dict:
    """Simulate the policy's signals under one risk genome. Returns trades + equity."""
    o = bars["open"].to_numpy(); h = bars["high"].to_numpy()
    l = bars["low"].to_numpy(); c = bars["close"].to_numpy()
    dates = bars.index.date
    atr_v = atr(bars, bt["atr_period"])
    slip = bt["slippage_ticks"] * bt["tick_size"]
    pv, comm = bt["point_value"], bt["commission_per_side"]

    equity = float(bt["initial_equity"])
    curve = np.empty(len(bars)); curve[0] = equity
    trades = []
    in_trade = False
    direction = qty = 0
    entry = stop = target = 0.0
    entry_i = 0
    day = dates[0]
    day_pnl = 0.0
    blocked = False

    def close_trade(i, price, reason):
        nonlocal equity, in_trade, day_pnl
        pnl = direction * (price - entry) * pv * qty - comm * qty  # exit-side commission
        equity += pnl
        day_pnl += pnl
        risk_dollars = abs(entry - stop) * pv * qty
        trades.append({
            "entry_ts": str(bars.index[entry_i]), "exit_ts": str(bars.index[i]),
            "dir": int(direction), "qty": int(qty),
            "entry": round(entry, 2), "exit": round(price, 2),
            "pnl": round(pnl, 2),
            "r_mult": round(pnl / risk_dollars, 3) if risk_dollars > 0 else 0.0,
            "reason": reason,
        })
        in_trade = False

    for i in range(1, len(bars)):
        if dates[i] != day:
            day, day_pnl, blocked = dates[i], 0.0, False
        sig = int(signals[i - 1])  # decision made on the previous closed bar
        just_exited = False

        if in_trade:
            hit_stop = (l[i] <= stop) if direction > 0 else (h[i] >= stop)
            hit_target = (h[i] >= target) if direction > 0 else (l[i] <= target)
            if hit_stop:  # conservative: stop fills first if both touch
                close_trade(i, stop - direction * slip, "stop")
                just_exited = True
            elif hit_target:
                close_trade(i, target - direction * slip, "target")
                just_exited = True
            elif sig != direction:
                close_trade(i, o[i] - direction * slip, "signal")
                just_exited = True

        risk_dollars = equity * genome["risk_frac"]
        if day_pnl <= -genome["max_daily_loss_r"] * risk_dollars:
            blocked = True  # daily circuit breaker: no new entries until tomorrow

        # No same-bar re-entry: the bar's open is already in the past after an
        # intrabar exit, so entering at it again would be lookahead.
        if not in_trade and not just_exited and not blocked and sig != 0 and atr_v[i] > 0:
            direction = sig
            stop_dist = genome["stop_atr_mult"] * atr_v[i]
            qty = max(1, min(bt["max_contracts"], int(risk_dollars / (stop_dist * pv))))
            entry = o[i] + direction * slip
            equity -= comm * qty  # entry-side commission
            stop = entry - direction * stop_dist
            target = entry + direction * genome["target_atr_mult"] * atr_v[i]
            entry_i = i
            in_trade = True

        mark = direction * (c[i] - entry) * pv * qty if in_trade else 0.0
        curve[i] = equity + mark

    if in_trade:
        close_trade(len(bars) - 1, c[-1], "eod")
        curve[-1] = equity

    return {"trades": trades, "equity_curve": curve, "index": bars.index}


def metrics(result: dict, initial_equity: float, bars_per_year: int = 69_000) -> dict:
    curve = result["equity_curve"]
    trades = result["trades"]
    rets = np.diff(curve) / curve[:-1]
    peak = np.maximum.accumulate(curve)
    dd = (curve - peak) / peak
    pnl = np.array([t["pnl"] for t in trades]) if trades else np.array([0.0])
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    return {
        "net_pnl": round(float(curve[-1] - initial_equity), 2),
        "return_pct": round(float(curve[-1] / initial_equity - 1) * 100, 2),
        "max_drawdown_pct": round(float(-dd.min()) * 100, 2),
        "sharpe": round(float(rets.mean() / (rets.std() + 1e-12) * np.sqrt(bars_per_year)), 2),
        "n_trades": len(trades),
        "win_rate_pct": round(100 * len(wins) / max(len(pnl), 1), 1),
        "profit_factor": round(float(wins.sum() / max(-losses.sum(), 1e-9)), 2),
        "avg_r": round(float(np.mean([t["r_mult"] for t in trades])), 3) if trades else 0.0,
    }
