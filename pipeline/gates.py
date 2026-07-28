"""Acceptance gates — the tests a candidate strategy must survive.

A gate is a named pass/fail check over a candidate's validation metrics and
risk reports. `config.yaml` sets which gates run and at what threshold, so
tightening standards is a config edit, not a code change.

Scaling up: add a function here with @gate("name") and list it under
`discovery.gates`. Gates that read a risk procedure's output (like
max_prob_ruin_pct) work for any procedure registered in pipeline/risk/.
"""
GATE_REGISTRY: dict[str, callable] = {}


def gate(name: str):
    def deco(fn):
        GATE_REGISTRY[name] = fn
        return fn
    return deco


# Each gate: (metrics, risk_reports, threshold, cfg) -> (passed, observed_value)

@gate("min_trades")
def _min_trades(m, risk, thr, cfg):
    """Too few trades and every other statistic is noise."""
    return m["n_trades"] >= thr, m["n_trades"]


@gate("min_profit_factor")
def _min_profit_factor(m, risk, thr, cfg):
    return m["profit_factor"] >= thr, m["profit_factor"]


@gate("min_sharpe")
def _min_sharpe(m, risk, thr, cfg):
    return m["sharpe"] >= thr, m["sharpe"]


@gate("max_drawdown_pct")
def _max_drawdown(m, risk, thr, cfg):
    return m["max_drawdown_pct"] <= thr, m["max_drawdown_pct"]


@gate("min_signal_changes")
def _min_signal_changes(m, risk, thr, cfg):
    """Reject collapsed policies that hold one position the whole period.

    Without this a buy-and-hold agent passes profit factor and Sharpe on any
    trending validation contract — the demo caught exactly that.
    """
    return m["signal_changes"] >= thr, m["signal_changes"]


@gate("max_position_concentration_pct")
def _max_concentration(m, risk, thr, cfg):
    """Reject near-collapsed policies that sit in one position almost always."""
    return m["position_concentration_pct"] <= thr, m["position_concentration_pct"]


@gate("max_prob_ruin_pct")
def _max_prob_ruin(m, risk, thr, cfg):
    """Monte Carlo: chance a resampled path breaches the ruin drawdown."""
    mc = risk.get("monte_carlo", {})
    key = f'prob_ruin_dd_over_{cfg["risk"]["monte_carlo"]["ruin_drawdown_pct"]}pct'
    observed = mc.get(key)
    if observed is None:
        return False, None
    return observed <= thr, observed


@gate("max_prob_losing_period_pct")
def _max_prob_losing(m, risk, thr, cfg):
    """Monte Carlo: share of resampled paths that end below starting equity."""
    observed = risk.get("monte_carlo", {}).get("prob_losing_period_pct")
    if observed is None:
        return False, None
    return observed <= thr, observed


def apply_gates(m: dict, risk: dict, cfg: dict) -> dict:
    """Run every configured gate. Returns verdict + per-gate detail."""
    results = []
    for name, thr in cfg["discovery"]["gates"].items():
        if name not in GATE_REGISTRY:
            raise KeyError(f"Unknown gate '{name}' — register it in pipeline/gates.py")
        passed, observed = GATE_REGISTRY[name](m, risk, thr, cfg)
        results.append({"gate": name, "threshold": thr,
                        "observed": observed, "passed": bool(passed)})
    failed = [r["gate"] for r in results if not r["passed"]]
    return {"passed": not failed, "failed_gates": failed, "detail": results}
