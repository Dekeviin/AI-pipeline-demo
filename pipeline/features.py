"""Stage 2 — Feature engineering, registry-based.

Each indicator is a small function registered under a name; config.yaml picks
which ones run (`features.active`) and with what params. To scale up, add a
function here with @feature("name") and list it in the config — no other code
changes. All transforms are price-relative or bounded so they are roughly
stationary and leak-free (no fitted scalers over future data).
"""
import pandas as pd

from . import db

FEATURE_REGISTRY: dict[str, callable] = {}


def feature(name: str):
    def deco(fn):
        FEATURE_REGISTRY[name] = fn
        return fn
    return deco


def _ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False).mean()


@feature("ema_fast")
def ema_fast(bars: pd.DataFrame, period: int = 9) -> pd.Series:
    """Close distance from fast EMA, in %."""
    return (bars["close"] / _ema(bars["close"], period) - 1.0) * 100.0


@feature("ema_slow")
def ema_slow(bars: pd.DataFrame, period: int = 21) -> pd.Series:
    """Close distance from slow EMA, in %."""
    return (bars["close"] / _ema(bars["close"], period) - 1.0) * 100.0


@feature("rsi")
def rsi(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder RSI, centered to [-1, 1]."""
    delta = bars["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / loss.replace(0, 1e-9)
    return ((100 - 100 / (1 + rs)) - 50.0) / 50.0


@feature("macd")
def macd(bars: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD line, normalized by price (%)."""
    line = _ema(bars["close"], fast) - _ema(bars["close"], slow)
    return line / bars["close"] * 100.0


@feature("macd_hist")
def macd_hist(bars: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """MACD histogram (line − signal), normalized by price (%)."""
    line = _ema(bars["close"], fast) - _ema(bars["close"], slow)
    sig = _ema(line, signal)
    return (line - sig) / bars["close"] * 100.0


def compute(bars: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Feature matrix for one contract's bars, warm-up rows dropped."""
    active = cfg["features"]["active"]
    params = cfg["features"].get("params", {})
    out = pd.DataFrame(index=bars.index)
    for name in active:
        if name not in FEATURE_REGISTRY:
            raise KeyError(f"Unknown feature '{name}' — register it in pipeline/features.py")
        out[name] = FEATURE_REGISTRY[name](bars, **params.get(name, {}))
    warmup = 40  # longest indicator (MACD slow=26 + signal) needs ~35 bars to settle
    return out.iloc[warmup:].dropna()


def run(cfg: dict, progress=None) -> dict:
    con = db.connect(cfg["data"]["db_path"])
    contracts = db.list_contracts(con)
    total = 0
    for i, contract in enumerate(contracts):
        bars = db.read_bars(con, "bars_5m", contract)
        total += db.write_features(con, contract, compute(bars, cfg))
        if progress:
            progress(int(100 * (i + 1) / len(contracts)),
                     {"Features loaded": str(len(cfg["features"]["active"])),
                      "Timeframe": f'{cfg["data"]["timeframe_min"]} min',
                      "Feature rows": f"{total:,}"})
    con.close()
    return {"contracts": len(contracts), "features": cfg["features"]["active"], "rows": total}
