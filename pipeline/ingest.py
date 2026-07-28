"""Stage 1 — Ingestion: NinjaTrader exports → SQLite seed → 5-min bars.

Raw format (NT 'Last' export): `yyyyMMdd HHmmss;open;high;low;close;volume`.
Swapping the source for a live NT/IBKR writer means replacing this one stage;
everything downstream keeps reading the same two tables.
"""
from pathlib import Path

import pandas as pd

from . import db


def parse_nt_export(path: Path) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=";",
        header=None,
        names=["ts", "open", "high", "low", "close", "volume"],
    )
    df["ts"] = pd.to_datetime(df["ts"], format="%Y%m%d %H%M%S")
    return df.set_index("ts").sort_index()


def resample(df_1m: pd.DataFrame, minutes: int) -> pd.DataFrame:
    out = (
        df_1m.resample(f"{minutes}min", label="left", closed="left")
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
             close=("close", "last"), volume=("volume", "sum"))
        .dropna()
    )
    out["volume"] = out["volume"].astype(int)
    return out


def run(cfg: dict, progress=None) -> dict:
    """Seed the DB from every configured contract file. Returns summary stats."""
    data_cfg = cfg["data"]
    raw_dir = Path(data_cfg["raw_dir"])
    wanted = data_cfg["contracts"]
    files = sorted(raw_dir.glob("*.txt"))
    if wanted != "all":
        files = [f for f in files if f.stem in wanted]
    if not files:
        raise FileNotFoundError(f"No contract files found in {raw_dir}")

    con = db.connect(data_cfg["db_path"])
    total_1m = total_5m = 0
    first_ts, last_ts = None, None
    for i, f in enumerate(files):
        df_1m = parse_nt_export(f)
        df_5m = resample(df_1m, data_cfg["timeframe_min"])
        total_1m += db.write_bars(con, "bars_1m", f.stem, df_1m)
        total_5m += db.write_bars(con, "bars_5m", f.stem, df_5m)
        first_ts = min(first_ts or df_1m.index[0], df_1m.index[0])
        last_ts = max(last_ts or df_1m.index[-1], df_1m.index[-1])
        if progress:
            progress(int(100 * (i + 1) / len(files)),
                     {"Contracts loaded": f"{i + 1}/{len(files)}", "Rows ingested": f"{total_1m:,}"})
    con.close()
    return {
        "contracts": len(files),
        "rows_1m": total_1m,
        "rows_5m": total_5m,
        "range": f"{first_ts:%Y-%m-%d} – {last_ts:%Y-%m-%d}",
    }
