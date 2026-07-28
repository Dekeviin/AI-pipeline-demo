"""SQLite market database — the seed every downstream stage reads from.

In production this DB is populated by a NinjaTrader/IBKR feed writer; the demo
seeds it from sample NT-format exports instead. Either way, ingestion is the
only stage that touches raw files — features/train/risk/evaluate only query here.
"""
import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS bars_1m (
  contract TEXT NOT NULL,
  ts       TEXT NOT NULL,          -- ISO-8601, exchange time
  open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
  volume INTEGER NOT NULL,
  PRIMARY KEY (contract, ts)
);
CREATE TABLE IF NOT EXISTS bars_5m (
  contract TEXT NOT NULL,
  ts       TEXT NOT NULL,
  open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL, close REAL NOT NULL,
  volume INTEGER NOT NULL,
  PRIMARY KEY (contract, ts)
);
-- Wide feature matrix keyed to bars_5m; columns beyond (contract, ts) are
-- whatever the active feature set produced. Rebuilt by the features stage.
"""


def connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(SCHEMA)
    return con


def write_bars(con: sqlite3.Connection, table: str, contract: str, df: pd.DataFrame) -> int:
    """Replace all bars for one contract in `table`. df indexed by DatetimeIndex."""
    con.execute(f"DELETE FROM {table} WHERE contract = ?", (contract,))
    rows = [
        (contract, ts.isoformat(), r.open, r.high, r.low, r.close, int(r.volume))
        for ts, r in df.iterrows()
    ]
    con.executemany(
        f"INSERT INTO {table} (contract, ts, open, high, low, close, volume) VALUES (?,?,?,?,?,?,?)",
        rows,
    )
    con.commit()
    return len(rows)


def read_bars(con: sqlite3.Connection, table: str, contract: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        f"SELECT ts, open, high, low, close, volume FROM {table} WHERE contract = ? ORDER BY ts",
        con,
        params=(contract,),
        parse_dates=["ts"],
    )
    return df.set_index("ts")


def write_features(con: sqlite3.Connection, contract: str, df: pd.DataFrame) -> int:
    """Replace the feature matrix for one contract (table is rebuilt to match columns)."""
    cols = ", ".join(f'"{c}" REAL NOT NULL' for c in df.columns)
    con.execute("CREATE TABLE IF NOT EXISTS features (contract TEXT, ts TEXT)")
    # Rebuild if the column set changed (feature list is config-driven).
    existing = {r[1] for r in con.execute("PRAGMA table_info(features)")}
    wanted = {"contract", "ts", *df.columns}
    if existing != wanted:
        con.execute("DROP TABLE features")
        con.execute(
            f'CREATE TABLE features (contract TEXT NOT NULL, ts TEXT NOT NULL, {cols}, '
            f"PRIMARY KEY (contract, ts))"
        )
    con.execute("DELETE FROM features WHERE contract = ?", (contract,))
    placeholders = ",".join("?" * (2 + len(df.columns)))
    quoted = ", ".join(f'"{c}"' for c in df.columns)
    con.executemany(
        f'INSERT INTO features (contract, ts, {quoted}) VALUES ({placeholders})',
        [(contract, ts.isoformat(), *row) for ts, row in zip(df.index, df.itertuples(index=False))],
    )
    con.commit()
    return len(df)


def read_features(con: sqlite3.Connection, contract: str, columns: list[str]) -> pd.DataFrame:
    quoted = ", ".join(f'"{c}"' for c in columns)
    df = pd.read_sql_query(
        f"SELECT ts, {quoted} FROM features WHERE contract = ? ORDER BY ts",
        con,
        params=(contract,),
        parse_dates=["ts"],
    )
    return df.set_index("ts")


def list_contracts(con: sqlite3.Connection, table: str = "bars_5m") -> list[str]:
    return [r[0] for r in con.execute(f"SELECT DISTINCT contract FROM {table} ORDER BY contract")]
