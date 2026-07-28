"""Shared feature-matrix loading + standardization (train fits, everyone reuses)."""
import numpy as np
import pandas as pd

from . import db


def load_contract(con, contract: str, feature_names: list[str]) -> dict:
    """Aligned features + bars + log returns for one contract."""
    feats = db.read_features(con, contract, feature_names)
    bars = db.read_bars(con, "bars_5m", contract).loc[feats.index]
    logret = np.log(bars["close"]).diff().fillna(0.0).to_numpy(dtype=np.float32)
    return {
        "contract": contract,
        "features": feats.to_numpy(dtype=np.float32),
        "log_returns": logret,
        "bars": bars,
        "index": feats.index,
    }


def fit_scaler(datasets: list[dict]) -> dict:
    x = np.concatenate([d["features"] for d in datasets])
    return {"mean": x.mean(axis=0).tolist(), "std": (x.std(axis=0) + 1e-8).tolist()}


def apply_scaler(dataset: dict, scaler: dict) -> dict:
    mean = np.asarray(scaler["mean"], dtype=np.float32)
    std = np.asarray(scaler["std"], dtype=np.float32)
    dataset["features"] = (dataset["features"] - mean) / std
    return dataset


def rolling_windows(features: np.ndarray, lookback: int) -> np.ndarray:
    """(T, F) → (T-lookback+1, F, lookback) windows ending at each bar."""
    sw = np.lib.stride_tricks.sliding_window_view(features, lookback, axis=0)  # (T-L+1, F, L)
    return np.ascontiguousarray(sw, dtype=np.float32)
