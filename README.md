<div align="center">

# 🧠 AI Pipeline Demo

### DB-seeded, end-to-end AI trading pipeline — MNQ 5-minute strategy

SQLite market DB → feature engineering → CNN encoder → PPO agent → GA risk optimization → Monte Carlo evaluation → automated strategy discovery — with optional live telemetry into [AlgoDashboard](https://github.com/Dekeviin/AlgoDashboard).

![Python](https://img.shields.io/badge/Python_3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

</div>

---

## ✦ About

A working, scaled-down replica of a production algo-trading research pipeline. In production the market DB is fed by **NinjaTrader / IBKR** writers; this demo seeds the identical schema from sample NT-format MNQ exports (2022–2025, 1.3M 1-minute bars) so the whole flow runs offline in minutes:

```
┌───────────┐   ┌──────────────┐   ┌───────────────────┐   ┌────────────────┐   ┌──────────────────┐
│  SQLite    │   │   Feature    │   │      Training     │   │      Risk      │   │    Evaluation    │
│  seed DB   │──▶│  engineering │──▶│    CNN encoder    │──▶│  GA optimizes  │──▶│  OOS backtest +  │
│ (NT/IBKR)  │   │  5 indicators│   │     → PPO agent   │   │  risk overlay  │   │   Monte Carlo    │
└───────────┘   └──────────────┘   └───────────────────┘   └────────────────┘   └──────────────────┘
     1-min → 5-min      EMA 9/21          lookback window        stop / target       2,000 bootstrap
     bars, per          RSI, MACD,        of 32 bars ×           sizing, daily       equity paths,
     contract           MACD hist         5 features             circuit breaker     P(ruin), VaR
```

**Proper split & no leakage:** the model trains on MAR/JUN/SEP 24, the GA tunes risk on DEC 24 (never trained on), and the final report runs on MAR 25 (never seen by either). Fills are next-bar-open, stops checked before targets intrabar, commission + slippage on every fill.

**What this demo is (and isn't):** the product here is the *pipeline* — clean data flow, honest evaluation, scalable registries. A ~1-minute CPU training run on 5 indicators has a razor-thin edge (profit factor ≈ 1.0), and the Monte Carlo stage says so out loud — that's the point of having a risk stage. Real alpha comes from feeding it more features, more compute, and more model — which is exactly what the extension points below are for.

---

## ✦ Quick start

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

.venv\Scripts\python run_pipeline.py                # full run, ~2–3 min on CPU
.venv\Scripts\python run_pipeline.py --stage train  # or re-run any single stage
```

Everything lands in `artifacts/`: `report.json`, `trades.csv`, `equity_curve.csv/png`, `monte_carlo.png`, `policy.pt`, `risk_genome.json`, `train_log.json`.

---

## ✦ The stages

| Stage | Module | What it does |
|---|---|---|
| **1 · Ingestion** | `pipeline/ingest.py` | Parses NT `Last` exports (`yyyyMMdd HHmmss;O;H;L;C;V`), seeds `bars_1m`, resamples to `bars_5m`. The **only** stage that touches raw files — swap in a live NT/IBKR writer and nothing downstream changes. |
| **2 · Features** | `pipeline/features.py` | 5 fast indicators — EMA(9), EMA(21), RSI(14), MACD line, MACD histogram — all price-relative/bounded (stationary, leak-free), stored back in the DB. |
| **3 · Training** | `pipeline/train.py` | 1-D **CNN** treats each indicator as a channel and convolves across the 32-bar window; a **PPO** actor-critic trains on random episode segments with switching friction in the reward. Action space is configurable — the demo uses long/short (direction only; the risk overlay manages position), `long_flat_short` is one config line away. |
| **4a · Risk GA** | `pipeline/risk/ga.py` | Policy frozen; a **genetic algorithm** (tournament selection, uniform crossover, gaussian mutation, elitism) evolves the risk genome — ATR stop/target multiples, %-equity sizing, daily-loss circuit breaker — on validation data. Fitness = return − ½·max drawdown. |
| **4b · Evaluation** | `pipeline/evaluate.py` | Frozen policy + winning genome on the untouched test contract, then every registered risk procedure — the demo ships **Monte Carlo** (2,000 bootstrap resamples of the trade P&Ls → terminal-equity percentiles, drawdown distribution, P(ruin), trade VaR). |
| **5 · Discovery** | `pipeline/search.py` | The funnel on top — samples N strategy configs, runs each through the whole pipeline, and keeps only those clearing every **acceptance gate**. See below. |

---

## ✦ Strategy discovery — the funnel

Stages 1–4 evaluate *one* strategy. Stage 5 is the part that makes it a research system: it generates candidates and throws most of them away.

```bash
.venv\Scripts\python run_discovery.py                  # search, gate, report survivors
.venv\Scripts\python run_discovery.py --refresh-data   # re-ingest first
```

```
search space  ──▶  N candidates  ──▶  each: train → GA risk → validation → Monte Carlo
                                              │
                                              ▼
                                       acceptance gates
                                    (trades, profit factor,
                                     Sharpe, drawdown, P(ruin))
                                              │
                                              ▼
                                         survivors ──▶ test-set confirmation
```

**Selection discipline is the whole point.** Candidates are ranked and gated **only** on the validation contract. The test contract is touched once, *after* survivors are chosen — so the final number is a real out-of-sample read, not the best of N peeks. Gates are a registry (`pipeline/gates.py`) with thresholds in `config.yaml`; a run that produces **zero** survivors is a valid, reported outcome, because a filter that never says no isn't a filter.

**Gates check the policy, not just the P&L.** The first version of this search let a candidate through with a profit factor of 1.30 and a Sharpe of 4.0 — it had collapsed to holding one position for all 18,464 bars, so it was buy-and-hold on a bullish December wearing a strategy's clothes. Outcome metrics structurally cannot catch that. Two degeneracy gates now do: `min_signal_changes` and `max_position_concentration_pct` (from `backtest.signal_diagnostics`), which also reject the opposite failure — a policy that flips position every single bar.

Each candidate varies its feature subset, lookback, CNN depth, and PPO entropy; the search is embarrassingly parallel, so a real run distributes candidates across workers. Output: `artifacts/discovery/discovery_report.json` — every candidate, its gate-by-gate verdict, and the survivors with test confirmation.

**Unattended runs.** `run_discovery.py` is built to be scheduled after the close (Windows Task Scheduler or cron — one-liners in the module docstring), so the search runs on its own and leaves a survivors report to read in the morning.

---

## ✦ Built to scale

Every extension point is a **registry + config key** — add a module, list it in `config.yaml`, done. The trainer/evaluator never import concrete implementations.

| Want to add… | Where | How |
|---|---|---|
| An indicator | `pipeline/features.py` | `@feature("vwap")` + add to `features.active` |
| A deeper CNN | `config.yaml` | append to `train.cnn.channels` — each entry is a conv block |
| A new encoder (LSTM, Transformer) | `pipeline/models/` | `@encoder("lstm")` + set `train.encoder: lstm` |
| A new agent (A2C, SAC-discrete) | `pipeline/models/` | `@agent("a2c")` + set `train.agent: a2c` |
| A different action space | `config.yaml` | `train.action_space: long_flat_short` (or add one in `pipeline/env.py`) |
| A risk gene (trailing stop, time stop) | `config.yaml` + `backtest.py` | add bounds under `risk.ga.genes`, read it in `run_overlay` |
| A risk procedure (VaR, Kelly, stress) | `pipeline/risk/` | `@risk_procedure("var")` + add to `risk.procedures` |
| An acceptance gate (min win rate, max exposure) | `pipeline/gates.py` | `@gate("min_win_rate")` + add to `discovery.gates` |
| A wider strategy search | `config.yaml` | extend `discovery.space` / raise `max_candidates` |
| Another instrument / timeframe | `config.yaml` | drop exports in `data/raw`, set `timeframe_min` |

---

## ✦ AlgoDashboard live link

The pipeline can mirror its progress into [AlgoDashboard](https://github.com/Dekeviin/AlgoDashboard)'s **AI-pipeline telemetry panel** (the four-stage Ingestion → Features → Training → Testing card) by writing the dashboard's own `pipeline_meta` / `pipeline_stages` / `pipeline_details` SQLite tables as stages run:

```bash
.venv\Scripts\python run_pipeline.py --sync-dashboard
```

Stage progress, per-stage details (rows ingested, timesteps, mean reward, GA generation, OOS return, MC P(ruin)…) update live; the dashboard's running-stage animation tracks the actual run. Off by default (`dashboard.enabled` in `config.yaml`) — note it overwrites the dashboard's seeded demo telemetry.

---

## ✦ Config is the contract

Every parameter — contracts, timeframe, feature set, CNN depth, PPO hyper-parameters, GA bounds, Monte Carlo paths, fees — lives in [`config.yaml`](config.yaml). A run is reproducible from that one file plus the seed DB.

