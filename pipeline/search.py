"""Strategy discovery — generate candidates, run each through the pipeline, filter.

This is the funnel the demo exists to show:

    search space  →  N candidate strategies
                     each: train (CNN→PPO) → GA risk overlay → validation backtest
                                           → Monte Carlo
                  →  acceptance gates
                  →  survivors
                  →  test-set confirmation (reported, never used to select)

Selection discipline: candidates are ranked and gated **only** on the validation
contract. The test contract is touched once, after survivors are chosen, so the
number in the final report is a genuine out-of-sample read rather than the best
of N peeks.
"""
import copy
import json
import shutil
import time
from pathlib import Path

import numpy as np

from . import evaluate, features, train
from .gates import apply_gates
from .risk import ga


def sample_candidates(cfg: dict, rng: np.random.Generator) -> list[dict]:
    """Random search over the configured space, de-duplicated.

    Random beats grid here: it covers more of each dimension for the same
    budget. Swap in a grid or Bayesian sampler without touching anything else.
    """
    space = cfg["discovery"]["space"]
    n_wanted = cfg["discovery"]["max_candidates"]
    seen, out = set(), []
    for _ in range(n_wanted * 50):
        if len(out) >= n_wanted:
            break
        pick = {k: space[k][int(rng.integers(len(space[k])))] for k in space}
        sig = json.dumps(pick, sort_keys=True)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(pick)
    return out


def apply_candidate(cfg: dict, cand: dict, artifacts_dir: Path) -> dict:
    """Project one candidate onto a full config the normal stages can run."""
    c = copy.deepcopy(cfg)
    c["features"]["active"] = list(cand["features"])
    c["train"]["lookback"] = cand["lookback"]
    c["train"]["action_space"] = cand["action_space"]
    c["train"]["cnn"]["channels"] = list(cand["cnn_channels"])
    c["train"]["ppo"]["ent_coef"] = cand["ent_coef"]
    c["train"]["ppo"]["total_timesteps"] = cfg["discovery"]["train_timesteps"]
    c["artifacts_dir"] = str(artifacts_dir)
    return c


def _ensure_features(cfg: dict):
    """Build the feature table once, as the union of everything the space needs.

    Candidates then read their own subset of columns straight from the DB
    instead of recomputing indicators per candidate.
    """
    union, seen = [], set()
    for combo in cfg["discovery"]["space"]["features"]:
        for f in combo:
            if f not in seen:
                seen.add(f)
                union.append(f)
    union_cfg = copy.deepcopy(cfg)
    union_cfg["features"]["active"] = union
    features.run(union_cfg)
    return union


def run(cfg: dict, progress=None) -> dict:
    disc = cfg["discovery"]
    root = Path(cfg["artifacts_dir"]) / "discovery"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(disc["seed"])
    candidates = sample_candidates(cfg, rng)
    union = _ensure_features(cfg)
    t0 = time.time()

    records = []
    for i, cand in enumerate(candidates):
        cand_dir = root / f"candidate_{i + 1:02d}"
        cand_dir.mkdir(parents=True, exist_ok=True)
        c = apply_candidate(cfg, cand, cand_dir)

        train.run(c)
        ga_out = ga.run(c)
        agent, meta = train.load_trained(c)
        val = evaluate.evaluate_on(c, c["train"]["val_contract"], agent, meta,
                                   ga_out["genome"])
        verdict = apply_gates(val["metrics"], val["risk"], c)

        records.append({
            "id": i + 1,
            "candidate": cand,
            "risk_genome": ga_out["genome"],
            "validation": {
                "contract": c["train"]["val_contract"],
                "metrics": val["metrics"],
                "monte_carlo": {k: v for k, v in val["risk"].get("monte_carlo", {}).items()
                                if not k.startswith("_")},
            },
            "gates": verdict,
        })
        status = "PASS" if verdict["passed"] else "fail"
        print(f"    candidate {i + 1:2d}/{len(candidates)}  {status:4}  "
              f'PF {val["metrics"]["profit_factor"]:.2f}  '
              f'DD {val["metrics"]["max_drawdown_pct"]:.1f}%  '
              f'trades {val["metrics"]["n_trades"]:4d}'
              + ("" if verdict["passed"] else f'  ← {", ".join(verdict["failed_gates"])}'))
        if progress:
            n_pass = sum(r["gates"]["passed"] for r in records)
            progress(int(100 * (i + 1) / len(candidates)), {
                "Phase": "strategy discovery",
                "Candidates tested": f"{i + 1}/{len(candidates)}",
                "Passing gates": str(n_pass),
                "Search space": f'{len(cfg["discovery"]["space"])} dimensions',
            })

    # Survivors ranked by validation fitness; test contract touched only now.
    survivors = [r for r in records if r["gates"]["passed"]]
    survivors.sort(key=lambda r: ga.fitness(r["validation"]["metrics"]), reverse=True)
    for r in survivors:
        c = apply_candidate(cfg, r["candidate"], root / f'candidate_{r["id"]:02d}')
        agent, meta = train.load_trained(c)
        test = evaluate.evaluate_on(c, c["train"]["test_contract"], agent, meta,
                                    r["risk_genome"])
        r["test_confirmation"] = {
            "contract": c["train"]["test_contract"],
            "metrics": test["metrics"],
            "monte_carlo": {k: v for k, v in test["risk"].get("monte_carlo", {}).items()
                            if not k.startswith("_")},
        }

    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "search_space": disc["space"],
        "feature_union_built": union,
        "gates": disc["gates"],
        "selection_policy": (
            "Ranked and gated on the validation contract only; the test contract "
            "is evaluated once, after selection, for survivors."
        ),
        "funnel": {
            "candidates_generated": len(records),
            "passed_gates": len(survivors),
            "survivor_ids": [r["id"] for r in survivors],
        },
        "survivors": survivors,
        "all_candidates": records,
        "seconds": round(time.time() - t0, 1),
    }
    (root / "discovery_report.json").write_text(json.dumps(report, indent=2))
    return {"candidates": len(records), "survivors": len(survivors),
            "seconds": report["seconds"], "report": str(root / "discovery_report.json")}
