"""Stage 4a — Genetic algorithm over the risk overlay.

The trained policy's signals are frozen; the GA evolves only the risk genome
(stop/target distances, position sizing, daily circuit breaker) on the
validation contract — data the model never trained on. Adding a gene is a
config edit (risk.ga.genes) plus one line reading it in backtest.run_overlay.
"""
import json
import time
from pathlib import Path

import numpy as np

from .. import backtest, dataset, db, train
from ..env import ACTION_SPACES


def fitness(m: dict) -> float:
    """Return% with a strong drawdown penalty; degenerate genomes lose.

    The 2× penalty matters: at 0.5× the GA maxes leverage (tight stops, big
    size) and Monte Carlo then shows >90% chance of a 30% drawdown.
    """
    if m["n_trades"] < 10:
        return -100.0
    return m["return_pct"] - 2.0 * m["max_drawdown_pct"]


def run(cfg: dict, progress=None) -> dict:
    ga = cfg["risk"]["ga"]
    bt = cfg["backtest"]
    rng = np.random.default_rng(ga["seed"])
    gene_names = list(ga["genes"].keys())
    bounds = np.array([ga["genes"][g] for g in gene_names])  # (G, 2)

    # Frozen policy → signals on the validation contract, computed once.
    agent, meta = train.load_trained(cfg)
    con = db.connect(cfg["data"]["db_path"])
    val = dataset.load_contract(con, cfg["train"]["val_contract"], meta["features"])
    con.close()
    dataset.apply_scaler(val, meta["scaler"])
    signals = backtest.compute_signals(agent, val, meta["lookback"],
                                       ACTION_SPACES[meta["action_space"]])

    def evaluate(genome_vec) -> tuple[float, dict]:
        genome = dict(zip(gene_names, genome_vec))
        result = backtest.run_overlay(val["bars"], signals, genome, bt)
        m = backtest.metrics(result, bt["initial_equity"])
        return fitness(m), m

    pop = rng.uniform(bounds[:, 0], bounds[:, 1], size=(ga["population"], len(gene_names)))
    scores = np.full(len(pop), -np.inf)
    stats = [None] * len(pop)
    history = []
    t0 = time.time()

    for gen in range(ga["generations"]):
        for i in range(len(pop)):
            scores[i], stats[i] = evaluate(pop[i])
        order = np.argsort(scores)[::-1]
        pop, scores = pop[order], scores[order]
        stats = [stats[i] for i in order]
        history.append({"generation": gen + 1, "best_fitness": round(float(scores[0]), 3),
                        "mean_fitness": round(float(scores.mean()), 3),
                        "best_metrics": stats[0]})
        if progress:
            progress(int(100 * (gen + 1) / ga["generations"]), {
                "Risk optimizer": f'GA gen {gen + 1}/{ga["generations"]}',
                "Best fitness": f"{scores[0]:.2f}",
                "Best val return": f'{stats[0]["return_pct"]:.1f}%',
                "Best val max DD": f'{stats[0]["max_drawdown_pct"]:.1f}%',
            })
        if gen == ga["generations"] - 1:
            break

        # Elitism (keep top 2) + tournament-selected offspring.
        next_pop = [pop[0].copy(), pop[1].copy()]
        while len(next_pop) < len(pop):
            def tourney():
                idx = rng.integers(len(pop), size=ga["tournament_k"])
                return pop[idx[np.argmin(idx)]]  # pop is fitness-sorted: lower index = fitter
            a, b = tourney(), tourney()
            child = np.where(rng.random(len(gene_names)) < 0.5, a, b) \
                if rng.random() < ga["crossover_rate"] else a.copy()
            mutate = rng.random(len(gene_names)) < ga["mutation_rate"]
            span = bounds[:, 1] - bounds[:, 0]
            child = child + mutate * rng.normal(0, ga["mutation_scale"], len(gene_names)) * span
            next_pop.append(np.clip(child, bounds[:, 0], bounds[:, 1]))
        pop = np.array(next_pop)

    best_genome = dict(zip(gene_names, (round(float(v), 5) for v in pop[0])))
    art = Path(cfg["artifacts_dir"])
    art.mkdir(parents=True, exist_ok=True)
    (art / "risk_genome.json").write_text(json.dumps({
        "genome": best_genome,
        "fitness": round(float(scores[0]), 3),
        "val_contract": cfg["train"]["val_contract"],
        "val_metrics": stats[0],
        "history": history,
        "seconds": round(time.time() - t0, 1),
    }, indent=2))
    return {"genome": best_genome, "val_metrics": stats[0],
            "generations": ga["generations"], "seconds": round(time.time() - t0, 1)}
