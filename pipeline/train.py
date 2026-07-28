"""Stage 3 — Model training: CNN encoder → PPO agent on 5-min bars.

Encoder and agent come from the model registries, so config.yaml decides the
architecture. Saves policy weights, the feature scaler, and a training log to
artifacts/ for the risk and evaluation stages.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from . import dataset, db
from .env import ACTION_SPACES, TradingEnv
from .models import AGENT_REGISTRY, ENCODER_REGISTRY
from .models.ppo import compute_gae


def pick_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def build_agent(cfg: dict, n_features: int, device: str):
    tc = cfg["train"]
    action_map = ACTION_SPACES[tc["action_space"]]
    encoder_cls = ENCODER_REGISTRY[tc["encoder"]]
    encoder = encoder_cls(n_features=n_features, lookback=tc["lookback"], **tc[tc["encoder"]])
    return AGENT_REGISTRY[tc["agent"]](encoder, tc[tc["agent"]], device,
                                       n_actions=len(action_map))


def run(cfg: dict, progress=None) -> dict:
    tc = cfg["train"]
    pc = tc["ppo"]
    device = pick_device(tc["device"])
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    con = db.connect(cfg["data"]["db_path"])
    feature_names = cfg["features"]["active"]
    train_sets = [dataset.load_contract(con, c, feature_names) for c in tc["train_contracts"]]
    con.close()

    scaler = dataset.fit_scaler(train_sets)
    train_sets = [dataset.apply_scaler(d, scaler) for d in train_sets]

    # Reward on drift-adjusted returns (training only — backtests keep real
    # returns): with raw drifting prices PPO collapses to unconditional
    # buy-and-hold instead of learning to time with the features.
    env_sets = train_sets
    if pc.get("demean_returns", True):
        env_sets = [{**d, "log_returns": d["log_returns"] - d["log_returns"].mean()}
                    for d in train_sets]

    env = TradingEnv(env_sets, ACTION_SPACES[tc["action_space"]], tc["lookback"],
                     pc["episode_len"], pc["cost_bp"], rng)
    agent = build_agent(cfg, len(feature_names), device)

    total, n_steps = pc["total_timesteps"], pc["n_steps"]
    reward_scale = pc.get("reward_scale", 100.0)
    obs = env.reset()
    steps_done, ep_rewards, recent_ep = 0, [], []
    log = []
    t0 = time.time()

    while steps_done < total:
        buf = {k: [] for k in ("windows", "positions", "actions", "logprobs",
                               "rewards", "values", "dones")}
        ep_r = 0.0
        for _ in range(n_steps):
            window, pos = obs
            action, logp, value = agent.act(window, pos)
            nxt, reward, done = env.step(action)
            buf["windows"].append(window)
            buf["positions"].append(pos)
            buf["actions"].append(action)
            buf["logprobs"].append(logp)
            buf["rewards"].append(reward * reward_scale)
            buf["values"].append(value)
            buf["dones"].append(float(done))
            ep_r += reward
            obs = env.reset() if done else nxt
            if done:
                recent_ep.append(ep_r)
                ep_r = 0.0
        window, pos = obs
        _, _, last_value = agent.act(window, pos)
        advs, rets = compute_gae(buf["rewards"], buf["values"], buf["dones"],
                                 last_value, pc["gamma"], pc["gae_lambda"])
        stats = agent.update({
            "windows": np.asarray(buf["windows"], dtype=np.float32),
            "positions": np.asarray(buf["positions"], dtype=np.float32),
            "actions": np.asarray(buf["actions"], dtype=np.int64),
            "logprobs": np.asarray(buf["logprobs"], dtype=np.float32),
            "advantages": advs,
            "returns": rets,
        })
        steps_done += n_steps
        mean_ep = float(np.mean(recent_ep[-20:])) if recent_ep else 0.0
        log.append({"steps": steps_done, "mean_ep_reward": mean_ep, **stats})
        if progress:
            progress(int(100 * steps_done / total), {
                "Model": f'{tc["encoder"].upper()} → {tc["agent"].upper()}',
                "Timesteps": f"{steps_done:,}/{total:,}",
                "Mean episode reward": f"{mean_ep:.5f}",
                "Entropy": f'{stats["entropy"]:.3f}',
            })

    art = Path(cfg["artifacts_dir"])
    art.mkdir(parents=True, exist_ok=True)
    agent.save(art / "policy.pt")
    meta = {
        "features": feature_names,
        "lookback": tc["lookback"],
        "encoder": tc["encoder"],
        "agent": tc["agent"],
        "action_space": tc["action_space"],
        "encoder_cfg": tc[tc["encoder"]],
        "scaler": scaler,
        "train_contracts": tc["train_contracts"],
        "timesteps": total,
        "train_seconds": round(time.time() - t0, 1),
    }
    (art / "model_meta.json").write_text(json.dumps(meta, indent=2))
    (art / "train_log.json").write_text(json.dumps(log, indent=2))
    return {"timesteps": total, "mean_ep_reward": log[-1]["mean_ep_reward"],
            "seconds": meta["train_seconds"], "device": device}


def load_trained(cfg: dict):
    """Rebuild the trained agent + scaler from artifacts (used by risk/evaluate)."""
    art = Path(cfg["artifacts_dir"])
    meta = json.loads((art / "model_meta.json").read_text())
    device = pick_device(cfg["train"]["device"])
    agent = build_agent(cfg, len(meta["features"]), device)
    agent.load(art / "policy.pt")
    return agent, meta
