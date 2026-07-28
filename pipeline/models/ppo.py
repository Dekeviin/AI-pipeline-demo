"""PPO agent (clipped surrogate + GAE) on top of a pluggable encoder.

Actor and critic share the encoder embedding; the current position (−1/0/+1)
is concatenated to the embedding so the policy can reason about switching
costs. The discrete action set comes from pipeline.env.ACTION_SPACES.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from . import agent

class ActorCritic(nn.Module):
    def __init__(self, encoder: nn.Module, n_actions: int):
        super().__init__()
        self.encoder = encoder
        head_in = encoder.embed_dim + 1  # +1: current position scalar
        self.actor = nn.Sequential(nn.Linear(head_in, 64), nn.Tanh(), nn.Linear(64, n_actions))
        self.critic = nn.Sequential(nn.Linear(head_in, 64), nn.Tanh(), nn.Linear(64, 1))

    def forward(self, window: torch.Tensor, pos: torch.Tensor):
        z = torch.cat([self.encoder(window), pos.unsqueeze(-1)], dim=-1)
        return self.actor(z), self.critic(z).squeeze(-1)


@agent("ppo")
class PPOAgent:
    def __init__(self, encoder: nn.Module, cfg: dict, device: str = "cpu", n_actions: int = 3):
        self.cfg = cfg
        self.device = device
        self.net = ActorCritic(encoder, n_actions).to(device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg["lr"])

    @torch.no_grad()
    def act(self, window: np.ndarray, pos: float):
        """Sample an action for one observation. Returns (action, logprob, value)."""
        w = torch.as_tensor(window, dtype=torch.float32, device=self.device).unsqueeze(0)
        p = torch.as_tensor([pos], dtype=torch.float32, device=self.device)
        logits, value = self.net(w, p)
        dist = Categorical(logits=logits)
        a = dist.sample()
        return int(a.item()), float(dist.log_prob(a).item()), float(value.item())

    @torch.no_grad()
    def act_deterministic(self, windows: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Greedy actions for a batch — used by the backtester."""
        self.net.eval()
        w = torch.as_tensor(windows, dtype=torch.float32, device=self.device)
        p = torch.as_tensor(positions, dtype=torch.float32, device=self.device)
        logits, _ = self.net(w, p)
        self.net.train()
        return logits.argmax(dim=-1).cpu().numpy()

    def update(self, buf: dict) -> dict:
        """One PPO update over a filled rollout buffer. Returns loss stats."""
        c = self.cfg
        device = self.device
        windows = torch.as_tensor(buf["windows"], dtype=torch.float32, device=device)
        positions = torch.as_tensor(buf["positions"], dtype=torch.float32, device=device)
        actions = torch.as_tensor(buf["actions"], dtype=torch.int64, device=device)
        old_logp = torch.as_tensor(buf["logprobs"], dtype=torch.float32, device=device)
        advs = torch.as_tensor(buf["advantages"], dtype=torch.float32, device=device)
        returns = torch.as_tensor(buf["returns"], dtype=torch.float32, device=device)
        advs = (advs - advs.mean()) / (advs.std() + 1e-8)

        n = len(actions)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        n_batches = 0
        for _ in range(c["epochs"]):
            for idx in torch.randperm(n, device=device).split(c["batch_size"]):
                logits, values = self.net(windows[idx], positions[idx])
                dist = Categorical(logits=logits)
                logp = dist.log_prob(actions[idx])
                ratio = torch.exp(logp - old_logp[idx])
                clipped = torch.clamp(ratio, 1 - c["clip_eps"], 1 + c["clip_eps"])
                policy_loss = -torch.min(ratio * advs[idx], clipped * advs[idx]).mean()
                value_loss = nn.functional.mse_loss(values, returns[idx])
                entropy = dist.entropy().mean()
                loss = policy_loss + c["vf_coef"] * value_loss - c["ent_coef"] * entropy

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()

                stats["policy_loss"] += float(policy_loss)
                stats["value_loss"] += float(value_loss)
                stats["entropy"] += float(entropy)
                n_batches += 1
        return {k: v / n_batches for k, v in stats.items()}

    def save(self, path: str):
        torch.save(self.net.state_dict(), path)

    def load(self, path: str):
        self.net.load_state_dict(torch.load(path, map_location=self.device))


def compute_gae(rewards, values, dones, last_value, gamma, lam):
    """Generalized advantage estimation over one rollout."""
    n = len(rewards)
    advs = np.zeros(n, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(n)):
        next_v = last_value if t == n - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        advs[t] = gae
    return advs, advs + np.asarray(values, dtype=np.float32)
