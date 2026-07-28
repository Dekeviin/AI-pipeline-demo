"""5-minute trading environment for RL training.

Episodes are random contiguous segments from one contract (never crossing
contract boundaries). Observation = (n_features, lookback) window + current
position; reward = position × next-bar log return − switching friction.
"""
import numpy as np


# Action spaces are config-selectable (train.action_space). long_short keeps the
# agent always in the market — direction only, position management belongs to the
# risk overlay. long_flat_short adds a flat action, but beware: with a weak edge
# PPO can collapse into never trading.
ACTION_SPACES = {
    "long_short": {0: -1, 1: 1},
    "long_flat_short": {0: -1, 1: 0, 2: 1},
}


class TradingEnv:
    def __init__(self, datasets: list[dict], action_map: dict[int, int], lookback: int,
                 episode_len: int, cost_bp: float, rng: np.random.Generator):
        """datasets: [{'features': (T,F) float32, 'log_returns': (T,) float32}, ...]"""
        self.datasets = datasets
        self.action_map = action_map
        self.lookback = lookback
        self.episode_len = episode_len
        self.cost = cost_bp / 10_000.0
        self.rng = rng
        self._reset_state()

    def _reset_state(self):
        self.data = None
        self.t = 0
        self.end = 0
        self.pos = 0

    def reset(self):
        d = self.datasets[self.rng.integers(len(self.datasets))]
        T = len(d["log_returns"])
        span = min(self.episode_len, T - self.lookback - 1)
        start = int(self.rng.integers(self.lookback, T - span))
        self.data, self.t, self.end, self.pos = d, start, start + span, 0
        return self._obs()

    def _obs(self):
        window = self.data["features"][self.t - self.lookback + 1: self.t + 1].T  # (F, L)
        return np.ascontiguousarray(window), float(self.pos)

    def step(self, action: int):
        new_pos = self.action_map[action]
        ret = float(self.data["log_returns"][self.t + 1])
        reward = new_pos * ret - self.cost * abs(new_pos - self.pos)
        self.pos = new_pos
        self.t += 1
        done = self.t >= self.end
        return (None, reward, done) if done else (self._obs(), reward, done)
