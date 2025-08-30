from __future__ import annotations
from typing import Tuple
import numpy as np


class ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int = 1_000_000) -> None:
        self.obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rews_buf = np.zeros((capacity,), dtype=np.float32)
        self.done_buf = np.zeros((capacity,), dtype=np.float32)
        self.capacity = int(capacity)
        self.ptr = 0
        self.size = 0

    def store(self, obs: np.ndarray, act: np.ndarray, rew: float, next_obs: np.ndarray, done: bool) -> None:
        idx = self.ptr
        self.obs_buf[idx] = obs
        self.acts_buf[idx] = act
        self.rews_buf[idx] = rew
        self.next_obs_buf[idx] = next_obs
        self.done_buf[idx] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs_buf[idxs],
            self.acts_buf[idxs],
            self.rews_buf[idxs],
            self.next_obs_buf[idxs],
            self.done_buf[idxs],
        )

    def __len__(self) -> int:
        return self.size

