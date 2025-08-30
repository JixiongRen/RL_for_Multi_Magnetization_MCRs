from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .networks import Actor, Critic


@dataclass
class DDPGConfig:
    s_dim: int
    a_dim: int
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    actor_hidden: Tuple[int, int] = (256, 256)
    critic_hidden: Tuple[int, int] = (256, 256)
    device: str = "auto"  # "cuda" | "cpu" | "auto"
    max_grad_norm: Optional[float] = None


class OUNoise:
    def __init__(self, size: int, theta: float = 0.15, sigma: float = 0.2, dt: float = 1.0):
        self.size = size
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.x_prev = np.zeros(self.size, dtype=np.float32)

    def reset(self) -> None:
        self.x_prev[...] = 0.0

    def __call__(self) -> np.ndarray:
        x = self.x_prev + self.theta * (-self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn(self.size)
        self.x_prev = x.astype(np.float32)
        return self.x_prev


class DDPG:
    def __init__(self, cfg: DDPGConfig) -> None:
        device = torch.device("cuda" if (cfg.device == "auto" and torch.cuda.is_available()) else (cfg.device if cfg.device != "auto" else "cpu"))
        print(f"use device: {device}")
        self.device = device

        self.actor = Actor(cfg.s_dim, cfg.a_dim, cfg.actor_hidden).to(self.device)
        self.actor_target = Actor(cfg.s_dim, cfg.a_dim, cfg.actor_hidden).to(self.device)
        self.critic = Critic(cfg.s_dim, cfg.a_dim, cfg.critic_hidden).to(self.device)
        self.critic_target = Critic(cfg.s_dim, cfg.a_dim, cfg.critic_hidden).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.lr_actor)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.lr_critic)

        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.max_grad_norm = cfg.max_grad_norm

        self.ou_noise = OUNoise(cfg.a_dim)

    @torch.no_grad()
    def act(self, obs: np.ndarray, noise_scale: float = 0.0) -> np.ndarray:
        self.actor.eval()
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a = self.actor(o).squeeze(0).cpu().numpy()
        if noise_scale > 0:
            a = np.clip(a + noise_scale * self.ou_noise(), -1.0, 1.0).astype(np.float32)
        return a.astype(np.float32)

    def soft_update(self, net: nn.Module, target_net: nn.Module, tau: float) -> None:
        with torch.no_grad():
            for p, p_targ in zip(net.parameters(), target_net.parameters()):
                p_targ.data.mul_(1 - tau)
                p_targ.data.add_(tau * p.data)

    def train_step(self, batch: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> Tuple[float, float]:
        obs, acts, rews, next_obs, dones = batch
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        acts_t = torch.as_tensor(acts, dtype=torch.float32, device=self.device)
        rews_t = torch.as_tensor(rews, dtype=torch.float32, device=self.device)
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

        # Critic update
        with torch.no_grad():
            next_acts = self.actor_target(next_obs_t)
            target_q = self.critic_target(next_obs_t, next_acts)
            backup = rews_t + (1.0 - dones_t) * self.gamma * target_q

        q_vals = self.critic(obs_t, acts_t)
        critic_loss = torch.nn.functional.mse_loss(q_vals, backup)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_opt.step()

        # Actor update (maximize Q(s, mu(s)))
        pred_acts = self.actor(obs_t)
        actor_loss = -self.critic(obs_t, pred_acts).mean()
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
        self.actor_opt.step()

        # Target nets update
        self.soft_update(self.actor, self.actor_target, self.tau)
        self.soft_update(self.critic, self.critic_target, self.tau)

        return float(actor_loss.item()), float(critic_loss.item())

    def save(self, path: str) -> None:
        ckpt = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
        }
        torch.save(ckpt, path)

    def load(self, path: str, map_location: Optional[str] = None) -> None:
        ckpt = torch.load(path, map_location=map_location)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        self.critic_opt.load_state_dict(ckpt["critic_opt"])

