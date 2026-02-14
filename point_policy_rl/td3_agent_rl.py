from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from point_policy_rl.utils_rl import soft_update


@dataclass
class TD3ConfigRL:
    obs_dim: int
    action_dim: int
    device: str = "cuda"
    actor_hidden_dim: int = 256
    critic_hidden_dim: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    actor_update_freq: int = 2


def _mlp(input_dim: int, hidden_dim: int, output_dim: int):
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, output_dim),
    )


class _Actor:
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, action_scale: np.ndarray):
        import torch
        import torch.nn as nn

        self.net = _mlp(obs_dim, hidden_dim, action_dim)
        self._tanh = nn.Tanh()
        self.action_scale = torch.as_tensor(action_scale, dtype=torch.float32)

    def to(self, device: str):
        self.net.to(device)
        self.action_scale = self.action_scale.to(device)
        return self

    def parameters(self):
        return self.net.parameters()

    def __call__(self, obs):
        import torch

        out = self._tanh(self.net(obs))
        return out * self.action_scale


class _Critic:
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int):
        self.q1 = _mlp(obs_dim + action_dim, hidden_dim, 1)
        self.q2 = _mlp(obs_dim + action_dim, hidden_dim, 1)

    def to(self, device: str):
        self.q1.to(device)
        self.q2.to(device)
        return self

    def parameters(self):
        for p in self.q1.parameters():
            yield p
        for p in self.q2.parameters():
            yield p

    def __call__(self, obs, action):
        import torch

        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def q1_only(self, obs, action):
        import torch

        x = torch.cat([obs, action], dim=-1)
        return self.q1(x)


class ResidualTD3AgentRL:
    def __init__(self, cfg: TD3ConfigRL, action_scale: np.ndarray):
        import torch

        self.cfg = cfg
        self.device = cfg.device

        self.actor = _Actor(cfg.obs_dim, cfg.action_dim, cfg.actor_hidden_dim, action_scale).to(cfg.device)
        self.actor_target = _Actor(cfg.obs_dim, cfg.action_dim, cfg.actor_hidden_dim, action_scale).to(cfg.device)
        self.actor_target.net.load_state_dict(self.actor.net.state_dict())

        self.critic = _Critic(cfg.obs_dim, cfg.action_dim, cfg.critic_hidden_dim).to(cfg.device)
        self.critic_target = _Critic(cfg.obs_dim, cfg.action_dim, cfg.critic_hidden_dim).to(cfg.device)
        self.critic_target.q1.load_state_dict(self.critic.q1.state_dict())
        self.critic_target.q2.load_state_dict(self.critic.q2.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)

        self.action_scale = torch.as_tensor(action_scale, dtype=torch.float32, device=cfg.device)
        self.total_updates = 0

    def select_action(
        self,
        obs_np: np.ndarray,
        exploration_std: float = 0.0,
        deterministic: bool = False,
    ) -> np.ndarray:
        import torch

        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(obs).squeeze(0)
            if not deterministic and exploration_std > 0.0:
                noise = torch.randn_like(action) * (exploration_std * self.action_scale)
                action = action + noise
            action = torch.clamp(action, -self.action_scale, self.action_scale)
        return action.detach().cpu().numpy().astype(np.float32)

    def update_from_batch(self, batch: dict) -> dict[str, float]:
        import torch
        import torch.nn.functional as F

        self.total_updates += 1
        obs = batch["obs"]
        action = batch["action"]
        reward = batch["reward"]
        next_obs = batch["next_obs"]
        done = batch["done"]

        with torch.no_grad():
            next_action = self.actor_target(next_obs)
            noise = torch.randn_like(next_action) * (self.cfg.policy_noise * self.action_scale)
            noise = torch.clamp(noise, -self.cfg.noise_clip * self.action_scale, self.cfg.noise_clip * self.action_scale)
            next_action = next_action + noise
            next_action = torch.clamp(next_action, -self.action_scale, self.action_scale)

            target_q1, target_q2 = self.critic_target(next_obs, next_action)
            target_q = torch.minimum(target_q1, target_q2)
            target = reward + (1.0 - done) * self.cfg.gamma * target_q

        q1, q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss_value = 0.0
        if self.total_updates % self.cfg.actor_update_freq == 0:
            pi = self.actor(obs)
            actor_loss = -self.critic.q1_only(obs, pi).mean()

            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self.actor_opt.step()

            soft_update(self.actor_target.net, self.actor.net, self.cfg.tau)
            soft_update(self.critic_target.q1, self.critic.q1, self.cfg.tau)
            soft_update(self.critic_target.q2, self.critic.q2, self.cfg.tau)
            actor_loss_value = float(actor_loss.item())

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": actor_loss_value,
            "q1_mean": float(q1.mean().item()),
            "q2_mean": float(q2.mean().item()),
            "target_q_mean": float(target.mean().item()),
        }

    def update(self, replay, batch_size: int) -> dict[str, float]:
        if len(replay) < batch_size:
            return {}
        batch = replay.sample(batch_size=batch_size, device=self.device)
        return self.update_from_batch(batch)

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.net.state_dict(),
            "actor_target": self.actor_target.net.state_dict(),
            "critic_q1": self.critic.q1.state_dict(),
            "critic_q2": self.critic.q2.state_dict(),
            "critic_target_q1": self.critic_target.q1.state_dict(),
            "critic_target_q2": self.critic_target.q2.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "total_updates": self.total_updates,
            "cfg": self.cfg.__dict__,
            "action_scale": self.action_scale.detach().cpu().numpy().tolist(),
        }

    def load_state_dict(self, payload: dict) -> None:
        self.actor.net.load_state_dict(payload["actor"])
        self.actor_target.net.load_state_dict(payload["actor_target"])
        self.critic.q1.load_state_dict(payload["critic_q1"])
        self.critic.q2.load_state_dict(payload["critic_q2"])
        self.critic_target.q1.load_state_dict(payload["critic_target_q1"])
        self.critic_target.q2.load_state_dict(payload["critic_target_q2"])
        self.actor_opt.load_state_dict(payload["actor_opt"])
        self.critic_opt.load_state_dict(payload["critic_opt"])
        self.total_updates = int(payload.get("total_updates", 0))
