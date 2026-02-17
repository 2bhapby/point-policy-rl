from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from point_policy_rl.utils_rl import add_point_policy_path, load_hydra_config_yaml


class FrozenPointPolicyBaseRL:
    """Load a trained Point-Policy BC checkpoint and expose frozen `act` API."""

    def __init__(
        self,
        repo_root: Path,
        bc_weight: Path,
        device: str,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.bc_weight = Path(bc_weight).resolve()
        self.device = device

        if not self.bc_weight.exists():
            raise FileNotFoundError(f"bc_weight not found: {self.bc_weight}")

        self.run_dir = self.bc_weight.parent.parent
        hydra_cfg = self.run_dir / ".hydra" / "config.yaml"
        if not hydra_cfg.exists():
            raise FileNotFoundError(f"BC hydra config not found: {hydra_cfg}")

        self.cfg = load_hydra_config_yaml(hydra_cfg)
        self.stats: dict[str, Any] | None = None
        self.agent = None
        self._pp_utils = None

    def _build_agent_kwargs(self, max_episode_len: int) -> dict[str, Any]:
        cfg = self.cfg
        agent_cfg = cfg.get("agent", {})
        suite_cfg = cfg.get("suite", {})

        kwargs = {
            "obs_shape": None,  # set by caller
            "action_shape": None,  # set by caller
            "device": self.device,
            "lr": float(agent_cfg.get("lr", 1e-4)),
            "hidden_dim": int(suite_cfg.get("hidden_dim", 256)),
            "stddev_schedule": float(agent_cfg.get("stddev_schedule", 0.1)),
            "use_tb": bool(cfg.get("use_tb", False)),
            "policy_head": str(cfg.get("policy_head", "deterministic")),
            "pixel_keys": list(suite_cfg.get("pixel_keys", ["pixels1", "pixels2"])),
            "history": bool(suite_cfg.get("history", True)),
            "history_len": int(suite_cfg.get("history_len", 10)),
            "eval_history_len": int(suite_cfg.get("eval_history_len", 10)),
            "temporal_agg": bool(cfg.get("temporal_agg", True)),
            "max_episode_len": int(max_episode_len),
            "num_queries": int(cfg.get("num_queries", 20)),
            "use_robot_points": bool(suite_cfg.get("use_robot_points", True)),
            "num_robot_points": int(suite_cfg.get("num_robot_points", 9)),
            "use_object_points": bool(suite_cfg.get("use_object_points", True)),
            "num_object_points": int(suite_cfg.get("num_object_points", 10)),
            "point_dim": int(suite_cfg.get("point_dim", 3)),
            "pred_gripper": bool(agent_cfg.get("pred_gripper", True)),
            "gripper_loss_weight": float(agent_cfg.get("gripper_loss_weight", 1.0)),
            "gripper_agg_mode": str(agent_cfg.get("gripper_agg_mode", "avg")),
            "condition_on_gripper_state": bool(agent_cfg.get("condition_on_gripper_state", True)),
            "separate_gripper_head": bool(agent_cfg.get("separate_gripper_head", False)),
            "gripper_bce_weight": float(agent_cfg.get("gripper_bce_weight", 1.0)),
            "gripper_label_mode": str(agent_cfg.get("gripper_label_mode", "command")),
            "gripper_head_use_transformer_context": bool(
                agent_cfg.get("gripper_head_use_transformer_context", False)
            ),
        }
        return kwargs

    def build(self, obs_spec, action_spec, max_episode_len: int) -> None:
        add_point_policy_path(self.repo_root)

        import torch
        import utils as pp_utils
        agent_cfg = self.cfg.get("agent", {}) if isinstance(self.cfg, dict) else {}
        target = str(agent_cfg.get("_target_", "agent.point_policy.BCAgent")).strip()
        if "." not in target:
            raise ValueError(f"Invalid agent target: {target}")
        module_name, class_name = target.rsplit(".", 1)
        try:
            agent_mod = importlib.import_module(module_name)
            BCAgent = getattr(agent_mod, class_name)
        except Exception as exc:
            raise ImportError(f"Failed to import BC agent target '{target}': {exc}") from exc

        self._pp_utils = pp_utils

        kwargs = self._build_agent_kwargs(max_episode_len=max_episode_len)
        obs_shape = {}
        for key in kwargs["pixel_keys"]:
            obs_shape[key] = obs_spec[key].shape
        obs_shape[self.cfg["suite"]["feature_key"]] = obs_spec[self.cfg["suite"]["feature_key"]].shape

        kwargs["obs_shape"] = obs_shape
        kwargs["action_shape"] = action_spec.shape

        self.agent = BCAgent(**kwargs)

        payload = torch.load(self.bc_weight, map_location=self.device)
        if "stats" not in payload:
            raise KeyError(
                f"stats missing in BC checkpoint: {self.bc_weight}. "
                "Please pass a workspace snapshot (*.pt) saved by point_policy/train.py"
            )
        self.stats = payload["stats"]

        self.agent.load_snapshot(payload, eval=True)
        self.agent.buffer_reset()

    def reset_episode(self) -> None:
        if self.agent is None:
            raise RuntimeError("build() must be called before reset_episode()")
        self.agent.buffer_reset()

    def act(self, observation: dict[str, Any], step: int, global_step: int) -> dict[str, Any]:
        if self.agent is None or self.stats is None or self._pp_utils is None:
            raise RuntimeError("build() must be called before act()")

        import torch

        with torch.no_grad(), self._pp_utils.eval_mode(self.agent):
            action_dict = self.agent.act(
                observation,
                self.stats,
                step,
                global_step,
                eval_mode=True,
            )
        return action_dict
