#!/usr/bin/env python3

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import hydra
import numpy as np
import torch

import utils
from eval_point_track_basefix_v1 import Workspace


def _to_numpy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    if np.isscalar(value):
        return np.asarray(value)
    return value


def _filter_observation(
    obs: Dict[str, Any],
    pixel_keys: List[str],
    save_images: bool,
) -> Dict[str, Any]:
    filtered: Dict[str, Any] = {}
    image_keys = set(pixel_keys)
    image_keys.update([f"pixels{i + 1}" for i in range(8)])
    for key, value in obs.items():
        if (key in image_keys) and (not save_images):
            continue
        filtered[key] = _to_numpy(value)
    return filtered


def _extract_policy_debug(action_dict: Dict[str, Any]) -> Dict[str, float]:
    debug = {}
    pred_g = None
    if "gripper" in action_dict:
        pred_g = float(np.asarray(action_dict["gripper"]).reshape(-1)[0])
    elif "future_gripper_states" in action_dict:
        pred_g = float(np.asarray(action_dict["future_gripper_states"]).reshape(-1)[0])
    if pred_g is not None:
        debug["pred_gripper"] = pred_g
    return debug


def _resolve_output_path(root_dir: Path, output_path_cfg: str) -> Path:
    output_path = Path(output_path_cfg).expanduser()
    if not output_path.is_absolute():
        output_path = (root_dir / output_path).resolve()
    return output_path


def _build_dataset_payload(
    episodes: List[Dict[str, Any]],
    cfg,
    env_idx: int,
    max_rollouts: int,
    target_successes: int,
    max_steps: int,
) -> Dict[str, Any]:
    lengths = [len(ep.get("steps", [])) for ep in episodes]
    return {
        "format": "rlds_like_success_only_v1",
        "collector": {
            "script": "collect_success_rlds_basefix_v1.py",
            "suite_name": str(cfg.suite.name),
            "task_name": str(cfg.suite.task.task_name),
            "benchmark_name": str(cfg.suite.task.benchmark_name),
            "env_idx": int(env_idx),
            "max_rollouts": int(max_rollouts),
            "target_successes": int(target_successes),
            "max_steps_per_rollout": int(max_steps),
            "save_images": bool(getattr(cfg, "collect_save_images", False)),
            "bc_weight": str(cfg.bc_weight),
            "seed": int(cfg.seed),
        },
        "episodes": episodes,
        "stats": {
            "num_success_episodes": int(len(episodes)),
            "mean_episode_len": float(np.mean(lengths)) if lengths else 0.0,
            "max_episode_len": int(np.max(lengths)) if lengths else 0,
            "min_episode_len": int(np.min(lengths)) if lengths else 0,
        },
    }


@hydra.main(config_path="cfgs", config_name="config_eval_basefix_v1", version_base=None)
def main(cfg):
    workspace = Workspace(cfg)

    snapshots = {}
    bc_snapshot = Path(cfg.bc_weight)
    if not bc_snapshot.exists():
        raise FileNotFoundError(f"bc weight not found: {bc_snapshot}")
    print(f"[collect] loading bc weight: {bc_snapshot}")
    snapshots["bc"] = bc_snapshot
    workspace.load_snapshot(snapshots)

    workspace.agent.train(False)

    env_idx = int(getattr(cfg, "collect_env_idx", 0))
    if env_idx < 0 or env_idx >= len(workspace.env):
        raise IndexError(
            f"collect_env_idx={env_idx} out of range. num_envs={len(workspace.env)}"
        )
    env = workspace.env[env_idx]

    max_rollouts = int(getattr(cfg, "collect_max_rollouts", 100))
    target_successes = int(getattr(cfg, "collect_target_successes", 50))
    save_images = bool(getattr(cfg, "collect_save_images", False))
    max_steps = int(
        getattr(
            cfg,
            "collect_max_steps",
            getattr(cfg, "eval_rollout_steps", getattr(cfg, "eval_max_episode_len", 3000)),
        )
    )
    if max_steps <= 0:
        max_steps = int(getattr(cfg, "eval_max_episode_len", 3000))
    output_path_cfg = str(
        getattr(
            cfg,
            "collect_output_path",
            "point_policy/exp_local/offline_rlds/libero10_success_only.pkl",
        )
    )
    output_path = _resolve_output_path(Path(cfg.root_dir).expanduser().resolve(), output_path_cfg)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pixel_keys = list(getattr(cfg.suite, "pixel_keys", []))
    stats = workspace.expert_replay_loader.dataset.stats
    successes: List[Dict[str, Any]] = []

    for rollout_idx in range(max_rollouts):
        if len(successes) >= target_successes:
            break

        print(
            "[collect] rollout "
            f"{rollout_idx + 1}/{max_rollouts} "
            f"(successes={len(successes)}/{target_successes})"
        )

        time_step = env.reset()
        workspace.agent.buffer_reset()
        episode_steps: List[Dict[str, Any]] = []
        episode_success = False

        for step_idx in range(max_steps):
            obs_t = _filter_observation(
                time_step.observation,
                pixel_keys=pixel_keys,
                save_images=save_images,
            )

            with torch.no_grad(), utils.eval_mode(workspace.agent):
                action_dict = workspace.agent.act(
                    time_step.observation,
                    stats,
                    step_idx,
                    workspace.global_step,
                    eval_mode=True,
                )

            next_time_step = env.step(action_dict)

            env_action = getattr(env, "_last_robot_action", None)
            if env_action is None:
                env_action = np.zeros((7,), dtype=np.float32)
            env_action = np.asarray(env_action, dtype=np.float32).reshape(-1)

            success_flag = bool(next_time_step.observation.get("goal_achieved", False))
            done_flag = bool(next_time_step.last())
            cutoff_flag = (step_idx + 1) >= max_steps
            is_last = bool(success_flag or done_flag or cutoff_flag)
            is_terminal = bool(success_flag and is_last)
            is_truncated = bool(is_last and (not success_flag))
            reward = 1.0 if is_terminal else 0.0
            discount = 0.0 if is_last else 1.0

            step_record = {
                "is_first": bool(step_idx == 0),
                "is_last": is_last,
                "is_terminal": is_terminal,
                "is_truncated": is_truncated,
                "reward": float(reward),
                "discount": float(discount),
                "observation": obs_t,
                "action": env_action.astype(np.float32),
                "policy_debug": _extract_policy_debug(action_dict),
                "info": {
                    "rollout_idx": int(rollout_idx),
                    "step_idx": int(step_idx),
                    "goal_achieved": bool(success_flag),
                    "env_done": bool(done_flag),
                },
            }
            episode_steps.append(step_record)

            time_step = next_time_step
            if is_terminal:
                episode_success = True
            if is_last:
                break

        if episode_success:
            episode_payload = {
                "episode_id": int(len(successes)),
                "steps": episode_steps,
                "metadata": {
                    "rollout_idx": int(rollout_idx),
                    "task_name": str(cfg.suite.task.task_name),
                    "benchmark_name": str(cfg.suite.task.benchmark_name),
                    "env_idx": int(env_idx),
                    "num_steps": int(len(episode_steps)),
                    "success": True,
                },
            }
            successes.append(episode_payload)
            print(
                f"[collect] SUCCESS saved "
                f"(rollout={rollout_idx}, steps={len(episode_steps)}, total={len(successes)})"
            )
        else:
            print(
                f"[collect] fail "
                f"(rollout={rollout_idx}, steps={len(episode_steps)})"
            )

    payload = _build_dataset_payload(
        episodes=successes,
        cfg=cfg,
        env_idx=env_idx,
        max_rollouts=max_rollouts,
        target_successes=target_successes,
        max_steps=max_steps,
    )
    with output_path.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    summary_path = output_path.with_suffix(".summary.txt")
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(f"output={output_path}\n")
        f.write(f"num_success_episodes={len(successes)}\n")
        f.write(f"target_successes={target_successes}\n")
        f.write(f"max_rollouts={max_rollouts}\n")
        if len(successes) > 0:
            lengths = [len(ep["steps"]) for ep in successes]
            f.write(f"mean_len={float(np.mean(lengths)):.2f}\n")
            f.write(f"min_len={int(np.min(lengths))}\n")
            f.write(f"max_len={int(np.max(lengths))}\n")

    print(f"[collect] done. saved: {output_path}")
    print(f"[collect] summary: {summary_path}")


if __name__ == "__main__":
    main()
