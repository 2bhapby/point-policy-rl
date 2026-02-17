#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
try:
    import imageio.v2 as imageio
except Exception:  # pragma: no cover
    imageio = None

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

# Allow direct execution as `python point_policy_rl/check_base_action_pipeline_rl.py`.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from point_policy_rl.base_policy_adapter_rl import FrozenPointPolicyBaseRL
from point_policy_rl.env_bridge_rl import build_single_env_from_bc_config, clip_action
from point_policy_rl.utils_rl import coerce_device, set_seed


def _is_basefix_v1_suite_name(suite_name: str | None) -> bool:
    name = str(suite_name or "").strip().lower()
    return name in {"libero_spatial_basefix_v1", "libero_object_basefix_v1"}


def _apply_basefix_v1_suite_preset(base_cfg: dict[str, Any], suite_name: str | None) -> list[str]:
    suite_cfg = base_cfg.get("suite")
    if not isinstance(suite_cfg, dict):
        return []

    canonical_suite_name = str(suite_name or "").strip()
    if canonical_suite_name:
        suite_cfg["suite"] = canonical_suite_name
        suite_cfg["name"] = canonical_suite_name

    preset: dict[str, Any] = {
        "pose_solve_mode": "rigid",
        "pose_delta_gain": 1.0,
        "normalize_delta_action_to_osc": True,
        "controller_output_max_pos": 0.05,
        "controller_output_max_rot": 0.5,
        "max_delta_pos": 0.05,
        "max_delta_rot": 0.25,
        "gripper_close_threshold": 0.2,
        "gripper_open_threshold": -0.2,
        "gripper_cmd_slew_rate": 2.0,
        "gripper_close_position_gate": False,
        "gripper_distance_control_mode": "threshold",
        "gripper_distance_value_source": "pred_points",
        "real_deploy_mode": True,
    }

    changed: list[str] = []
    for key, value in preset.items():
        if suite_cfg.get(key) != value:
            suite_cfg[key] = value
            changed.append(key)
    return changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-check Frozen Point-Policy loading and point2action pipeline."
    )
    parser.add_argument("--bc-weight", type=str, required=True)
    parser.add_argument("--suite", type=str, default="libero_object_basefix_v1")
    parser.add_argument("--task-name", type=str, default="")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--env-max-episode-len", type=int, default=3000)
    parser.add_argument("--num-steps", type=int, default=5)
    parser.add_argument("--eval-mode", action="store_true")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-dir", type=str, default="")
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-render-size", type=int, default=256)
    parser.add_argument("--video-tag", type=str, default="")
    parser.add_argument(
        "--debug-point-steps",
        type=int,
        default=0,
        help="If >0, emit point_tracks stats for the first N steps.",
    )
    return parser.parse_args()


def _to_list(arr: Any) -> list[float]:
    return np.asarray(arr, dtype=np.float32).reshape(-1).tolist()


def _slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")


def _point_debug_stats(
    obs: dict[str, Any],
    pixel_key: str,
    num_robot_points: int,
    num_object_points: int,
) -> dict[str, Any]:
    point_key = f"point_tracks_{pixel_key}"
    pts = np.asarray(obs[point_key], dtype=np.float32).reshape(-1, 3)
    stats: dict[str, Any] = {
        "point_key": point_key,
        "num_points": int(pts.shape[0]),
        "nan_count": int(np.isnan(pts).sum()),
        "abs_max": float(np.nanmax(np.abs(pts))) if pts.size else 0.0,
        "min_xyz": np.nanmin(pts, axis=0).astype(np.float32).tolist() if pts.size else [0.0, 0.0, 0.0],
        "max_xyz": np.nanmax(pts, axis=0).astype(np.float32).tolist() if pts.size else [0.0, 0.0, 0.0],
        "mean_xyz": np.nanmean(pts, axis=0).astype(np.float32).tolist() if pts.size else [0.0, 0.0, 0.0],
        "std_xyz": np.nanstd(pts, axis=0).astype(np.float32).tolist() if pts.size else [0.0, 0.0, 0.0],
    }
    if pts.size:
        robot_n = max(0, min(int(num_robot_points), int(pts.shape[0])))
        obj_n = max(0, min(int(num_object_points), int(pts.shape[0] - robot_n)))
        if robot_n > 0:
            r = pts[:robot_n]
            stats["robot_mean_xyz"] = np.nanmean(r, axis=0).astype(np.float32).tolist()
            stats["robot_std_xyz"] = np.nanstd(r, axis=0).astype(np.float32).tolist()
        if obj_n > 0:
            o = pts[robot_n : robot_n + obj_n]
            stats["object_mean_xyz"] = np.nanmean(o, axis=0).astype(np.float32).tolist()
            stats["object_std_xyz"] = np.nanstd(o, axis=0).astype(np.float32).tolist()
    return stats


def _normalize_frame(frame: np.ndarray, render_size: int) -> np.ndarray:
    out = np.asarray(frame)
    if out.ndim == 2:
        out = np.repeat(out[..., None], 3, axis=2)
    if out.ndim != 3:
        raise ValueError(f"Unsupported frame ndim={out.ndim}")
    if out.shape[2] == 1:
        out = np.repeat(out, 3, axis=2)
    elif out.shape[2] > 3:
        out = out[:, :, :3]
    if out.dtype != np.uint8:
        out = np.clip(out, 0, 255).astype(np.uint8)

    if (
        int(render_size) > 0
        and (out.shape[0] != int(render_size) or out.shape[1] != int(render_size))
        and cv2 is not None
    ):
        out = cv2.resize(
            out,
            dsize=(int(render_size), int(render_size)),
            interpolation=cv2.INTER_CUBIC,
        )
    return out


def _capture_frame(env, render_size: int) -> np.ndarray:
    if hasattr(env, "physics"):
        frame = env.physics.render(height=int(render_size), width=int(render_size), camera_id=0)
    else:
        try:
            frame = env.render(mode="rgb_array", width=int(render_size), height=int(render_size))
        except TypeError:
            frame = env.render()
    return _normalize_frame(frame, render_size=render_size)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    bc_weight = Path(args.bc_weight).expanduser().resolve()
    device = coerce_device(args.device)
    set_seed(args.seed)

    base = FrozenPointPolicyBaseRL(
        repo_root=repo_root,
        bc_weight=bc_weight,
        device=device,
    )

    if _is_basefix_v1_suite_name(args.suite) and isinstance(base.cfg, dict):
        changed = _apply_basefix_v1_suite_preset(base.cfg, args.suite)
        if changed:
            print(
                json.dumps(
                    {
                        "kind": "suite_override",
                        "suite": str(args.suite),
                        "changed_keys": sorted(changed),
                    }
                )
            )

    env, task_desc, pixel_key, low, high, suite_name = build_single_env_from_bc_config(
        cfg=base.cfg,
        repo_root=repo_root,
        suite_override=args.suite,
        task_override=(args.task_name if str(args.task_name).strip() else None),
        seed=int(args.seed),
        eval_mode=bool(args.eval_mode),
        max_episode_len=int(args.env_max_episode_len),
    )

    base.build(
        obs_spec=env.observation_spec(),
        action_spec=env.action_spec(),
        max_episode_len=int(args.env_max_episode_len),
    )
    base.reset_episode()

    header = {
        "kind": "header",
        "bc_weight": str(bc_weight),
        "suite_requested": str(args.suite),
        "suite_effective": str(suite_name),
        "task_name": str(args.task_name),
        "task_desc": str(task_desc),
        "pixel_key": str(pixel_key),
        "low": _to_list(low),
        "high": _to_list(high),
        "device": str(device),
        "num_steps": int(args.num_steps),
        "save_video": bool(args.save_video),
        "debug_point_steps": int(args.debug_point_steps),
    }
    print(json.dumps(header))

    ts = env.reset()
    obs = ts.observation
    suite_cfg = base.cfg.get("suite", {}) if isinstance(base.cfg, dict) else {}
    num_robot_points = int(suite_cfg.get("num_robot_points", 9)) if isinstance(suite_cfg, dict) else 9
    num_object_points = int(suite_cfg.get("num_object_points", 0)) if isinstance(suite_cfg, dict) else 0
    executed_steps = 0
    frames: list[np.ndarray] = []
    video_path: Path | None = None
    video_saved = False

    if bool(args.save_video):
        if imageio is None:
            print(json.dumps({"kind": "warn", "msg": "save_video requested but imageio is unavailable"}))
        else:
            video_dir = (
                Path(args.video_dir).expanduser().resolve()
                if str(args.video_dir).strip()
                else (repo_root / "point_policy" / "exp_local_rl" / "base_only_eval_videos").resolve()
            )
            video_dir.mkdir(parents=True, exist_ok=True)
            tag = _slugify(args.video_tag) or "base_only"
            task_tag = _slugify(args.task_name or "task")
            video_path = video_dir / f"{tag}__{task_tag}__seed{int(args.seed)}.mp4"
            try:
                frames.append(_capture_frame(env=env, render_size=int(args.video_render_size)))
            except Exception as exc:
                print(json.dumps({"kind": "warn", "msg": f"video capture init failed: {exc}"}))
                video_path = None

    try:
        for step in range(max(int(args.num_steps), 1)):
            action_dict = base.act(obs, step=step, global_step=step)
            action_raw = np.asarray(env.point2action(action_dict), dtype=np.float32).reshape(7)
            action_exec = clip_action(action_raw, low=low, high=high)
            clip_delta = float(np.max(np.abs(action_exec - action_raw)))

            forced_terminal_due_env_error = False
            try:
                ts_next = env.step(action_exec)
            except ValueError as exc:
                # robosuite can raise this when env has already terminated internally.
                if "terminated episode" in str(exc).lower():
                    forced_terminal_due_env_error = True
                    ts_next = None
                else:
                    raise

            if forced_terminal_due_env_error:
                row = {
                    "kind": "step",
                    "step": int(step),
                    "action_dict_keys": sorted(list(action_dict.keys())),
                    "action_raw": _to_list(action_raw),
                    "action_exec": _to_list(action_exec),
                    "action_finite": bool(np.isfinite(action_raw).all()),
                    "exec_clip_delta_max": float(clip_delta),
                    "reward": 0.0,
                    "done": True,
                    "goal_achieved": False,
                    "forced_terminal_due_env_error": True,
                }
                if "gripper" in action_dict:
                    row["pred_gripper"] = _to_list(action_dict["gripper"])
                elif "future_gripper_states" in action_dict:
                    row["pred_gripper"] = _to_list(action_dict["future_gripper_states"])
                print(json.dumps(row))
                executed_steps += 1
                break

            next_obs = ts_next.observation
            done = bool(ts_next.last())
            reward = float(ts_next.reward)
            goal_achieved = bool(next_obs.get("goal_achieved", False))

            row = {
                "kind": "step",
                "step": int(step),
                "action_dict_keys": sorted(list(action_dict.keys())),
                "action_raw": _to_list(action_raw),
                "action_exec": _to_list(action_exec),
                "action_finite": bool(np.isfinite(action_raw).all()),
                "exec_clip_delta_max": float(clip_delta),
                "reward": float(reward),
                "done": bool(done),
                "goal_achieved": bool(goal_achieved),
            }
            if "gripper" in action_dict:
                row["pred_gripper"] = _to_list(action_dict["gripper"])
            elif "future_gripper_states" in action_dict:
                row["pred_gripper"] = _to_list(action_dict["future_gripper_states"])
            if int(args.debug_point_steps) > 0 and int(step) < int(args.debug_point_steps):
                row["point_stats"] = _point_debug_stats(
                    obs=obs,
                    pixel_key=pixel_key,
                    num_robot_points=num_robot_points,
                    num_object_points=num_object_points,
                )
            print(json.dumps(row))

            executed_steps += 1
            obs = next_obs
            if bool(args.save_video) and imageio is not None and video_path is not None:
                try:
                    frames.append(_capture_frame(env=env, render_size=int(args.video_render_size)))
                except Exception as exc:
                    print(json.dumps({"kind": "warn", "msg": f"video capture step failed: {exc}"}))
                    video_path = None
            if done:
                break
    finally:
        try:
            if hasattr(env, "close"):
                env.close()
        except Exception as exc:
            print(json.dumps({"kind": "warn", "msg": f"env.close failed: {exc}"}))

    if bool(args.save_video) and imageio is not None and video_path is not None and len(frames) > 0:
        try:
            imageio.mimsave(str(video_path), frames, fps=max(1, int(args.video_fps)))
            video_saved = True
            print(json.dumps({"kind": "video", "saved": True, "video_path": str(video_path)}))
        except Exception as exc:
            print(json.dumps({"kind": "warn", "msg": f"video save failed: {exc}"}))

    if hasattr(env, "get_point2action_stats"):
        try:
            p2a = env.get_point2action_stats()
            print(json.dumps({"kind": "point2action_stats", "stats": p2a}))
        except Exception as exc:
            print(json.dumps({"kind": "warn", "msg": f"get_point2action_stats failed: {exc}"}))

    print(
        json.dumps(
            {
                "kind": "summary",
                "executed_steps": int(executed_steps),
                "requested_steps": int(args.num_steps),
                "video_saved": bool(video_saved),
                "video_path": str(video_path) if (video_saved and video_path is not None) else "",
            }
        )
    )


if __name__ == "__main__":
    main()
