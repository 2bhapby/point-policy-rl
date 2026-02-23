#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import imageio.v2 as imageio

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

# Allow `python point_policy_rl/eval_residual_td3_rl_rel.py` execution.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from point_policy_rl.base_policy_adapter_rl import FrozenPointPolicyBaseRL
from point_policy_rl.env_bridge_rl_rel import (
    build_single_env_from_bc_config,
    clip_action,
    observation_to_state,
)
from point_policy_rl.td3_agent_rl import ResidualTD3AgentRL, TD3ConfigRL
from point_policy_rl.train_resfit_residual_td3_rl_rel import (
    LinearActionNormalizerRL,
    _apply_basefix_v1_suite_preset,
    _build_agent_obs_batched,
    _build_resfit_qagent,
    _ensure_resfit_common_utils_compat,
    _extract_image_from_obs,
    _ensure_tabulate_compat,
    _ensure_torch_attention_compat,
    _is_basefix_v1_suite_name,
    _state_to_agent_obs_batched,
)
from point_policy_rl.utils_rl import coerce_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate residual TD3 checkpoint in _rl namespace")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--resfit-root",
        type=str,
        default="/sjw_alinlab2/home/sanghyeok/residual-offpolicy-rl",
    )
    parser.add_argument(
        "--suite",
        type=str,
        default=None,
        choices=[
            "libero_spatial",
            "libero_object",
            "libero_spatial_basefix_v1",
            "libero_object_basefix_v1",
        ],
    )
    parser.add_argument("--task-name", type=str, default=None)
    parser.add_argument(
        "--benchmark-name",
        type=str,
        default="",
        help="Optional benchmark override (e.g., LIBERO_10).",
    )
    parser.add_argument(
        "--task-order-index",
        type=int,
        default=-1,
        help="Optional task_order_index override. -1 means keep checkpoint config value.",
    )
    parser.add_argument("--env-max-episode-len", type=int, default=3000)
    parser.add_argument(
        "--state-coord-mode",
        type=str,
        default="",
        choices=["", "absolute", "relative_eef", "both_eef"],
        help="Optional override for REL branch state coordinate mode.",
    )
    parser.add_argument(
        "--image-size-override",
        type=int,
        default=0,
        help="If >0, force image-only eval resize to this square size (e.g., 84).",
    )
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-dir", type=str, default=None)
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-render-size", type=int, default=256)
    parser.add_argument("--video-tag", type=str, default="")
    return parser.parse_args()


def _resolve_bc_weight(payload: dict) -> str:
    bc_weight = payload.get("bc_weight")
    if bc_weight is not None and str(bc_weight).strip() != "":
        return str(bc_weight)
    args = payload.get("args", {})
    if isinstance(args, dict):
        bc_weight = args.get("bc_weight")
        if bc_weight is not None and str(bc_weight).strip() != "":
            return str(bc_weight)
    raise KeyError("Missing bc_weight in checkpoint payload (expected top-level or args.bc_weight)")


def _build_resfit_args_for_eval(payload: dict, cli_resfit_root: str) -> argparse.Namespace:
    args = payload.get("args", {})
    if not isinstance(args, dict):
        args = {}

    def _get(name: str, default):
        value = args.get(name, default)
        if value is None:
            return default
        return value

    return argparse.Namespace(
        resfit_root=str(_get("resfit_root", cli_resfit_root)),
        actor_lr=float(_get("actor_lr", 1e-4)),
        critic_lr=float(_get("critic_lr", 1e-4)),
        critic_target_tau=float(_get("critic_target_tau", 0.005)),
        freeze_encoder=bool(_get("freeze_encoder", False)),
        critic_hidden_dim=int(_get("critic_hidden_dim", 1024)),
        num_q_heads=int(_get("num_q_heads", 10)),
        policy_gradient_type=str(_get("policy_gradient_type", "ensemble_mean")),
        actor_hidden_dim=int(_get("actor_hidden_dim", 1024)),
        residual_action_scale=float(_get("residual_action_scale", 1.0)),
        dummy_image_size=int(_get("dummy_image_size", 84)),
        camera_key=str(_get("camera_key", "observation.images.agentview")),
        residual_obs_mode=str(_get("residual_obs_mode", "point_state_dummy_image")),
        image_obs_key=str(_get("image_obs_key_requested", _get("image_obs_key", ""))),
        image_obs_key_used=str(_get("image_obs_key_used", "")),
        prop_dim=int(_get("prop_dim", 1)),
        obs_shape=tuple(int(x) for x in _get("obs_shape", (3, int(_get("dummy_image_size", 84)), int(_get("dummy_image_size", 84))))),
    )


def _slugify(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")


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


def _resolve_video_dir(ckpt_path: Path, cli_video_dir: str | None) -> Path:
    if cli_video_dir is not None and str(cli_video_dir).strip() != "":
        out_dir = Path(cli_video_dir).expanduser().resolve()
    else:
        run_dir = ckpt_path.parent.parent if ckpt_path.parent.name == "snapshot" else ckpt_path.parent
        out_dir = (run_dir / "eval_videos_rl").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _build_video_filename(
    ep_idx: int,
    suite_name: str | None,
    task_name: str | None,
    success: int,
    video_tag: str,
) -> str:
    parts: list[str] = []
    tag = _slugify(video_tag)
    if tag:
        parts.append(tag)
    parts.append(f"ep{int(ep_idx):03d}")
    parts.append(_slugify(suite_name or "suite"))
    parts.append(_slugify(task_name or "task")[:80] or "task")
    parts.append("success" if int(success) > 0 else "fail")
    return "__".join(parts) + ".mp4"


def _dedup_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    idx = 2
    while True:
        candidate = path.with_name(f"{stem}_v{idx}{suffix}")
        if not candidate.exists():
            return candidate
        idx += 1


def main() -> None:
    args = parse_args()
    _ensure_torch_attention_compat()
    _ensure_tabulate_compat()
    device = coerce_device(args.device)
    set_seed(args.seed)

    import torch

    ckpt_path = Path(args.ckpt).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    payload = torch.load(ckpt_path, map_location=device)
    ckpt_format = "resfit_agent" if "agent" in payload else "td3_legacy"
    if ckpt_format == "td3_legacy" and "td3" not in payload:
        raise KeyError("Unsupported checkpoint format: expected 'td3' or 'agent' key")

    # ResFiT checkpoints need the external resfit package root on sys.path.
    if ckpt_format == "resfit_agent":
        resfit_root = Path(args.resfit_root).expanduser().resolve()
        if not (resfit_root / "resfit").exists():
            raise FileNotFoundError(
                f"Invalid resfit root (missing 'resfit/' package): {resfit_root}"
            )
        if str(resfit_root) not in sys.path:
            sys.path.insert(0, str(resfit_root))

    bc_weight = _resolve_bc_weight(payload)

    repo_root = Path(__file__).resolve().parents[1]

    base = FrozenPointPolicyBaseRL(
        repo_root=repo_root,
        bc_weight=Path(bc_weight),
        device=device,
    )

    payload_args = payload.get("args", {})
    if not isinstance(payload_args, dict):
        payload_args = {}

    suite_override = args.suite or payload.get("suite") or payload_args.get("suite")
    task_override = args.task_name or payload.get("task_name") or payload_args.get("task_name")
    benchmark_override = str(
        args.benchmark_name
        or payload.get("benchmark_name")
        or payload_args.get("benchmark_name")
        or ""
    ).strip()
    task_order_saved = payload.get("task_order_index", payload_args.get("task_order_index", -1))
    try:
        task_order_saved = int(task_order_saved)
    except Exception:
        task_order_saved = -1
    task_order_override = int(args.task_order_index) if int(args.task_order_index) >= 0 else int(task_order_saved)
    if _is_basefix_v1_suite_name(suite_override) and isinstance(base.cfg, dict):
        changed = _apply_basefix_v1_suite_preset(base.cfg, suite_override)
        changed_keys = ",".join(sorted(changed)) if changed else "none"
        print(
            "[suite-override] applied basefix_v1 preset "
            f"suite={suite_override} changed={changed_keys}"
        )
    if isinstance(base.cfg, dict):
        suite_cfg = base.cfg.get("suite")
        if isinstance(suite_cfg, dict):
            task_cfg = suite_cfg.get("task")
            if not isinstance(task_cfg, dict):
                task_cfg = {}
                suite_cfg["task"] = task_cfg
            if benchmark_override:
                task_cfg["benchmark_name"] = benchmark_override
            if task_order_override >= 0:
                task_cfg["task_order_index"] = int(task_order_override)
    if benchmark_override:
        print(f"[suite-override] benchmark_name={benchmark_override}")
    if task_order_override >= 0:
        print(f"[suite-override] task_order_index={task_order_override}")

    env, _, pixel_key, low_default, high_default, _ = build_single_env_from_bc_config(
        cfg=base.cfg,
        repo_root=repo_root,
        suite_override=suite_override,
        task_override=task_override,
        seed=args.seed,
        eval_mode=True,
        max_episode_len=args.env_max_episode_len,
    )

    base.build(
        obs_spec=env.observation_spec(),
        action_spec=env.action_spec(),
        max_episode_len=args.env_max_episode_len,
    )

    # Bootstrap state dim from env+base observation.
    t0 = env.reset()
    obs0 = t0.observation
    base.reset_episode()
    base_dict0 = base.act(obs0, 0, 0)
    base_action0 = np.asarray(env.point2action(base_dict0), dtype=np.float32).reshape(7)

    include_eef_pos = bool(payload.get("include_eef_pos", payload_args.get("include_eef_pos", False)))
    state_coord_mode_saved = str(
        payload.get("state_coord_mode", payload_args.get("state_coord_mode", "relative_eef"))
    ).strip()
    state_coord_mode_cli = str(args.state_coord_mode).strip()
    state_coord_mode = state_coord_mode_cli if state_coord_mode_cli else state_coord_mode_saved
    if state_coord_mode not in {"absolute", "relative_eef", "both_eef"}:
        raise ValueError(
            f"Unsupported state_coord_mode={state_coord_mode}. "
            "Use one of: absolute, relative_eef, both_eef."
        )
    print(
        f"[state-rel] include_eef_pos={include_eef_pos} state_coord_mode={state_coord_mode}"
    )
    state0 = observation_to_state(
        obs=obs0,
        pixel_key=pixel_key,
        base_action_7d=base_action0,
        include_eef_pos=include_eef_pos,
        state_coord_mode=state_coord_mode,
    )

    td3 = None
    q_agent = None
    resfit_utils = None
    action_normalizer = None
    residual_obs_mode = "point_state_dummy_image"
    image_obs_key = ""
    image_hw: tuple[int, int] | None = None

    if ckpt_format == "td3_legacy":
        td3_payload = payload["td3"]
        td3_cfg_saved = td3_payload.get("cfg", {})

        residual_scale_src = payload.get("residual_scale")
        if residual_scale_src is None:
            residual_scale_src = td3_payload.get("action_scale")
        residual_scale = np.asarray(residual_scale_src, dtype=np.float32).reshape(7)

        td3_cfg = TD3ConfigRL(
            obs_dim=int(payload.get("obs_dim", state0.shape[0])),
            action_dim=int(payload.get("action_dim", 7)),
            device=device,
            actor_hidden_dim=int(td3_cfg_saved.get("actor_hidden_dim", 256)),
            critic_hidden_dim=int(td3_cfg_saved.get("critic_hidden_dim", 256)),
            actor_lr=float(td3_cfg_saved.get("actor_lr", 3e-4)),
            critic_lr=float(td3_cfg_saved.get("critic_lr", 3e-4)),
            gamma=float(td3_cfg_saved.get("gamma", 0.99)),
            tau=float(td3_cfg_saved.get("tau", 0.005)),
            policy_noise=float(td3_cfg_saved.get("policy_noise", 0.2)),
            noise_clip=float(td3_cfg_saved.get("noise_clip", 0.5)),
            actor_update_freq=int(td3_cfg_saved.get("actor_update_freq", 2)),
        )
        td3 = ResidualTD3AgentRL(cfg=td3_cfg, action_scale=residual_scale)
        td3.load_state_dict(td3_payload)
    else:
        q_args = _build_resfit_args_for_eval(payload=payload, cli_resfit_root=args.resfit_root)
        resfit_utils = _ensure_resfit_common_utils_compat(Path(q_args.resfit_root).expanduser().resolve())
        residual_obs_mode = str(getattr(q_args, "residual_obs_mode", "point_state_dummy_image")).strip().lower()
        if residual_obs_mode not in {"point_state_dummy_image", "image_only"}:
            raise ValueError(f"Unsupported residual_obs_mode in checkpoint: {residual_obs_mode}")
        image_obs_key = str(
            getattr(q_args, "image_obs_key_used", "") or getattr(q_args, "image_obs_key", "")
        ).strip()
        if residual_obs_mode == "image_only":
            fixed_size = int(args.image_size_override) if int(args.image_size_override) > 0 else int(
                getattr(q_args, "dummy_image_size", 84)
            )
            if fixed_size <= 0:
                fixed_size = 84
            fixed_hw = (fixed_size, fixed_size)
            sample_img, image_key_used = _extract_image_from_obs(
                obs=obs0,
                pixel_key=pixel_key,
                image_obs_key=image_obs_key,
                target_hw=fixed_hw,
            )
            obs_shape = tuple(int(x) for x in sample_img.shape)
            image_hw = fixed_hw
            prop_dim = int(max(getattr(q_args, "prop_dim", 1), 1))
            print(
                "[obs-mode] image_only "
                f"image_key={image_key_used} obs_shape={obs_shape} fixed_hw={image_hw} "
                f"prop_dim={prop_dim}"
            )
        else:
            prop_dim = int(state0.shape[0] - 7)
            obs_shape = tuple(int(x) for x in getattr(q_args, "obs_shape", (3, int(q_args.dummy_image_size), int(q_args.dummy_image_size))))
            print(
                f"[obs-mode] point_state_dummy_image obs_shape={obs_shape} prop_dim={prop_dim}"
            )
        q_agent = _build_resfit_qagent(
            args=q_args,
            device=device,
            action_dim=7,
            prop_dim=prop_dim,
            obs_shape=obs_shape,
        )
        q_agent.load_state_dict(payload["agent"])
        normalizer_payload = payload.get("action_normalizer", {})
        if not isinstance(normalizer_payload, dict):
            normalizer_payload = {}
        action_normalizer = LinearActionNormalizerRL(
            low=np.asarray(normalizer_payload.get("low", low_default.tolist()), dtype=np.float32).reshape(7),
            high=np.asarray(normalizer_payload.get("high", high_default.tolist()), dtype=np.float32).reshape(7),
        )

    if action_normalizer is not None:
        low = np.asarray(action_normalizer.low, dtype=np.float32).reshape(7)
        high = np.asarray(action_normalizer.high, dtype=np.float32).reshape(7)
    else:
        low = np.asarray(payload.get("low", low_default.tolist()), dtype=np.float32).reshape(7)
        high = np.asarray(payload.get("high", high_default.tolist()), dtype=np.float32).reshape(7)

    episode_returns = []
    successes = []
    saved_videos: list[str] = []
    video_dir: Path | None = None
    video_enabled = bool(args.save_video)
    if video_enabled:
        video_dir = _resolve_video_dir(ckpt_path=ckpt_path, cli_video_dir=args.video_dir)
        print(
            f"[video] enabled dir={video_dir} fps={int(args.video_fps)} "
            f"render_size={int(args.video_render_size)}"
        )

    for ep_idx in range(args.episodes):
        time_step = env.reset()
        obs = time_step.observation
        base.reset_episode()
        ep_ret = 0.0
        done = False
        step_in_ep = 0
        success = 0
        frames: list[np.ndarray] = []

        if video_enabled:
            try:
                frames.append(_capture_frame(env=env, render_size=int(args.video_render_size)))
            except Exception as exc:
                print(f"[video] capture disabled after init failure: {exc}")
                video_enabled = False

        while not done:
            base_action_dict = base.act(obs, step_in_ep, step_in_ep)
            base_action = np.asarray(env.point2action(base_action_dict), dtype=np.float32).reshape(7)
            if td3 is not None:
                state = observation_to_state(
                    obs=obs,
                    pixel_key=pixel_key,
                    base_action_7d=base_action,
                    include_eef_pos=include_eef_pos,
                    state_coord_mode=state_coord_mode,
                )
                residual_action = td3.select_action(state, exploration_std=0.0, deterministic=True)
                env_action = clip_action(base_action + residual_action, low=low, high=high)
            else:
                assert q_agent is not None
                assert resfit_utils is not None
                assert action_normalizer is not None
                if residual_obs_mode == "image_only":
                    agent_obs = _build_agent_obs_batched(
                        residual_obs_mode="image_only",
                        state_vec=None,
                        obs_raw=obs,
                        base_action_raw=base_action,
                        action_normalizer=action_normalizer,
                        camera_key=q_args.camera_key,
                        dummy_image_size=int(q_args.dummy_image_size),
                        pixel_key=pixel_key,
                        image_obs_key=image_obs_key,
                        image_hw=image_hw,
                        image_only_prop_dim=int(max(getattr(q_args, "prop_dim", 1), 1)),
                        device=device,
                    )
                else:
                    state = observation_to_state(
                        obs=obs,
                        pixel_key=pixel_key,
                        base_action_7d=base_action,
                        include_eef_pos=include_eef_pos,
                        state_coord_mode=state_coord_mode,
                    )
                    agent_obs = _state_to_agent_obs_batched(
                        state_vec=state,
                        action_normalizer=action_normalizer,
                        camera_key=q_args.camera_key,
                        dummy_image_size=int(q_args.dummy_image_size),
                        device=device,
                    )
                with resfit_utils.eval_mode(q_agent):
                    residual_norm = (
                        q_agent.act(agent_obs, eval_mode=True, stddev=0.0, cpu=True)
                        .squeeze(0)
                        .numpy()
                        .astype(np.float32)
                    )
                base_norm = action_normalizer.normalize(base_action)
                combined_norm = np.clip(base_norm + residual_norm, -1.0, 1.0).astype(np.float32)
                env_action = clip_action(action_normalizer.denormalize(combined_norm), low=low, high=high)

            time_step = env.step(env_action)
            obs = time_step.observation
            done = bool(time_step.last())
            goal_achieved = bool(obs.get("goal_achieved", False))
            if td3 is not None:
                ep_ret += float(time_step.reward)
            else:
                ep_ret += 1.0 if (done and goal_achieved) else 0.0
            if goal_achieved:
                success = 1
            step_in_ep += 1
            if video_enabled:
                try:
                    frames.append(_capture_frame(env=env, render_size=int(args.video_render_size)))
                except Exception as exc:
                    print(f"[video] capture disabled while stepping: {exc}")
                    video_enabled = False

        episode_returns.append(ep_ret)
        successes.append(success)
        if video_enabled and video_dir is not None and len(frames) > 0:
            video_name = _build_video_filename(
                ep_idx=ep_idx,
                suite_name=suite_override,
                task_name=task_override,
                success=success,
                video_tag=args.video_tag,
            )
            video_path = _dedup_path(video_dir / video_name)
            imageio.mimsave(str(video_path), frames, fps=max(1, int(args.video_fps)))
            saved_videos.append(str(video_path))
            print(f"[video] saved: {video_path}")

    result = {
        "episodes": args.episodes,
        "mean_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "mean_success": float(np.mean(successes)) if successes else 0.0,
        "checkpoint": str(ckpt_path),
        "checkpoint_format": ckpt_format,
        "suite": suite_override,
        "task": task_override,
    }
    if video_dir is not None:
        result["video_dir"] = str(video_dir)
        result["saved_videos"] = saved_videos
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
