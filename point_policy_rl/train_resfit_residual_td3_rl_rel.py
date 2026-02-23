#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import contextlib
from collections import deque
import importlib
import importlib.util
import json
import sys
import types
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

# Allow `python point_policy_rl/train_resfit_residual_td3_rl_rel.py` execution.
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
from point_policy_rl.offline_demo_builder_rl_rel import build_offline_transitions_from_expert_demos_rl
from point_policy_rl.utils_rl import build_run_dir, coerce_device, dump_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Residual TD3 using ResFiT QAgent + TorchRL replay (isolated _rl path)"
    )
    parser.add_argument("--bc-weight", type=str, required=True)
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
        help="Optional benchmark override (e.g., LIBERO_SPATIAL, LIBERO_10).",
    )
    parser.add_argument(
        "--task-order-index",
        type=int,
        default=-1,
        help="Optional task_order_index override. -1 means keep checkpoint config value.",
    )
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--resfit-root",
        type=str,
        default="/sjw_alinlab2/home/sanghyeok/residual-offpolicy-rl",
    )

    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--warmup-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=300000)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--sampling-strategy", type=str, default="uniform", choices=["uniform", "prioritized_replay"])
    parser.add_argument("--priority-alpha", type=float, default=0.6)
    parser.add_argument("--priority-beta", type=float, default=0.4)
    parser.add_argument("--prefetch-batches", type=int, default=4)

    parser.add_argument("--num-updates-per-iteration", type=int, default=4)
    parser.add_argument("--actor-updates-per-iteration", type=int, default=1)
    parser.add_argument("--update-every-n-steps", type=int, default=1)
    parser.add_argument("--stddev-max", type=float, default=0.1)
    parser.add_argument("--stddev-min", type=float, default=0.1)
    parser.add_argument("--stddev-step", type=int, default=300000)
    parser.add_argument("--random-action-noise-scale", type=float, default=1.0)

    parser.add_argument("--residual-action-scale", type=float, default=1.0)
    parser.add_argument(
        "--residual-gripper-mode",
        type=str,
        default="full",
        choices=["full", "zero", "scale"],
        help="How to apply residual on gripper dim (index 6): full/zero/scale.",
    )
    parser.add_argument(
        "--residual-gripper-scale",
        type=float,
        default=1.0,
        help="Scale factor for residual gripper when --residual-gripper-mode=scale.",
    )
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--critic-target-tau", type=float, default=0.005)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--actor-hidden-dim", type=int, default=1024)
    parser.add_argument("--critic-hidden-dim", type=int, default=1024)
    parser.add_argument("--num-q-heads", type=int, default=10)
    parser.add_argument("--policy-gradient-type", type=str, default="ensemble_mean")

    parser.add_argument("--offline-fraction", type=float, default=0.5)
    parser.add_argument(
        "--offline-fraction-start",
        type=float,
        default=None,
        help="Optional scheduled offline fraction start value. "
        "If set with --offline-fraction-end, overrides fixed fraction during training.",
    )
    parser.add_argument(
        "--offline-fraction-end",
        type=float,
        default=None,
        help="Optional scheduled offline fraction end value.",
    )
    parser.add_argument(
        "--offline-fraction-decay-steps",
        type=int,
        default=100000,
        help="Linear decay steps for offline fraction schedule.",
    )
    parser.add_argument(
        "--offline-source",
        type=str,
        default="expert_demo",
        choices=["expert_demo", "base_rollout"],
        help="Source used to initialize offline buffer.",
    )
    parser.add_argument("--offline-demo-root", type=str, default="")
    parser.add_argument("--offline-max-demos", type=int, default=0)
    parser.add_argument(
        "--offline-rollout-episodes",
        type=int,
        default=0,
        help="Target episodes for offline collection when --offline-source=base_rollout. "
        "If <=0, falls back to offline_max_demos, then 50.",
    )
    parser.add_argument(
        "--offline-rollout-max-attempt-mult",
        type=int,
        default=20,
        help="Max collection attempts = target_episodes * this value (base_rollout mode).",
    )
    parser.add_argument(
        "--offline-success-only",
        action="store_true",
        help="Keep only successful episodes in offline collection.",
    )
    parser.add_argument("--offline-step-reward", type=float, default=0.0)
    parser.add_argument("--offline-terminal-reward", type=float, default=1.0)
    parser.add_argument(
        "--offline-base-action-mode",
        type=str,
        default="demo_delta",
        choices=["demo_delta", "bc_track_delta"],
    )
    parser.add_argument(
        "--online-bucket-sampling",
        action="store_true",
        help="Enable separate online success/failure replay buckets and mixed sampling.",
    )
    parser.add_argument(
        "--online-success-batch-fraction",
        type=float,
        default=0.5,
        help="Target fraction of online batch sampled from success bucket when "
        "--online-bucket-sampling is enabled.",
    )
    parser.add_argument(
        "--online-fail-max-steps",
        type=int,
        default=0,
        help="If >0, cap transitions from each failed episode before inserting to failure bucket.",
    )
    parser.add_argument(
        "--online-fail-stride",
        type=int,
        default=1,
        help="Subsample stride for failed episodes in failure bucket (>=1).",
    )

    parser.add_argument("--include-eef-pos", action="store_true")
    parser.add_argument(
        "--state-coord-mode",
        type=str,
        default="relative_eef",
        choices=["absolute", "relative_eef", "both_eef"],
        help="State point coordinate mode for REL branch input.",
    )
    parser.add_argument("--env-max-episode-len", type=int, default=3000)
    parser.add_argument("--dummy-image-size", type=int, default=84)
    parser.add_argument("--camera-key", type=str, default="observation.images.agentview")
    parser.add_argument(
        "--residual-obs-mode",
        type=str,
        default="point_state_dummy_image",
        choices=["point_state_dummy_image", "image_only"],
        help="Residual policy observation mode.",
    )
    parser.add_argument(
        "--image-obs-key",
        type=str,
        default="",
        help="Observation key for real image input when --residual-obs-mode=image_only. "
        "If empty, uses pixel_key from suite (e.g., pixels1).",
    )
    parser.add_argument(
        "--suite-gripper-close-position-gate",
        type=str,
        default="",
        help="Optional override for suite.gripper_close_position_gate (e.g., true/false)",
    )
    parser.add_argument(
        "--suite-gripper-cmd-slew-rate",
        type=float,
        default=None,
        help="Optional override for suite.gripper_cmd_slew_rate",
    )

    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--eval-save-video", action="store_true")
    parser.add_argument("--eval-video-dir", type=str, default="")
    parser.add_argument("--eval-video-fps", type=int, default=20)
    parser.add_argument("--eval-video-render-size", type=int, default=256)
    parser.add_argument("--eval-video-max-episodes", type=int, default=1)
    parser.add_argument("--eval-video-online-every-episodes", type=int, default=10)
    parser.add_argument("--eval-video-tag", type=str, default="")
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--log-every", type=int, default=200)
    parser.add_argument(
        "--strict-resfit",
        action="store_true",
        help="Enable strict ResFiT recipe/runtime checks (fail fast on mismatch).",
    )
    parser.add_argument(
        "--strict-max-residual-action-scale",
        type=float,
        default=0.5,
        help="Upper bound for residual action scale when --strict-resfit is enabled.",
    )
    parser.add_argument(
        "--strict-max-random-action-noise-scale",
        type=float,
        default=0.5,
        help="Upper bound for random warmup residual scale when --strict-resfit is enabled.",
    )

    parser.add_argument("--wandb-enable", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="point-policy-residual-rl")
    parser.add_argument("--wandb-entity", type=str, default="")
    parser.add_argument("--wandb-group", type=str, default="")
    parser.add_argument("--wandb-name", type=str, default="")
    parser.add_argument("--wandb-tags", type=str, default="")
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
    )

    parser.add_argument("--output-root", type=str, default="point_policy/exp_local_rl")
    parser.add_argument("--run-name", type=str, default="")
    return parser.parse_args()


class LinearActionNormalizerRL:
    def __init__(self, low: np.ndarray, high: np.ndarray):
        low = np.asarray(low, dtype=np.float32).reshape(-1)
        high = np.asarray(high, dtype=np.float32).reshape(-1)
        if low.shape != high.shape:
            raise ValueError(f"low/high shape mismatch: {low.shape} vs {high.shape}")
        scale = 0.5 * (high - low)
        scale = np.where(np.abs(scale) < 1e-6, 1.0, scale).astype(np.float32)
        self.low = low
        self.high = high
        self.center = (0.5 * (high + low)).astype(np.float32)
        self.scale = scale

    def normalize(self, action_raw: np.ndarray) -> np.ndarray:
        action_raw = np.asarray(action_raw, dtype=np.float32).reshape(self.center.shape)
        action_norm = (action_raw - self.center) / self.scale
        return np.clip(action_norm, -1.0, 1.0).astype(np.float32)

    def denormalize(self, action_norm: np.ndarray) -> np.ndarray:
        action_norm = np.asarray(action_norm, dtype=np.float32).reshape(self.center.shape)
        action_raw = action_norm * self.scale + self.center
        return np.clip(action_raw, self.low, self.high).astype(np.float32)


def _assert_strict_resfit_args(args: argparse.Namespace) -> None:
    if not bool(args.strict_resfit):
        return

    def _close(a: float, b: float, tol: float = 1e-12) -> bool:
        return abs(float(a) - float(b)) <= tol

    errors: list[str] = []
    if not _close(float(args.offline_fraction), 0.5):
        errors.append(
            f"offline_fraction must be 0.5 (got {args.offline_fraction})"
        )
    if not _close(float(args.critic_target_tau), 0.005):
        errors.append(
            f"critic_target_tau must be 0.005 (got {args.critic_target_tau})"
        )
    if (
        args.offline_fraction_start is not None
        or args.offline_fraction_end is not None
    ):
        errors.append(
            "offline fraction schedule is incompatible with --strict-resfit "
            "(keep fixed offline_fraction=0.5)."
        )
    if float(args.residual_action_scale) > float(args.strict_max_residual_action_scale):
        errors.append(
            "residual_action_scale exceeds strict bound "
            f"({args.residual_action_scale} > {args.strict_max_residual_action_scale})"
        )
    if float(args.random_action_noise_scale) > float(args.strict_max_random_action_noise_scale):
        errors.append(
            "random_action_noise_scale exceeds strict bound "
            f"({args.random_action_noise_scale} > {args.strict_max_random_action_noise_scale})"
        )

    if errors:
        msg = "\n".join([f"[strict-resfit] {e}" for e in errors])
        raise ValueError(msg)

    print(
        "[strict-resfit] enabled: "
        "offline_fraction=0.5 critic_target_tau=0.005 "
        f"residual_action_scale<={args.strict_max_residual_action_scale} "
        f"random_action_noise_scale<={args.strict_max_random_action_noise_scale}"
    )


def _action_diag(base_norm: np.ndarray, residual_norm: np.ndarray) -> dict[str, float]:
    base_norm = np.asarray(base_norm, dtype=np.float32).reshape(-1)
    residual_norm = np.asarray(residual_norm, dtype=np.float32).reshape(-1)
    pre = base_norm + residual_norm
    post = np.clip(pre, -1.0, 1.0)
    clip_mask = (pre < -1.0) | (pre > 1.0)
    base_in = (base_norm >= -1.0) & (base_norm <= 1.0)
    residual_induced = clip_mask & base_in

    base_l2 = float(np.linalg.norm(base_norm))
    res_l2 = float(np.linalg.norm(residual_norm))
    return {
        "clip_rate_dim": float(clip_mask.mean()),
        "clip_any": float(clip_mask.any()),
        "residual_induced_clip_rate_dim": float(residual_induced.mean()),
        "base_norm_l2": base_l2,
        "residual_norm_l2": res_l2,
        "res_over_base": float(res_l2 / (base_l2 + 1e-6)),
        "clip_delta_l1_mean": float(np.abs(pre - post).mean()),
        "clip_delta_linf": float(np.max(np.abs(pre - post))),
    }


def _apply_residual_gripper_mode(
    residual_norm: np.ndarray,
    *,
    mode: str,
    scale: float,
) -> np.ndarray:
    out = np.asarray(residual_norm, dtype=np.float32).reshape(-1).copy()
    if out.size < 7:
        return out.astype(np.float32)

    mode_norm = str(mode).strip().lower()
    if mode_norm == "zero":
        out[6] = 0.0
    elif mode_norm == "scale":
        out[6] = float(out[6] * float(scale))
    return out.astype(np.float32)


def _get_batch_tensor(batch, keys: tuple[str, ...]):
    # TensorDict path.
    try:
        return batch[keys]  # type: ignore[index]
    except Exception:
        pass

    # Dict-like fallback path.
    cur = batch
    for key in keys:
        cur = cur[key]
    return cur


def _assert_tensor_range(tensor, *, lo: float, hi: float, name: str) -> None:
    tmin = float(tensor.min().item())
    tmax = float(tensor.max().item())
    if tmin < lo or tmax > hi:
        raise ValueError(
            f"[strict-resfit] {name} out of range [{lo}, {hi}] "
            f"(min={tmin}, max={tmax})"
        )


def _slugify(text: str) -> str:
    return "".join(ch if (ch.isalnum() or ch in "._-") else "_" for ch in str(text)).strip("_")


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


def _parse_bool_text(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean text: {value}")


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

    # Keep RL env point2action behavior aligned with basefix_v1 line.
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


def _write_csv_row(csv_path: Path, row: dict[str, float | int]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _ensure_local_bddl(repo_root: Path) -> None:
    try:
        import bddl  # type: ignore  # noqa: F401
        return
    except Exception:
        pass

    third_party_dir = (repo_root / "point_policy_rl" / "third_party").resolve()
    if str(third_party_dir) not in sys.path:
        sys.path.insert(0, str(third_party_dir))

    try:
        import bddl  # type: ignore  # noqa: F401
        print(f"[deps] using vendored bddl from {third_party_dir}")
    except Exception as exc:
        print(f"[deps] bddl import failed after fallback path insert: {exc}")


def _ensure_gym_compat() -> None:
    try:
        import gym  # type: ignore  # noqa: F401
        return
    except Exception:
        pass

    try:
        import gymnasium as gymn  # type: ignore
    except Exception as exc:
        print(f"[deps] gym/gymnasium unavailable: {exc}")
        return

    sys.modules.setdefault("gym", gymn)
    for sub in ("spaces", "envs", "wrappers", "vector", "error", "utils", "logger"):
        try:
            sub_mod = importlib.import_module(f"gymnasium.{sub}")
            sys.modules.setdefault(f"gym.{sub}", sub_mod)
        except Exception:
            pass
    print("[deps] using gymnasium as gym compatibility layer")


def _ensure_torch_attention_compat() -> None:
    """
    Provide `torch.nn.attention` compatibility for torch versions where this
    module does not exist (e.g., older 2.x builds).
    """
    try:
        import torch.nn.attention  # type: ignore  # noqa: F401
        return
    except Exception:
        pass

    try:
        import torch
        import torch.nn.functional as F
    except Exception:
        return

    module_name = "torch.nn.attention"
    if module_name in sys.modules:
        # Module shim already installed in this process.
        pass
    else:
        compat = types.ModuleType(module_name)

        class _SDPBackend:
            FLASH_ATTENTION = "flash"
            EFFICIENT_ATTENTION = "efficient"
            MATH = "math"

        def _sdpa_kernel(_backends):
            # Best-effort fallback: use torch.backends.cuda.sdp_kernel if available,
            # otherwise no-op context manager.
            try:
                sdp_kernel_fn = getattr(getattr(torch.backends, "cuda", object()), "sdp_kernel", None)
                if callable(sdp_kernel_fn):
                    return sdp_kernel_fn(
                        enable_flash=True,
                        enable_mem_efficient=True,
                        enable_math=True,
                    )
            except Exception:
                pass
            return contextlib.nullcontext()

        compat.SDPBackend = _SDPBackend
        compat.sdpa_kernel = _sdpa_kernel
        sys.modules[module_name] = compat
        print("[deps] installed torch.nn.attention compatibility module")

    if not hasattr(F, "scaled_dot_product_attention"):
        def _scaled_dot_product_attention_compat(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        ):
            import math

            scale = 1.0 / math.sqrt(float(q.shape[-1]))
            attn = torch.matmul(q, k.transpose(-2, -1)) * scale

            if is_causal:
                t_q = int(attn.shape[-2])
                t_k = int(attn.shape[-1])
                causal = torch.tril(torch.ones((t_q, t_k), dtype=torch.bool, device=attn.device))
                attn = attn.masked_fill(~causal, float("-inf"))

            if attn_mask is not None:
                if hasattr(attn_mask, "dtype") and attn_mask.dtype == torch.bool:
                    attn = attn.masked_fill(~attn_mask, float("-inf"))
                else:
                    attn = attn + attn_mask

            probs = torch.softmax(attn, dim=-1)
            if float(dropout_p) > 0.0:
                probs = torch.dropout(probs, float(dropout_p), train=True)
            return torch.matmul(probs, v)

        F.scaled_dot_product_attention = _scaled_dot_product_attention_compat
        print("[deps] installed scaled_dot_product_attention compatibility function")


def _patch_torch_load_weights_only_default() -> None:
    try:
        import inspect
        import torch
    except Exception:
        return

    orig_load = torch.load
    if bool(getattr(orig_load, "_point_policy_rl_patched", False)):
        return

    try:
        sig = inspect.signature(orig_load)
        has_weights_only = "weights_only" in sig.parameters
    except Exception:
        has_weights_only = False
    if not has_weights_only:
        print("[deps] torch.load has no weights_only parameter; skip patch")
        return

    def _load_compat(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return orig_load(*args, **kwargs)

    _load_compat._point_policy_rl_patched = True  # type: ignore[attr-defined]
    torch.load = _load_compat  # type: ignore[assignment]
    print("[deps] patched torch.load default weights_only=False")


def _ensure_local_robomimic(repo_root: Path) -> None:
    try:
        import robomimic.utils.tensor_utils as _tensor_utils  # type: ignore  # noqa: F401
        return
    except Exception:
        pass

    candidates = [
        Path("/sjw_alinlab2/home/sanghyeok/robomimic"),
        (repo_root.parent / "robomimic"),
    ]
    for cand in candidates:
        cand = cand.expanduser().resolve()
        if not (cand / "robomimic" / "__init__.py").exists():
            continue
        if str(cand) not in sys.path:
            sys.path.insert(0, str(cand))
        try:
            import robomimic.utils.tensor_utils as _tensor_utils  # type: ignore  # noqa: F401
            print(f"[deps] using robomimic from {cand}")
            return
        except Exception:
            continue

    print("[deps] robomimic import unavailable (required by Point-Policy BC adapter)")


def _ensure_tabulate_compat() -> None:
    try:
        import tabulate  # type: ignore  # noqa: F401
        return
    except Exception:
        pass

    stub = types.ModuleType("tabulate")

    def _tabulate(rows, headers=(), **_kwargs):
        lines = []
        if headers:
            lines.append(" | ".join(str(h) for h in headers))
        for row in rows:
            lines.append(" | ".join(str(v) for v in row))
        return "\n".join(lines)

    stub.tabulate = _tabulate
    sys.modules["tabulate"] = stub
    print("[deps] using internal tabulate stub (tabulate package not found)")


def _ensure_resfit_common_utils_compat(resfit_root: Path):
    try:
        from resfit.rl_finetuning.off_policy.common_utils import utils as resfit_utils
        return resfit_utils
    except Exception as exc:
        print(
            "[deps] resfit common_utils import failed; using compatibility loader "
            f"reason={type(exc).__name__}: {exc}"
        )

    module_name = "resfit.rl_finetuning.off_policy.common_utils"
    utils_module_name = f"{module_name}.utils"
    if module_name in sys.modules and utils_module_name in sys.modules:
        return sys.modules[utils_module_name]

    import torch
    from torch import nn

    compat = types.ModuleType(module_name)
    compat.__path__ = []  # type: ignore[attr-defined]

    class RandomShiftsAug:
        def __init__(self, pad):
            self.pad = int(pad)

        def __call__(self, x):
            n, _c, h, w = x.size()
            assert h == w
            padding = tuple([self.pad] * 4)
            x = nn.functional.pad(x, padding, "replicate")
            eps = 1.0 / (h + 2 * self.pad)
            arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
            arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
            base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
            base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
            shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
            shift *= 2.0 / (h + 2 * self.pad)
            grid = base_grid + shift
            return nn.functional.grid_sample(x, grid, padding_mode="zeros", align_corners=False)

    def wrap_ruler(text: str, max_len=40):
        text_len = len(text)
        if text_len > max_len:
            return text
        left_len = (max_len - text_len) // 2
        right_len = max_len - text_len - left_len
        return ("=" * left_len) + text + ("=" * right_len)

    def count_parameters(model):
        total_params = 0
        for _name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            total_params += int(parameter.numel())
        print(f"trainable_params={total_params}")

    class MultiCounter:
        pass

    utils_path = (resfit_root / "resfit" / "rl_finetuning" / "off_policy" / "common_utils" / "utils.py").resolve()
    spec = importlib.util.spec_from_file_location(utils_module_name, str(utils_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load resfit utils module from {utils_path}")
    utils_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils_mod)

    compat.RandomShiftsAug = RandomShiftsAug
    compat.wrap_ruler = wrap_ruler
    compat.count_parameters = count_parameters
    compat.MultiCounter = MultiCounter
    compat.utils = utils_mod

    sys.modules[module_name] = compat
    sys.modules[utils_module_name] = utils_mod
    print("[deps] installed resfit common_utils compatibility module")
    return utils_mod


def _ensure_resfit_rlpd_compat():
    try:
        import resfit.rl_finetuning.config.rlpd as rlpd_mod  # type: ignore
        return rlpd_mod
    except Exception as exc:
        print(
            "[deps] resfit rlpd import failed; using compatibility dataclasses "
            f"reason={type(exc).__name__}: {exc}"
        )

    from dataclasses import dataclass, field
    from typing import Optional
    from torch import nn

    module_name = "resfit.rl_finetuning.config.rlpd"
    if module_name in sys.modules:
        return sys.modules[module_name]

    compat = types.ModuleType(module_name)

    @dataclass
    class VitEncoderConfig:
        depth: int = 1
        embed_dim: int = 128
        embed_norm: int = 0
        embed_style: str = "embed2"
        num_heads: int = 4
        patch_size: int = 8
        stride: int = -1
        act_layer = nn.GELU

    @dataclass
    class PointEncoderConfig:
        hidden_dim: int = 256
        embed_dim: int = 128
        use_layer_norm: bool = True

    @dataclass
    class CriticLossCfg:
        type: str = "mse"
        n_bins: int = 51
        v_min: float = 0.0
        v_max: float = 1.0
        sigma: float = -1.0

        def __post_init__(self):
            assert self.type in ["mse", "hl_gauss", "c51"]

    @dataclass
    class CriticConfig:
        drop: float = 0
        feature_dim: int = 128
        fuse_patch: int = 1
        hidden_dim: int = 1024
        norm_weight: int = 0
        orth: int = 1
        spatial_emb: int = 1024
        num_q: int = 10
        loss: CriticLossCfg = field(default_factory=CriticLossCfg)
        policy_gradient_type: str = "ensemble_mean"
        num_layers: int = 2
        use_layer_norm: bool = True
        min_q_heads: int = 2

        def __post_init__(self):
            assert self.policy_gradient_type in ["ensemble_mean", "min_random_pair", "q1"]

    @dataclass
    class ActorConfig:
        feature_dim: int = 128
        hidden_dim: int = 1024
        dropout: float = 0
        orth: int = 1
        max_action_norm: float = -1
        spatial_emb: int = 0
        actor_last_layer_init_scale: Optional[float] = None
        actor_last_layer_init_distribution: str = "normal"
        actor_intermediate_layer_init_distribution: str = "default"
        action_l2_reg_weight: float = 0.0
        action_scale: float = 1.0
        num_layers: int = 2
        use_layer_norm: bool = True

    @dataclass
    class QAgentConfig:
        device: str = "cuda"
        actor_lr: float = 1e-4
        critic_lr: float = 1e-4
        critic_target_tau: float = 0.005
        stddev_clip: float = 0.3
        lr_warmup_steps: int = 0
        lr_warmup_start: float = 1e-8
        use_prop: int = 1
        enc_type: str = "vit"
        vit: VitEncoderConfig = field(default_factory=VitEncoderConfig)
        point: PointEncoderConfig = field(default_factory=PointEncoderConfig)
        critic: CriticConfig = field(default_factory=CriticConfig)
        actor: ActorConfig = field(default_factory=ActorConfig)
        critic_grad_clip_norm: float = 1.0
        actor_grad_clip_norm: float = 1.0
        bc_loss_coef: float = 0.0
        bc_loss_dynamic: int = 0
        bc_backprop_encoder: bool = False
        freeze_encoder: bool = False
        clip_q_target_to_reward_range: bool = False
        target_action_noise: bool = True
        act_method: str = "rl"

    compat.VitEncoderConfig = VitEncoderConfig
    compat.PointEncoderConfig = PointEncoderConfig
    compat.CriticLossCfg = CriticLossCfg
    compat.CriticConfig = CriticConfig
    compat.ActorConfig = ActorConfig
    compat.QAgentConfig = QAgentConfig
    sys.modules[module_name] = compat
    print("[deps] installed resfit rlpd compatibility module")
    return compat


def _ensure_resfit_encoder_compat():
    try:
        import resfit.rl_finetuning.off_policy.networks.encoder as _enc_mod  # type: ignore
        return _enc_mod
    except Exception as exc:
        print(
            "[deps] resfit encoder import failed; using compatibility module "
            f"reason={type(exc).__name__}: {exc}"
        )

    module_name = "resfit.rl_finetuning.off_policy.networks.encoder"
    if module_name in sys.modules:
        return sys.modules[module_name]

    import torch
    from torch import nn
    from resfit.rl_finetuning.config.rlpd import PointEncoderConfig, VitEncoderConfig
    from resfit.rl_finetuning.off_policy.networks.min_vit import MinVit

    compat = types.ModuleType(module_name)

    class VitEncoder(nn.Module):
        def __init__(self, obs_shape, cfg):
            super().__init__()
            self.obs_shape = obs_shape
            self.cfg = cfg
            self.vit = MinVit(
                embed_style=cfg.embed_style,
                embed_dim=cfg.embed_dim,
                embed_norm=cfg.embed_norm,
                num_head=cfg.num_heads,
                depth=cfg.depth,
            )
            self.num_patch = self.vit.num_patches
            self.patch_repr_dim = self.cfg.embed_dim
            self.repr_dim = self.cfg.embed_dim * self.vit.num_patches

        def forward(self, obs, flatten=True):
            if obs.max() > 5:
                obs = obs / 255.0
            obs = obs - 0.5
            feats = self.vit.forward(obs)
            if flatten:
                feats = feats.flatten(1, 2)
            return feats

    class PointEncoder(nn.Module):
        def __init__(self, obs_shape, cfg):
            super().__init__()
            self.obs_shape = tuple(obs_shape)
            self.cfg = cfg
            if len(self.obs_shape) == 1:
                self.num_tokens = 1
                token_dim = int(self.obs_shape[0])
            elif len(self.obs_shape) == 2:
                self.num_tokens = int(self.obs_shape[0])
                token_dim = int(self.obs_shape[1])
            else:
                raise ValueError(
                    "PointEncoder expects rank-1 or rank-2 obs_shape, "
                    f"got obs_shape={self.obs_shape}"
                )
            layers = [
                nn.Linear(token_dim, int(getattr(cfg, "hidden_dim", 256))),
                nn.GELU(),
                nn.Linear(int(getattr(cfg, "hidden_dim", 256)), int(getattr(cfg, "embed_dim", 128))),
            ]
            if bool(getattr(cfg, "use_layer_norm", True)):
                layers.append(nn.LayerNorm(int(getattr(cfg, "embed_dim", 128))))
            self.token_mlp = nn.Sequential(*layers)
            self.patch_repr_dim = int(getattr(cfg, "embed_dim", 128))
            self.repr_dim = int(self.num_tokens * self.patch_repr_dim)

        def forward(self, obs, flatten=True):
            obs = obs.float()
            if obs.dim() == 2:
                obs = obs.unsqueeze(1)
            if obs.dim() != 3:
                raise ValueError(f"PointEncoder expects [B,N,D], got shape={tuple(obs.shape)}")
            feats = self.token_mlp(obs)
            if flatten:
                feats = feats.flatten(1, 2)
            return feats

    compat.VitEncoder = VitEncoder
    compat.PointEncoder = PointEncoder
    compat.VitEncoderConfig = VitEncoderConfig
    compat.PointEncoderConfig = PointEncoderConfig
    sys.modules[module_name] = compat
    print("[deps] installed resfit encoder compatibility module")
    return compat


def _load_module_with_future_annotations(module_name: str, file_path: Path):
    file_path = Path(file_path).resolve()
    source = file_path.read_text(encoding="utf-8")
    if "from __future__ import annotations" not in source.splitlines()[:5]:
        source = "from __future__ import annotations\n" + source
    module = types.ModuleType(module_name)
    module.__file__ = str(file_path)
    module.__package__ = module_name.rsplit(".", 1)[0]
    code = compile(source, str(file_path), "exec")
    exec(code, module.__dict__)
    sys.modules[module_name] = module
    return module


def _ensure_resfit_actor_compat(resfit_root: Path):
    module_name = "resfit.rl_finetuning.off_policy.rl.actor"
    try:
        import resfit.rl_finetuning.off_policy.rl.actor as actor_mod  # type: ignore
        return actor_mod
    except Exception as exc:
        print(
            "[deps] resfit actor import failed; using future-annotations loader "
            f"reason={type(exc).__name__}: {exc}"
        )

    if module_name in sys.modules:
        return sys.modules[module_name]

    actor_path = (
        Path(resfit_root).resolve()
        / "resfit"
        / "rl_finetuning"
        / "off_policy"
        / "rl"
        / "actor.py"
    )
    actor_mod = _load_module_with_future_annotations(module_name, actor_path)
    print("[deps] installed resfit actor compatibility module")
    return actor_mod


def _ensure_resfit_critic_compat(resfit_root: Path):
    module_name = "resfit.rl_finetuning.off_policy.rl.critic"
    try:
        import resfit.rl_finetuning.off_policy.rl.critic as critic_mod  # type: ignore
        return critic_mod
    except Exception as exc:
        print(
            "[deps] resfit critic import failed; using torch.func compatibility loader "
            f"reason={type(exc).__name__}: {exc}"
        )

    if module_name in sys.modules:
        return sys.modules[module_name]

    critic_path = (
        Path(resfit_root).resolve()
        / "resfit"
        / "rl_finetuning"
        / "off_policy"
        / "rl"
        / "critic.py"
    )
    source = critic_path.read_text(encoding="utf-8")
    target_line = "from torch.func import functional_call, stack_module_state, vmap"
    fallback_block = """
try:
    from torch.func import functional_call, stack_module_state, vmap
except Exception:
    from collections import OrderedDict
    try:
        from torch.nn.utils.stateless import functional_call as _stateless_functional_call
    except Exception:
        _stateless_functional_call = None

    def stack_module_state(modules):
        if len(modules) == 0:
            return OrderedDict(), OrderedDict()
        first_params = OrderedDict(modules[0].named_parameters())
        first_buffers = OrderedDict(modules[0].named_buffers())
        params = OrderedDict()
        buffers = OrderedDict()
        for key in first_params.keys():
            params[key] = torch.stack(
                [OrderedDict(m.named_parameters())[key].detach().clone() for m in modules], dim=0
            )
        for key in first_buffers.keys():
            buffers[key] = torch.stack(
                [OrderedDict(m.named_buffers())[key].detach().clone() for m in modules], dim=0
            )
        return params, buffers

    def functional_call(module, state, args):
        params, buffers = state
        merged = {}
        merged.update(params)
        merged.update(buffers)
        if _stateless_functional_call is not None:
            return _stateless_functional_call(module, merged, args)
        with torch.no_grad():
            old_state = module.state_dict()
            module.load_state_dict(merged, strict=False)
        out = module(*args)
        with torch.no_grad():
            module.load_state_dict(old_state, strict=False)
        return out

    def vmap(fn, in_dims=(0, 0, None)):
        def _wrapped(params, buffers, shared):
            first = next(iter(params.values()))
            n = int(first.shape[0])
            outs = []
            for i in range(n):
                # Use cloned per-head tensors to avoid view/version conflicts on older torch.
                p_i = {k: v[i].clone() for k, v in params.items()}
                b_i = {k: v[i].clone() for k, v in buffers.items()}
                outs.append(fn(p_i, b_i, shared))
            return torch.stack(outs, dim=0)

        return _wrapped
"""
    if target_line in source:
        source = source.replace(target_line, fallback_block)
    source = source.replace("nn.ReLU(inplace=True)", "nn.ReLU()")

    module = types.ModuleType(module_name)
    module.__file__ = str(critic_path)
    module.__package__ = module_name.rsplit(".", 1)[0]
    code = compile(source, str(critic_path), "exec")
    exec(code, module.__dict__)

    # In older torch fallback paths, ensemble output can appear as [B, H, ...]
    # while q_agent expects [H, B, ...]. Normalize shape here.
    try:
        import torch

        nn = torch.nn
        HeadMLP = module.HeadMLP
        resfit_utils = module.utils

        class _SimpleSpatialEmbQEnsemble(nn.Module):
            """
            Fallback ensemble that avoids torch.func/stateless functional_call.
            Uses explicit ModuleList heads and stacks outputs as [H, B, out].
            """

            def __init__(
                self,
                *,
                num_patch: int,
                patch_dim: int,
                prop_dim: int,
                action_dim: int,
                fuse_patch: int,
                emb_dim: int,
                hidden_dim: int,
                orth: int,
                output_dim: int = 1,
                num_heads: int = 2,
                num_layers: int = 2,
                use_layer_norm: bool = True,
            ):
                super().__init__()
                if fuse_patch:
                    proj_in_dim = num_patch + action_dim + prop_dim
                    num_proj = patch_dim
                else:
                    proj_in_dim = patch_dim + action_dim + prop_dim
                    num_proj = num_patch

                self.fuse_patch = fuse_patch
                self.patch_dim = patch_dim
                self.prop_dim = prop_dim
                self.action_dim = action_dim
                self.num_heads = num_heads

                input_layers = [nn.Linear(proj_in_dim, emb_dim)]
                if use_layer_norm:
                    input_layers.append(nn.LayerNorm(emb_dim))
                input_layers.append(nn.ReLU(inplace=True))
                self.input_proj = nn.Sequential(*input_layers)
                self.weight = nn.Parameter(torch.zeros(1, num_proj, emb_dim))
                nn.init.normal_(self.weight)

                input_dim = emb_dim + action_dim + prop_dim
                self.heads = nn.ModuleList(
                    [HeadMLP(input_dim, hidden_dim, output_dim, num_layers, use_layer_norm) for _ in range(num_heads)]
                )
                if orth:
                    for head in self.heads:
                        head.apply(resfit_utils.orth_weight_init)

            def extra_repr(self) -> str:
                return f"heads: {self.num_heads}, weight: nn.Parameter ({self.weight.size()})"

            def _compute_trunk(self, feat: torch.Tensor, prop: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
                assert feat.size(-1) == self.patch_dim, "are you using CNN, need flatten&transpose"
                if self.fuse_patch:
                    feat = feat.transpose(1, 2)

                repeated_action = action.unsqueeze(1).repeat(1, feat.size(1), 1)
                all_feats = [feat, repeated_action]
                if self.prop_dim > 0:
                    repeated_prop = prop.unsqueeze(1).repeat(1, feat.size(1), 1)
                    all_feats.append(repeated_prop)

                x = torch.cat(all_feats, dim=-1)
                y = self.input_proj(x)
                z = (self.weight * y).sum(1)
                if self.prop_dim == 0:
                    z = torch.cat((z, action), dim=-1)
                else:
                    z = torch.cat((z, prop, action), dim=-1)
                return z

            def forward(self, feat: torch.Tensor, prop: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
                z = self._compute_trunk(feat, prop, action)
                out = [head(z) for head in self.heads]
                return torch.stack(out, dim=0)

        module.SpatialEmbQEnsemble = _SimpleSpatialEmbQEnsemble

        def _fix_head_batch_dims(out, batch_size: int, num_heads: int):
            if (
                isinstance(out, torch.Tensor)
                and out.ndim >= 2
                and int(out.shape[0]) == int(batch_size)
                and int(out.shape[1]) == int(num_heads)
            ):
                return out.transpose(0, 1).contiguous()
            return out

        _orig_forward = module.SpatialEmbQEnsemble.forward

        def _forward_with_head_batch_fix(self, feat, prop, action):
            out = _orig_forward(self, feat, prop, action)
            return _fix_head_batch_dims(out, int(feat.shape[0]), int(getattr(self, "num_heads", -1)))

        module.SpatialEmbQEnsemble.forward = _forward_with_head_batch_fix

        _orig_critic_forward = module.Critic.forward

        def _critic_forward_with_head_batch_fix(self, feat, prop, act, *, return_logits=False):
            out = _orig_critic_forward(self, feat, prop, act, return_logits=return_logits)
            heads = int(getattr(self.cfg, "num_q", getattr(self.q_ensemble, "num_heads", -1)))
            batch = int(feat.shape[0])
            if return_logits:
                q_per_head, logits_per_head = out
                return (
                    _fix_head_batch_dims(q_per_head, batch, heads),
                    _fix_head_batch_dims(logits_per_head, batch, heads),
                )
            return _fix_head_batch_dims(out, batch, heads)

        module.Critic.forward = _critic_forward_with_head_batch_fix
    except Exception as exc:
        print(f"[deps] skipped critic head/batch shape fix: {exc}")

    sys.modules[module_name] = module
    print("[deps] installed resfit critic compatibility module")
    return module


def _ensure_resfit_qagent_compat(resfit_root: Path):
    module_name = "resfit.rl_finetuning.off_policy.rl.q_agent"
    if module_name in sys.modules:
        return sys.modules[module_name]

    qagent_path = (
        Path(resfit_root).resolve()
        / "resfit"
        / "rl_finetuning"
        / "off_policy"
        / "rl"
        / "q_agent.py"
    )
    source = qagent_path.read_text(encoding="utf-8")

    target_line = 'q_all = self.critic(obs["feat"], obs["observation.state"], action).squeeze(-1)  # [K,B]'
    target_qmin_line = "target_q_min = target_all.squeeze(-1)  # [B]"
    target_q_line = "target_q = (reward + (discount * target_q_min)).detach()"

    target_qmin_replacement = """
            target_q_min = target_all.squeeze(-1)
            batch_n = int(reward.shape[0])
            if target_q_min.ndim == 1:
                pass
            elif target_q_min.ndim == 2:
                if target_q_min.shape[1] == batch_n:
                    target_q_min = target_q_min.min(dim=0).values
                elif target_q_min.shape[0] == batch_n:
                    target_q_min = target_q_min.min(dim=1).values
                elif target_q_min.numel() % batch_n == 0:
                    target_q_min = target_q_min.reshape(-1, batch_n).min(dim=0).values
                else:
                    raise RuntimeError(f"Unexpected target_q_min shape {tuple(target_q_min.shape)} for batch {batch_n}")
            else:
                if target_q_min.shape[-1] == batch_n:
                    target_q_min = target_q_min.reshape(-1, batch_n).min(dim=0).values
                elif target_q_min.shape[0] == batch_n:
                    target_q_min = target_q_min.reshape(batch_n, -1).min(dim=1).values
                elif target_q_min.numel() % batch_n == 0:
                    target_q_min = target_q_min.reshape(-1, batch_n).min(dim=0).values
                else:
                    raise RuntimeError(f"Unexpected target_q_min shape {tuple(target_q_min.shape)} for batch {batch_n}")
"""

    target_q_replacement = """
            target_q = (reward + (discount * target_q_min)).detach()
            batch_n = int(reward.shape[0])
            if target_q.ndim != 1:
                if target_q.shape[-1] == batch_n:
                    target_q = target_q.reshape(-1, batch_n).mean(dim=0)
                elif target_q.shape[0] == batch_n:
                    target_q = target_q.reshape(batch_n, -1).mean(dim=1)
                elif target_q.numel() == batch_n:
                    target_q = target_q.reshape(batch_n)
                elif target_q.numel() % batch_n == 0:
                    target_q = target_q.reshape(-1, batch_n).mean(dim=0)
                else:
                    raise RuntimeError(f"Unexpected target_q shape {tuple(target_q.shape)} for batch {batch_n}")
"""

    replacement_block = """
            q_all = self.critic(obs["feat"], obs["observation.state"], action).squeeze(-1)  # [K,B]
            # Compatibility: normalize q_all to [K, B] regardless of backend/fallback path.
            batch_n = int(target_q.shape[0])
            if q_all.ndim == 1:
                q_all = q_all.unsqueeze(0)
            elif q_all.ndim == 2:
                if q_all.shape[1] == batch_n:
                    pass
                elif q_all.shape[0] == batch_n:
                    q_all = q_all.transpose(0, 1).contiguous()
                elif q_all.numel() % batch_n == 0:
                    q_all = q_all.reshape(-1, batch_n)
                else:
                    raise RuntimeError(f"Unexpected q_all shape {tuple(q_all.shape)} for batch {batch_n}")
            else:
                if q_all.shape[-1] == batch_n:
                    q_all = q_all.reshape(-1, batch_n)
                elif q_all.shape[0] == batch_n:
                    q_all = q_all.movedim(0, -1).reshape(-1, batch_n)
                elif q_all.numel() % batch_n == 0:
                    q_all = q_all.reshape(-1, batch_n)
                else:
                    raise RuntimeError(f"Unexpected q_all shape {tuple(q_all.shape)} for batch {batch_n}")
"""
    if target_line in source:
        source = source.replace(target_line, replacement_block)
    else:
        print("[deps] q_agent compatibility patch target not found; keeping upstream source")

    if target_qmin_line in source:
        source = source.replace(target_qmin_line, target_qmin_replacement)

    if target_q_line in source:
        source = source.replace(target_q_line, target_q_replacement)

    td_error_line = 'td_errors = torch.abs(q_all - target_q.unsqueeze(0)).mean(dim=0)  # [B] - mean across heads'
    td_error_replacement = (
        "td_errors = torch.abs(q_all - target_q.reshape(1, -1)).mean(dim=0)  # [B] - mean across heads"
    )
    if td_error_line in source:
        source = source.replace(td_error_line, td_error_replacement)

    retain_graph_line = "critic_loss.backward(retain_graph=True)"
    if retain_graph_line in source:
        source = source.replace(retain_graph_line, "critic_loss.backward()")

    module = types.ModuleType(module_name)
    module.__file__ = str(qagent_path)
    module.__package__ = module_name.rsplit(".", 1)[0]
    code = compile(source, str(qagent_path), "exec")
    exec(code, module.__dict__)
    sys.modules[module_name] = module
    print("[deps] installed resfit q_agent compatibility module")
    return module


def _parse_tags(tags_raw: str) -> list[str]:
    tags = [tag.strip() for tag in str(tags_raw).split(",")]
    return [tag for tag in tags if tag]


def _init_wandb(
    args: argparse.Namespace,
    run_dir: Path,
    run_meta: dict[str, Any],
):
    if not bool(args.wandb_enable):
        return None, None
    if str(args.wandb_mode).strip().lower() == "disabled":
        print("[wandb] disabled by --wandb-mode=disabled")
        return None, None

    try:
        import wandb  # type: ignore
    except Exception as exc:
        print(f"[wandb] import failed: {exc}. Continue without W&B logging.")
        return None, None

    try:
        wb_run = wandb.init(
            project=str(args.wandb_project),
            entity=(str(args.wandb_entity).strip() or None),
            group=(str(args.wandb_group).strip() or None),
            name=(str(args.wandb_name).strip() or run_dir.name),
            tags=_parse_tags(args.wandb_tags),
            mode=str(args.wandb_mode),
            dir=str(run_dir),
            config=run_meta,
            reinit=True,
        )
    except Exception as exc:
        print(f"[wandb] init failed: {exc}. Continue without W&B logging.")
        return None, None

    print(f"[wandb] run initialized: project={args.wandb_project}, name={wb_run.name}")
    return wandb, wb_run


def _split_state_vec(state_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    state_vec = np.asarray(state_vec, dtype=np.float32).reshape(-1)
    if state_vec.shape[0] < 8:
        raise ValueError(f"state vector too small: shape={state_vec.shape}")
    return state_vec[:-7].astype(np.float32), state_vec[-7:].astype(np.float32)


def _prop_to_dummy_image(prop_state: np.ndarray, image_size: int) -> np.ndarray:
    prop_state = np.asarray(prop_state, dtype=np.float32).reshape(-1)
    img = np.zeros((3, image_size, image_size), dtype=np.uint8)
    n = min(prop_state.shape[0], image_size * image_size)
    if n > 0:
        vals = np.clip(np.tanh(prop_state[:n]), -1.0, 1.0)
        img[0].reshape(-1)[:n] = ((vals + 1.0) * 127.5).astype(np.uint8)
    return img


def _state_to_agent_obs(
    state_vec: np.ndarray,
    action_normalizer: LinearActionNormalizerRL,
    camera_key: str,
    dummy_image_size: int,
):
    import torch

    prop_state, base_action_raw = _split_state_vec(state_vec)
    base_action_norm = action_normalizer.normalize(base_action_raw)
    dummy_img = _prop_to_dummy_image(prop_state, dummy_image_size)
    return {
        camera_key: torch.as_tensor(dummy_img, dtype=torch.uint8),
        "observation.state": torch.as_tensor(prop_state, dtype=torch.float32),
        "observation.base_action": torch.as_tensor(base_action_norm, dtype=torch.float32),
    }


def _state_to_agent_obs_batched(
    state_vec: np.ndarray,
    action_normalizer: LinearActionNormalizerRL,
    camera_key: str,
    dummy_image_size: int,
    device: str,
):
    import torch

    obs = _state_to_agent_obs(
        state_vec=state_vec,
        action_normalizer=action_normalizer,
        camera_key=camera_key,
        dummy_image_size=dummy_image_size,
    )
    return {
        camera_key: obs[camera_key].unsqueeze(0).to(device),
        "observation.state": obs["observation.state"].unsqueeze(0).to(device),
        "observation.base_action": obs["observation.base_action"].unsqueeze(0).to(device),
    }


def _resolve_nested_value(obs: dict[str, Any], dotted_key: str) -> Any | None:
    cur: Any = obs
    for part in str(dotted_key).split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _normalize_image_to_chw_uint8(
    image: np.ndarray,
    *,
    target_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    if arr.ndim != 3:
        raise ValueError(f"Unsupported image ndim={arr.ndim}, shape={arr.shape}")

    # Accept both CHW and HWC.
    if arr.shape[0] in (1, 3) and arr.shape[-1] not in (1, 3):
        chw = arr
    else:
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=2)
        if arr.shape[-1] > 3:
            arr = arr[..., :3]
        chw = np.transpose(arr, (2, 0, 1))

    if chw.dtype != np.uint8:
        chw_f = np.asarray(chw, dtype=np.float32)
        if float(np.nanmax(chw_f)) <= 1.5:
            chw_f = chw_f * 255.0
        chw = np.clip(chw_f, 0.0, 255.0).astype(np.uint8)

    if target_hw is not None:
        target_h, target_w = int(target_hw[0]), int(target_hw[1])
        if target_h > 0 and target_w > 0:
            c, h, w = chw.shape
            if (h, w) != (target_h, target_w):
                if cv2 is None:
                    raise RuntimeError(
                        "cv2 is required to resize image observations, but it is unavailable."
                    )
                hwc = np.transpose(chw, (1, 2, 0))
                hwc = cv2.resize(hwc, dsize=(target_w, target_h), interpolation=cv2.INTER_AREA)
                chw = np.transpose(hwc, (2, 0, 1)).astype(np.uint8)
    return chw


def _extract_image_from_obs(
    *,
    obs: dict[str, Any],
    pixel_key: str,
    image_obs_key: str,
    target_hw: tuple[int, int] | None = None,
) -> tuple[np.ndarray, str]:
    explicit = str(image_obs_key).strip()
    suffix = str(explicit).split(".")[-1] if explicit else ""
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    if pixel_key:
        candidates.append(str(pixel_key))
    if suffix:
        candidates.append(suffix)
        if not suffix.endswith("_image"):
            candidates.append(f"{suffix}_image")
    candidates.extend(
        [
            "pixels1",
            "pixels2",
            "agentview_image",
            "robot0_eye_in_hand_image",
        ]
    )

    seen: set[str] = set()
    for key in candidates:
        if key in seen:
            continue
        seen.add(key)
        value = obs.get(key, None)
        if value is None and "." in key:
            value = _resolve_nested_value(obs, key)
        if value is None:
            continue
        chw = _normalize_image_to_chw_uint8(np.asarray(value), target_hw=target_hw)
        return chw, key

    available = sorted([str(k) for k in obs.keys()])
    raise KeyError(
        "Image key not found in observation. "
        f"requested={explicit or '<auto>'}, pixel_key={pixel_key}, available={available}"
    )


def _obs_to_agent_obs_image_only(
    *,
    obs_raw: dict[str, Any],
    base_action_raw: np.ndarray,
    action_normalizer: LinearActionNormalizerRL,
    camera_key: str,
    pixel_key: str,
    image_obs_key: str,
    image_hw: tuple[int, int] | None,
    prop_dim: int,
):
    import torch

    image_chw, _ = _extract_image_from_obs(
        obs=obs_raw,
        pixel_key=pixel_key,
        image_obs_key=image_obs_key,
        target_hw=image_hw,
    )
    base_action_norm = action_normalizer.normalize(base_action_raw)
    prop_state = np.zeros((max(int(prop_dim), 1),), dtype=np.float32)
    return {
        camera_key: torch.as_tensor(image_chw, dtype=torch.uint8),
        "observation.state": torch.as_tensor(prop_state, dtype=torch.float32),
        "observation.base_action": torch.as_tensor(base_action_norm, dtype=torch.float32),
    }


def _build_agent_obs(
    *,
    residual_obs_mode: str,
    state_vec: np.ndarray | None,
    obs_raw: dict[str, Any] | None,
    base_action_raw: np.ndarray | None,
    action_normalizer: LinearActionNormalizerRL,
    camera_key: str,
    dummy_image_size: int,
    pixel_key: str,
    image_obs_key: str,
    image_hw: tuple[int, int] | None,
    image_only_prop_dim: int,
):
    mode = str(residual_obs_mode).strip().lower()
    if mode == "image_only":
        if obs_raw is None or base_action_raw is None:
            raise ValueError("image_only mode requires obs_raw and base_action_raw.")
        return _obs_to_agent_obs_image_only(
            obs_raw=obs_raw,
            base_action_raw=base_action_raw,
            action_normalizer=action_normalizer,
            camera_key=camera_key,
            pixel_key=pixel_key,
            image_obs_key=image_obs_key,
            image_hw=image_hw,
            prop_dim=image_only_prop_dim,
        )
    if state_vec is None:
        raise ValueError("point_state_dummy_image mode requires state_vec.")
    return _state_to_agent_obs(
        state_vec=state_vec,
        action_normalizer=action_normalizer,
        camera_key=camera_key,
        dummy_image_size=dummy_image_size,
    )


def _build_agent_obs_batched(
    *,
    residual_obs_mode: str,
    state_vec: np.ndarray | None,
    obs_raw: dict[str, Any] | None,
    base_action_raw: np.ndarray | None,
    action_normalizer: LinearActionNormalizerRL,
    camera_key: str,
    dummy_image_size: int,
    pixel_key: str,
    image_obs_key: str,
    image_hw: tuple[int, int] | None,
    image_only_prop_dim: int,
    device: str,
):
    obs = _build_agent_obs(
        residual_obs_mode=residual_obs_mode,
        state_vec=state_vec,
        obs_raw=obs_raw,
        base_action_raw=base_action_raw,
        action_normalizer=action_normalizer,
        camera_key=camera_key,
        dummy_image_size=dummy_image_size,
        pixel_key=pixel_key,
        image_obs_key=image_obs_key,
        image_hw=image_hw,
        image_only_prop_dim=image_only_prop_dim,
    )
    return {
        camera_key: obs[camera_key].unsqueeze(0).to(device),
        "observation.state": obs["observation.state"].unsqueeze(0).to(device),
        "observation.base_action": obs["observation.base_action"].unsqueeze(0).to(device),
    }


class _SimpleBatchRL(dict):
    def __getitem__(self, key):
        if isinstance(key, tuple):
            cur = self
            for k in key:
                cur = dict.__getitem__(cur, k)
            return cur
        return dict.__getitem__(self, key)

    def to(self, device, non_blocking=True):
        import torch

        def _move(x):
            if isinstance(x, torch.Tensor):
                return x.to(device=device, non_blocking=non_blocking)
            if isinstance(x, dict):
                return {k: _move(v) for k, v in x.items()}
            return x

        return _SimpleBatchRL(_move(dict(self)))

    @staticmethod
    def cat(parts: list):
        import torch

        dict_parts = [dict(p) if isinstance(p, _SimpleBatchRL) else p for p in parts]

        def _cat(values):
            first = values[0]
            if isinstance(first, torch.Tensor):
                return torch.cat(values, dim=0)
            if isinstance(first, dict):
                return {k: _cat([v[k] for v in values]) for k in first.keys()}
            raise TypeError(f"Unsupported batch value type: {type(first)}")

        return _SimpleBatchRL(_cat(dict_parts))


class _SimpleReplayBufferRL:
    def __init__(self, max_size: int, batch_size: int, gamma: float, n_step: int):
        self.max_size = int(max_size)
        self.batch_size = max(int(batch_size), 1)
        self.gamma = float(gamma)
        self.n_step = max(int(n_step), 1)
        self._storage: list[dict[str, Any]] = []
        self._ptr = 0
        self._pending: deque[dict[str, Any]] = deque()

    @staticmethod
    def _to_float(x: Any) -> float:
        if hasattr(x, "item"):
            try:
                return float(x.item())
            except Exception:
                pass
        return float(x)

    @staticmethod
    def _to_bool(x: Any) -> bool:
        if hasattr(x, "item"):
            try:
                return bool(x.item())
            except Exception:
                pass
        return bool(x)

    def _append_storage(self, transition: dict[str, Any]) -> None:
        if len(self._storage) < self.max_size:
            self._storage.append(transition)
        else:
            self._storage[self._ptr] = transition
            self._ptr = (self._ptr + 1) % self.max_size

    def _build_nstep_transition(self) -> dict[str, Any]:
        import torch

        first = self._pending[0]
        reward_sum = 0.0
        discount = 1.0
        steps = 0
        done = False
        next_obs = first["next_obs"]

        for tr in list(self._pending)[: self.n_step]:
            reward_sum += discount * self._to_float(tr["reward"])
            steps += 1
            done = self._to_bool(tr["done"])
            next_obs = tr["next_obs"]
            if done:
                break
            discount *= self.gamma

        gamma_pow = float(self.gamma**steps)
        nonterminal = float((steps == self.n_step) and (not done))
        return {
            "obs": first["obs"],
            "next_obs": next_obs,
            "action": first["action"],
            "reward": torch.tensor(reward_sum, dtype=torch.float32),
            "done": torch.tensor(done, dtype=torch.bool),
            "gamma": torch.tensor(gamma_pow, dtype=torch.float32),
            "nonterminal": torch.tensor(nonterminal, dtype=torch.float32),
            "_priority": first["_priority"],
        }

    def add(self, transition: dict[str, Any]) -> None:
        self._pending.append(transition)
        transition_done = self._to_bool(transition["done"])

        if transition_done:
            while self._pending:
                self._append_storage(self._build_nstep_transition())
                self._pending.popleft()
            return

        if len(self._pending) >= self.n_step:
            self._append_storage(self._build_nstep_transition())
            self._pending.popleft()

    def sample(self, batch_size: int):
        import torch

        n = max(int(batch_size), 1)
        idx = np.random.randint(0, len(self._storage), size=n)
        transitions = [self._storage[i] for i in idx]

        obs_keys = list(transitions[0]["obs"].keys())
        next_obs_keys = list(transitions[0]["next_obs"].keys())

        obs = {k: torch.stack([tr["obs"][k] for tr in transitions], dim=0) for k in obs_keys}
        next_obs = {k: torch.stack([tr["next_obs"][k] for tr in transitions], dim=0) for k in next_obs_keys}
        action = torch.stack([tr["action"] for tr in transitions], dim=0)
        reward = torch.stack([tr["reward"] for tr in transitions], dim=0).view(n, 1)
        done = torch.stack([tr["done"] for tr in transitions], dim=0).view(n, 1)
        nonterminal = torch.stack([tr["nonterminal"] for tr in transitions], dim=0).view(n, 1)
        gamma = torch.stack([tr["gamma"] for tr in transitions], dim=0).view(n, 1)
        priority = torch.stack([tr["_priority"] for tr in transitions], dim=0).view(n, 1)

        return _SimpleBatchRL(
            {
                "obs": obs,
                "next": {
                    "obs": next_obs,
                    "done": done,
                    "reward": reward,
                },
                "action": action,
                "gamma": gamma,
                "nonterminal": nonterminal,
                "_priority": priority,
            }
        )

    def update_tensordict_priority(self, _batch) -> None:
        # Uniform replay fallback: no-op.
        return

    def __len__(self) -> int:
        return len(self._storage)


def _move_batch_to_device(batch, device):
    if hasattr(batch, "to"):
        try:
            return batch.to(device, non_blocking=True)
        except TypeError:
            return batch.to(device)
    return batch


def _concat_batches(parts: list):
    import torch

    if len(parts) == 1:
        return parts[0]
    if isinstance(parts[0], _SimpleBatchRL):
        return _SimpleBatchRL.cat(parts)
    try:
        return torch.cat(parts, dim=0)
    except Exception:
        return _SimpleBatchRL.cat(parts)


def _add_transition_to_rb(
    rb,
    *,
    state_vec: np.ndarray | None,
    next_state_vec: np.ndarray | None,
    obs_raw: dict[str, Any] | None,
    next_obs_raw: dict[str, Any] | None,
    base_action_raw: np.ndarray | None,
    next_base_action_raw: np.ndarray | None,
    combined_action_raw: np.ndarray,
    reward: float,
    done: bool,
    action_normalizer: LinearActionNormalizerRL,
    residual_obs_mode: str,
    camera_key: str,
    dummy_image_size: int,
    pixel_key: str,
    image_obs_key: str,
    image_hw: tuple[int, int] | None,
    image_only_prop_dim: int,
) -> None:
    import torch

    obs = _build_agent_obs(
        residual_obs_mode=residual_obs_mode,
        state_vec=state_vec,
        obs_raw=obs_raw,
        base_action_raw=base_action_raw,
        action_normalizer=action_normalizer,
        camera_key=camera_key,
        dummy_image_size=dummy_image_size,
        pixel_key=pixel_key,
        image_obs_key=image_obs_key,
        image_hw=image_hw,
        image_only_prop_dim=image_only_prop_dim,
    )
    next_obs = _build_agent_obs(
        residual_obs_mode=residual_obs_mode,
        state_vec=next_state_vec,
        obs_raw=next_obs_raw,
        base_action_raw=next_base_action_raw,
        action_normalizer=action_normalizer,
        camera_key=camera_key,
        dummy_image_size=dummy_image_size,
        pixel_key=pixel_key,
        image_obs_key=image_obs_key,
        image_hw=image_hw,
        image_only_prop_dim=image_only_prop_dim,
    )

    action_norm = action_normalizer.normalize(combined_action_raw)
    if isinstance(rb, _SimpleReplayBufferRL):
        rb.add(
            {
                "obs": {k: v.detach().cpu() for k, v in obs.items()},
                "next_obs": {k: v.detach().cpu() for k, v in next_obs.items()},
                "action": torch.as_tensor(action_norm, dtype=torch.float32),
                "reward": torch.tensor(float(reward), dtype=torch.float32),
                "done": torch.tensor(bool(done), dtype=torch.bool),
                "_priority": torch.tensor(10.0, dtype=torch.float32),
            }
        )
        return

    from tensordict import TensorDict

    td = TensorDict(
        {
            "obs": TensorDict(obs, batch_size=[]),
            "next": TensorDict(
                {
                    "obs": TensorDict(next_obs, batch_size=[]),
                    "done": torch.tensor(bool(done), dtype=torch.bool),
                    "reward": torch.tensor(float(reward), dtype=torch.float32),
                },
                batch_size=[],
            ),
            "action": torch.as_tensor(action_norm, dtype=torch.float32),
            "_priority": torch.tensor(10.0, dtype=torch.float32),
        },
        batch_size=[],
    ).unsqueeze(0)
    rb.add(td)


def _select_online_bucket_transitions(
    episode_transitions: list[dict[str, Any]],
    *,
    episode_success: bool,
    fail_max_steps: int,
    fail_stride: int,
) -> list[dict[str, Any]]:
    if not episode_transitions:
        return []
    if episode_success:
        return episode_transitions

    stride = max(int(fail_stride), 1)
    selected = episode_transitions[::stride]

    max_steps = int(fail_max_steps)
    if max_steps > 0:
        selected = selected[:max_steps]

    # Keep terminal transition if it was dropped by stride/cap.
    last_tr = episode_transitions[-1]
    if selected:
        if selected[-1] is not last_tr:
            selected.append(last_tr)
    else:
        selected = [last_tr]
    return selected


def _sample_mixed_batch(
    *,
    online_rb,
    offline_rb,
    online_batch_size: int,
    offline_batch_size: int,
    fallback_batch_size: int,
    device,
    online_success_rb=None,
    online_fail_rb=None,
    online_success_batch_fraction: float = 0.5,
):
    parts = []
    used_online = 0
    used_offline = 0

    online_parts = []
    if online_batch_size > 0:
        if (
            online_success_rb is not None
            and online_fail_rb is not None
            and (len(online_success_rb) > 0 or len(online_fail_rb) > 0)
        ):
            frac = float(np.clip(online_success_batch_fraction, 0.0, 1.0))
            success_target = int(round(float(online_batch_size) * frac))
            success_target = int(np.clip(success_target, 0, online_batch_size))
            fail_target = int(online_batch_size - success_target)

            success_take = min(success_target, len(online_success_rb))
            fail_take = min(fail_target, len(online_fail_rb))

            if success_take > 0:
                online_parts.append(
                    _move_batch_to_device(online_success_rb.sample(success_take), device)
                )
            if fail_take > 0:
                online_parts.append(
                    _move_batch_to_device(online_fail_rb.sample(fail_take), device)
                )

            sampled_online = success_take + fail_take
            remaining = int(online_batch_size - sampled_online)
            if remaining > 0 and len(online_rb) >= remaining:
                online_parts.append(_move_batch_to_device(online_rb.sample(remaining), device))
                sampled_online += remaining

            if sampled_online > 0:
                used_online = sampled_online
        elif len(online_rb) >= online_batch_size:
            online_parts.append(_move_batch_to_device(online_rb.sample(online_batch_size), device))
            used_online = online_batch_size

    parts.extend(online_parts)
    if offline_rb is not None and offline_batch_size > 0 and len(offline_rb) >= offline_batch_size:
        parts.append(_move_batch_to_device(offline_rb.sample(offline_batch_size), device))
        used_offline = offline_batch_size

    # Top-up to keep effective batch size close to target when one source is scarce.
    target_total = max(int(fallback_batch_size), 1)
    sampled_total = int(used_online + used_offline)
    if parts and sampled_total < target_total:
        remaining = target_total - sampled_total
        if offline_rb is not None and len(offline_rb) >= remaining:
            parts.append(_move_batch_to_device(offline_rb.sample(remaining), device))
            used_offline += remaining
            sampled_total += remaining
        elif len(online_rb) >= remaining:
            parts.append(_move_batch_to_device(online_rb.sample(remaining), device))
            used_online += remaining
            sampled_total += remaining

    if not parts:
        if len(online_rb) >= fallback_batch_size:
            parts.append(_move_batch_to_device(online_rb.sample(fallback_batch_size), device))
            used_online = fallback_batch_size
        elif offline_rb is not None and len(offline_rb) >= fallback_batch_size:
            parts.append(_move_batch_to_device(offline_rb.sample(fallback_batch_size), device))
            used_offline = fallback_batch_size

    if not parts:
        return None, 0, 0
    return _concat_batches(parts), used_online, used_offline


def _build_resfit_qagent(
    args: argparse.Namespace,
    device: str,
    action_dim: int,
    prop_dim: int,
    obs_shape: tuple[int, int, int],
):
    _ensure_resfit_rlpd_compat()
    _ensure_resfit_encoder_compat()
    _ensure_resfit_actor_compat(Path(args.resfit_root))
    _ensure_resfit_critic_compat(Path(args.resfit_root))
    _ensure_resfit_qagent_compat(Path(args.resfit_root))
    from resfit.rl_finetuning.config.rlpd import (
        ActorConfig,
        CriticConfig,
        QAgentConfig,
        VitEncoderConfig,
    )
    from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent

    qcfg = QAgentConfig(
        device=device,
        actor_lr=float(args.actor_lr),
        critic_lr=float(args.critic_lr),
        critic_target_tau=float(args.critic_target_tau),
        use_prop=1,
        enc_type="vit",
        vit=VitEncoderConfig(
            depth=1,
            embed_dim=128,
            embed_norm=0,
            embed_style="embed2",
            num_heads=4,
            patch_size=8,
            stride=-1,
        ),
        critic=CriticConfig(
            hidden_dim=int(args.critic_hidden_dim),
            num_q=int(args.num_q_heads),
            policy_gradient_type=str(args.policy_gradient_type),
        ),
        actor=ActorConfig(
            hidden_dim=int(args.actor_hidden_dim),
            action_scale=float(args.residual_action_scale),
        ),
        freeze_encoder=bool(args.freeze_encoder),
    )

    return QAgent(
        obs_shape=tuple(int(x) for x in obs_shape),
        prop_shape=(int(prop_dim),),
        action_dim=int(action_dim),
        rl_cameras=[str(args.camera_key)],
        cfg=qcfg,
        residual_actor=True,
    )


def _evaluate(
    *,
    eval_env,
    eval_base: FrozenPointPolicyBaseRL,
    q_agent,
    resfit_utils,
    action_normalizer: LinearActionNormalizerRL,
    pixel_key: str,
    include_eef_pos: bool,
    state_coord_mode: str,
    residual_obs_mode: str,
    camera_key: str,
    dummy_image_size: int,
    image_obs_key: str,
    image_hw: tuple[int, int] | None,
    image_only_prop_dim: int,
    episodes: int,
    device: str,
    video_dir: Path | None = None,
    video_fps: int = 20,
    video_render_size: int = 256,
    video_max_episodes: int = 0,
    video_tag: str = "",
    video_phase: str = "online",
    step: int = 0,
    online_episode_idx: int = -1,
    residual_gripper_mode: str = "full",
    residual_gripper_scale: float = 1.0,
) -> dict[str, float]:
    returns = []
    successes = []
    saved_videos = 0

    for eval_ep_idx in range(episodes):
        capture_video = video_dir is not None and eval_ep_idx < max(int(video_max_episodes), 0)
        frames: list[np.ndarray] = []
        ts = eval_env.reset()
        obs = ts.observation
        eval_base.reset_episode()
        step_in_ep = 0
        done = False
        ep_ret = 0.0
        success = 0.0

        if capture_video:
            try:
                frames.append(_capture_frame(eval_env, render_size=int(video_render_size)))
            except Exception as exc:
                print(f"[eval-video] capture disabled (init): {exc}")
                capture_video = False

        while not done:
            base_action_dict = eval_base.act(obs, step_in_ep, step_in_ep)
            base_action_raw = np.asarray(eval_env.point2action(base_action_dict), dtype=np.float32).reshape(7)
            state_vec = observation_to_state(
                obs=obs,
                pixel_key=pixel_key,
                base_action_7d=base_action_raw,
                include_eef_pos=include_eef_pos,
                state_coord_mode=state_coord_mode,
            )
            agent_obs = _build_agent_obs_batched(
                residual_obs_mode=residual_obs_mode,
                state_vec=state_vec,
                obs_raw=obs,
                base_action_raw=base_action_raw,
                action_normalizer=action_normalizer,
                camera_key=camera_key,
                dummy_image_size=dummy_image_size,
                pixel_key=pixel_key,
                image_obs_key=image_obs_key,
                image_hw=image_hw,
                image_only_prop_dim=image_only_prop_dim,
                device=device,
            )
            with resfit_utils.eval_mode(q_agent):
                residual_norm = (
                    q_agent.act(agent_obs, eval_mode=True, stddev=0.0, cpu=True)
                    .squeeze(0)
                    .numpy()
                    .astype(np.float32)
                )
            residual_norm = _apply_residual_gripper_mode(
                residual_norm,
                mode=residual_gripper_mode,
                scale=float(residual_gripper_scale),
            )

            base_norm = action_normalizer.normalize(base_action_raw)
            combined_norm = np.clip(base_norm + residual_norm, -1.0, 1.0).astype(np.float32)
            env_action_raw = action_normalizer.denormalize(combined_norm)
            ts = eval_env.step(env_action_raw)
            obs = ts.observation
            done = bool(ts.last())
            ep_ret += float(ts.reward)
            success = 1.0 if bool(obs.get("goal_achieved", False)) else success
            step_in_ep += 1
            if capture_video:
                try:
                    frames.append(_capture_frame(eval_env, render_size=int(video_render_size)))
                except Exception as exc:
                    print(f"[eval-video] capture disabled (step): {exc}")
                    capture_video = False

        if capture_video and frames:
            tag = _slugify(video_tag)
            phase = _slugify(video_phase) or "online"
            status = "success" if int(success) > 0 else "fail"
            name_parts = [
                phase,
                f"step{int(step):07d}",
                f"train_ep{int(online_episode_idx):06d}" if int(online_episode_idx) >= 0 else "train_epNA",
                f"eval_ep{int(eval_ep_idx):02d}",
                status,
            ]
            if tag:
                name_parts.insert(0, tag)
            video_path = _dedup_path(video_dir / ("__".join(name_parts) + ".mp4"))
            try:
                if imageio is None:
                    raise RuntimeError("imageio is not available")
                imageio.mimsave(str(video_path), frames, fps=max(1, int(video_fps)))
                saved_videos += 1
                print(f"[eval-video] saved: {video_path}")
            except Exception as exc:
                print(f"[eval-video] save failed: {exc}")

        returns.append(ep_ret)
        successes.append(success)

    return {
        "eval/episode_return": float(np.mean(returns)) if returns else 0.0,
        "eval/success": float(np.mean(successes)) if successes else 0.0,
        "eval/videos_saved": int(saved_videos),
    }


def _collect_offline_transitions_from_base_rollout(
    *,
    env,
    base: FrozenPointPolicyBaseRL,
    pixel_key: str,
    include_eef_pos: bool,
    state_coord_mode: str,
    low: np.ndarray,
    high: np.ndarray,
    target_episodes: int,
    success_only: bool,
    terminal_reward: float,
    step_reward: float,
    max_episode_len: int,
    max_attempt_mult: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    target_episodes = max(1, int(target_episodes))
    max_attempts = max(target_episodes * max(1, int(max_attempt_mult)), target_episodes)

    transitions: list[dict[str, Any]] = []
    accepted_episodes = 0
    success_episodes = 0
    attempted_episodes = 0

    while accepted_episodes < target_episodes and attempted_episodes < max_attempts:
        ts = env.reset()
        obs = ts.observation
        base.reset_episode()
        step_in_ep = 0
        episode_transitions: list[dict[str, Any]] = []
        episode_success = False
        done = False

        base_action_dict = base.act(obs, step_in_ep, step_in_ep)
        while not done and step_in_ep < int(max_episode_len):
            base_action_raw = np.asarray(env.point2action(base_action_dict), dtype=np.float32).reshape(7)
            state_vec = observation_to_state(
                obs=obs,
                pixel_key=pixel_key,
                base_action_7d=base_action_raw,
                include_eef_pos=include_eef_pos,
                state_coord_mode=state_coord_mode,
            )

            forced_terminal_due_env_error = False
            try:
                ts_next = env.step(clip_action(base_action_raw, low=low, high=high))
                next_obs = ts_next.observation
                done = bool(ts_next.last())
            except ValueError as exc:
                if "terminated episode" not in str(exc):
                    raise
                forced_terminal_due_env_error = True
                next_obs = obs
                done = True
                print(f"[offline-rollout] forced terminal transition due to env step error: {exc}")

            goal_achieved = bool(next_obs.get("goal_achieved", False))
            episode_success = episode_success or goal_achieved
            reward = float(terminal_reward if (done and goal_achieved) else step_reward)

            if done:
                next_base_action_raw = np.zeros((7,), dtype=np.float32)
                next_base_action_dict = None
            else:
                next_base_action_dict = base.act(next_obs, step_in_ep + 1, step_in_ep + 1)
                next_base_action_raw = np.asarray(
                    env.point2action(next_base_action_dict), dtype=np.float32
                ).reshape(7)

            next_state_vec = observation_to_state(
                obs=next_obs,
                pixel_key=pixel_key,
                base_action_7d=next_base_action_raw,
                include_eef_pos=include_eef_pos,
                state_coord_mode=state_coord_mode,
            )
            episode_transitions.append(
                {
                    "obs": state_vec,
                    "action": base_action_raw.copy(),
                    "reward": reward,
                    "next_obs": next_state_vec,
                    "done": bool(done),
                    "base_action": base_action_raw.copy(),
                    "next_base_action": next_base_action_raw.copy(),
                    "obs_raw": obs,
                    "next_obs_raw": next_obs,
                }
            )

            if done:
                break
            obs = next_obs
            base_action_dict = next_base_action_dict
            step_in_ep += 1
            if forced_terminal_due_env_error:
                break

        attempted_episodes += 1
        keep_episode = episode_success if success_only else True
        if keep_episode and episode_transitions:
            transitions.extend(episode_transitions)
            accepted_episodes += 1
            if episode_success:
                success_episodes += 1
        if attempted_episodes % 10 == 0 or accepted_episodes == target_episodes:
            print(
                "[offline-rollout] progress "
                f"accepted={accepted_episodes}/{target_episodes} "
                f"attempted={attempted_episodes}/{max_attempts} "
                f"success_eps={success_episodes}"
            )

    stats = {
        "attempted_episodes": int(attempted_episodes),
        "accepted_episodes": int(accepted_episodes),
        "success_episodes": int(success_episodes),
        "target_episodes": int(target_episodes),
        "max_attempts": int(max_attempts),
        "transitions": int(len(transitions)),
    }
    if accepted_episodes < target_episodes:
        print(
            "[offline-rollout] warning: collected fewer episodes than requested "
            f"(accepted={accepted_episodes}, target={target_episodes})."
        )
    return transitions, stats


def main() -> None:
    args = parse_args()
    _assert_strict_resfit_args(args)
    repo_root = Path(__file__).resolve().parents[1]
    output_root = (repo_root / args.output_root).resolve()
    device = coerce_device(args.device)
    set_seed(args.seed)
    _ensure_local_bddl(repo_root)
    _ensure_gym_compat()
    _ensure_torch_attention_compat()
    _patch_torch_load_weights_only_default()
    _ensure_local_robomimic(repo_root)
    _ensure_tabulate_compat()

    resfit_root = Path(args.resfit_root).expanduser().resolve()
    if not (resfit_root / "resfit").exists():
        raise FileNotFoundError(f"Invalid resfit root (missing 'resfit/' package): {resfit_root}")
    if str(resfit_root) not in sys.path:
        sys.path.insert(0, str(resfit_root))

    import torch

    torchrl_available = True
    torchrl_import_error = ""
    try:
        from torchrl.data import (
            LazyTensorStorage,
            TensorDictPrioritizedReplayBuffer,
            TensorDictReplayBuffer,
        )
        from resfit.rl_finetuning.utils.rb_transforms import MultiStepTransform
    except Exception as exc:
        torchrl_available = False
        torchrl_import_error = f"{type(exc).__name__}: {exc}"
        LazyTensorStorage = None  # type: ignore[assignment]
        TensorDictPrioritizedReplayBuffer = None  # type: ignore[assignment]
        TensorDictReplayBuffer = None  # type: ignore[assignment]
        MultiStepTransform = None  # type: ignore[assignment]
        print(
            "[replay] torchrl import failed; using simple uniform replay fallback "
            f"(n-step enabled, no prioritized replay). reason={torchrl_import_error}"
        )

    resfit_utils = _ensure_resfit_common_utils_compat(resfit_root)

    train_base = FrozenPointPolicyBaseRL(
        repo_root=repo_root,
        bc_weight=Path(args.bc_weight),
        device=device,
    )
    eval_base = FrozenPointPolicyBaseRL(
        repo_root=repo_root,
        bc_weight=Path(args.bc_weight),
        device=device,
    )

    suite_override_name = str(args.suite or "").strip()
    if _is_basefix_v1_suite_name(suite_override_name):
        changed_total: set[str] = set()
        for base_cfg in (train_base.cfg, eval_base.cfg):
            if isinstance(base_cfg, dict):
                changed_total.update(_apply_basefix_v1_suite_preset(base_cfg, suite_override_name))
        changed_keys = ",".join(sorted(changed_total)) if changed_total else "none"
        print(
            "[suite-override] applied basefix_v1 preset "
            f"suite={suite_override_name} changed={changed_keys}"
        )

    if args.suite_gripper_close_position_gate.strip() != "":
        gate_value = _parse_bool_text(args.suite_gripper_close_position_gate)
        for base_cfg in (train_base.cfg, eval_base.cfg):
            if isinstance(base_cfg, dict):
                suite_cfg = base_cfg.get("suite")
                if isinstance(suite_cfg, dict):
                    suite_cfg["gripper_close_position_gate"] = gate_value
        print(f"[suite-override] gripper_close_position_gate={gate_value}")
    if args.suite_gripper_cmd_slew_rate is not None:
        slew_value = float(args.suite_gripper_cmd_slew_rate)
        for base_cfg in (train_base.cfg, eval_base.cfg):
            if isinstance(base_cfg, dict):
                suite_cfg = base_cfg.get("suite")
                if isinstance(suite_cfg, dict):
                    suite_cfg["gripper_cmd_slew_rate"] = slew_value
        print(f"[suite-override] gripper_cmd_slew_rate={slew_value}")

    benchmark_override = str(args.benchmark_name).strip()
    task_order_override = int(args.task_order_index)
    if benchmark_override or task_order_override >= 0:
        for base_cfg in (train_base.cfg, eval_base.cfg):
            if not isinstance(base_cfg, dict):
                continue
            suite_cfg = base_cfg.get("suite")
            if not isinstance(suite_cfg, dict):
                continue
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

    cfg_suite = train_base.cfg.get("suite", {}) if isinstance(train_base.cfg, dict) else {}
    cfg_task = cfg_suite.get("task", {}) if isinstance(cfg_suite, dict) else {}
    task_name = str(args.task_name or cfg_task.get("task_name") or "unknown_task")
    suite_for_dir = str(args.suite or cfg_suite.get("name") or cfg_suite.get("suite") or "unknown_suite")
    run_name = args.run_name or "default"
    run_dir = build_run_dir(output_root=output_root, suite_name=suite_for_dir, run_name=run_name)
    ckpt_dir = run_dir / "snapshot"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    wandb_mod, wandb_run = _init_wandb(
        args=args,
        run_dir=run_dir,
        run_meta={
            "backend": "resfit_qagent",
            "phase": "bootstrap",
            "resfit_root": str(resfit_root),
            "bc_weight": str(Path(args.bc_weight).resolve()),
            "suite": suite_for_dir,
            "task_name": task_name,
            "args": vars(args),
        },
    )

    train_env, task_desc, pixel_key, low, high, suite_name = build_single_env_from_bc_config(
        cfg=train_base.cfg,
        repo_root=repo_root,
        suite_override=args.suite,
        task_override=args.task_name,
        seed=args.seed,
        eval_mode=False,
        max_episode_len=args.env_max_episode_len,
    )
    eval_env, _, _, _, _, _ = build_single_env_from_bc_config(
        cfg=eval_base.cfg,
        repo_root=repo_root,
        suite_override=args.suite,
        task_override=args.task_name,
        seed=args.seed + 123,
        eval_mode=True,
        max_episode_len=args.env_max_episode_len,
    )

    train_base.build(
        obs_spec=train_env.observation_spec(),
        action_spec=train_env.action_spec(),
        max_episode_len=args.env_max_episode_len,
    )
    eval_base.build(
        obs_spec=eval_env.observation_spec(),
        action_spec=eval_env.action_spec(),
        max_episode_len=args.env_max_episode_len,
    )

    action_normalizer = LinearActionNormalizerRL(low=low, high=high)

    ts = train_env.reset()
    obs = ts.observation
    train_base.reset_episode()
    step_in_ep = 0
    base_action_dict = train_base.act(obs, step_in_ep, 0)
    base_action_raw = np.asarray(train_env.point2action(base_action_dict), dtype=np.float32).reshape(7)
    state_vec = observation_to_state(
        obs=obs,
        pixel_key=pixel_key,
        base_action_7d=base_action_raw,
        include_eef_pos=args.include_eef_pos,
        state_coord_mode=args.state_coord_mode,
    )
    residual_obs_mode = str(args.residual_obs_mode).strip().lower()
    image_obs_key = str(args.image_obs_key).strip()
    image_hw: tuple[int, int] | None = None
    image_key_used = ""
    if residual_obs_mode == "image_only":
        # ResFiT MinViT in this codebase is configured for fixed patch count (81),
        # so image input must be resized to a fixed square size (default: 84x84).
        fixed_hw = (int(args.dummy_image_size), int(args.dummy_image_size))
        sample_img, image_key_used = _extract_image_from_obs(
            obs=obs,
            pixel_key=pixel_key,
            image_obs_key=image_obs_key,
            target_hw=fixed_hw,
        )
        obs_shape = tuple(int(x) for x in sample_img.shape)
        image_hw = fixed_hw
        prop_dim = 1
        print(
            "[obs-mode] image_only enabled "
            f"image_key={image_key_used} obs_shape={obs_shape} "
            f"fixed_hw={image_hw} prop_dim={prop_dim}"
        )
    else:
        prop_state, _ = _split_state_vec(state_vec)
        prop_dim = int(prop_state.shape[0])
        obs_shape = (3, int(args.dummy_image_size), int(args.dummy_image_size))
        print(
            "[obs-mode] point_state_dummy_image "
            f"obs_shape={obs_shape} prop_dim={prop_dim}"
        )

    q_agent = _build_resfit_qagent(
        args=args,
        device=device,
        action_dim=7,
        prop_dim=prop_dim,
        obs_shape=obs_shape,
    )

    requested_sampling_strategy = str(args.sampling_strategy)
    effective_sampling_strategy = requested_sampling_strategy
    alpha = float(args.priority_alpha) if requested_sampling_strategy == "prioritized_replay" else 0.0
    beta = float(args.priority_beta) if requested_sampling_strategy == "prioritized_replay" else 0.0
    if not torchrl_available:
        if requested_sampling_strategy != "uniform":
            print(
                "[replay] forcing sampling_strategy=uniform because torchrl is unavailable "
                f"({torchrl_import_error})"
            )
        effective_sampling_strategy = "uniform"

    requested_offline_fraction = float(args.offline_fraction)
    requested_offline_fraction_start = (
        None if args.offline_fraction_start is None else float(args.offline_fraction_start)
    )
    requested_offline_fraction_end = (
        None if args.offline_fraction_end is None else float(args.offline_fraction_end)
    )
    if (requested_offline_fraction_start is None) != (requested_offline_fraction_end is None):
        raise ValueError(
            "offline fraction schedule requires both "
            "--offline-fraction-start and --offline-fraction-end."
        )
    offline_fraction_schedule_enabled = (
        requested_offline_fraction_start is not None and requested_offline_fraction_end is not None
    )
    offline_fraction_decay_steps = max(int(args.offline_fraction_decay_steps), 1)
    if offline_fraction_schedule_enabled:
        offline_fraction_start = float(np.clip(requested_offline_fraction_start, 0.0, 1.0))
        offline_fraction_end = float(np.clip(requested_offline_fraction_end, 0.0, 1.0))
        offline_fraction_bootstrap = max(offline_fraction_start, offline_fraction_end)
    else:
        offline_fraction_start = float(np.clip(requested_offline_fraction, 0.0, 1.0))
        offline_fraction_end = offline_fraction_start
        offline_fraction_bootstrap = offline_fraction_start

    offline_fraction = float(offline_fraction_bootstrap)
    offline_batch_size = int(args.batch_size * offline_fraction)
    if offline_fraction > 0.0 and offline_batch_size == 0:
        offline_batch_size = 1
    online_batch_size = int(args.batch_size - offline_batch_size)

    def _build_replay_buffer(max_size: int, batch_size: int):
        nonlocal effective_sampling_strategy
        if not torchrl_available:
            return _SimpleReplayBufferRL(
                max_size=int(max_size),
                batch_size=max(int(batch_size), 1),
                gamma=float(args.gamma),
                n_step=int(args.n_step),
            )

        common_kwargs = {
            "storage": LazyTensorStorage(max_size=int(max_size), device="cpu"),
            "transform": MultiStepTransform(n_steps=int(args.n_step), gamma=float(args.gamma)),
            "batch_size": max(int(batch_size), 1),
            "pin_memory": True,
            "prefetch": int(args.prefetch_batches),
        }
        if effective_sampling_strategy != "prioritized_replay":
            return TensorDictReplayBuffer(**common_kwargs)

        try:
            return TensorDictPrioritizedReplayBuffer(
                alpha=alpha,
                beta=beta,
                eps=1e-6,
                priority_key="_priority",
                **common_kwargs,
            )
        except Exception as exc:
            effective_sampling_strategy = "uniform"
            print(
                "[replay] prioritized replay init failed; falling back to uniform replay. "
                f"reason={type(exc).__name__}: {exc}"
            )
            return TensorDictReplayBuffer(**common_kwargs)

    online_rb = _build_replay_buffer(
        max_size=int(args.buffer_size),
        batch_size=max(online_batch_size, 1),
    )
    online_bucket_sampling = bool(args.online_bucket_sampling)
    online_success_batch_fraction = float(np.clip(args.online_success_batch_fraction, 0.0, 1.0))
    online_fail_max_steps = max(int(args.online_fail_max_steps), 0)
    online_fail_stride = max(int(args.online_fail_stride), 1)

    online_success_rb = None
    online_fail_rb = None
    if online_bucket_sampling:
        online_success_rb = _SimpleReplayBufferRL(
            max_size=int(args.buffer_size),
            batch_size=max(online_batch_size, 1),
            gamma=float(args.gamma),
            n_step=1,
        )
        online_fail_rb = _SimpleReplayBufferRL(
            max_size=int(args.buffer_size),
            batch_size=max(online_batch_size, 1),
            gamma=float(args.gamma),
            n_step=1,
        )
        print(
            "[online-bucket] enabled "
            f"success_batch_fraction={online_success_batch_fraction} "
            f"fail_max_steps={online_fail_max_steps} fail_stride={online_fail_stride}"
        )

    offline_rb = None
    offline_transition_count = 0
    offline_source = str(args.offline_source).strip().lower()
    offline_demo_root: Path | None = None
    offline_rollout_stats: dict[str, int] = {}

    if offline_fraction_bootstrap > 0.0:
        if offline_source == "base_rollout":
            target_rollout_episodes = int(args.offline_rollout_episodes)
            if target_rollout_episodes <= 0:
                target_rollout_episodes = int(args.offline_max_demos)
            if target_rollout_episodes <= 0:
                target_rollout_episodes = 50

            collect_env, _, collect_pixel_key, _, _, _ = build_single_env_from_bc_config(
                cfg=train_base.cfg,
                repo_root=repo_root,
                suite_override=args.suite,
                task_override=args.task_name,
                seed=args.seed + 777,
                eval_mode=False,
                max_episode_len=args.env_max_episode_len,
            )
            if str(collect_pixel_key) != str(pixel_key):
                print(
                    "[offline-rollout] pixel key mismatch "
                    f"train={pixel_key} collect={collect_pixel_key}; using collect key"
                )

            collect_base = FrozenPointPolicyBaseRL(
                repo_root=repo_root,
                bc_weight=Path(args.bc_weight),
                device=device,
            )
            collect_base.build(
                obs_spec=collect_env.observation_spec(),
                action_spec=collect_env.action_spec(),
                max_episode_len=args.env_max_episode_len,
            )
            offline_transitions, offline_rollout_stats = _collect_offline_transitions_from_base_rollout(
                env=collect_env,
                base=collect_base,
                pixel_key=collect_pixel_key,
                include_eef_pos=args.include_eef_pos,
                state_coord_mode=args.state_coord_mode,
                low=low,
                high=high,
                target_episodes=target_rollout_episodes,
                success_only=bool(args.offline_success_only),
                terminal_reward=float(args.offline_terminal_reward),
                step_reward=float(args.offline_step_reward),
                max_episode_len=int(args.env_max_episode_len),
                max_attempt_mult=int(args.offline_rollout_max_attempt_mult),
            )
            try:
                if hasattr(collect_env, "close"):
                    collect_env.close()
            except Exception:
                pass
        else:
            if args.offline_demo_root.strip():
                offline_demo_root = Path(args.offline_demo_root).expanduser().resolve()
            else:
                offline_demo_root = (repo_root / "expert_demos" / suite_name).resolve()

            max_demos = args.offline_max_demos if args.offline_max_demos > 0 else None
            bc_action_fn = None
            bc_reset_episode_fn = None
            if args.offline_base_action_mode == "bc_track_delta":
                offline_base = FrozenPointPolicyBaseRL(
                    repo_root=repo_root,
                    bc_weight=Path(args.bc_weight),
                    device=device,
                )
                offline_base.build(
                    obs_spec=train_env.observation_spec(),
                    action_spec=train_env.action_spec(),
                    max_episode_len=args.env_max_episode_len,
                )

                def _bc_reset_episode() -> None:
                    offline_base.reset_episode()

                def _bc_action(obs_step: dict[str, Any], step_idx: int) -> dict[str, Any]:
                    return offline_base.act(obs_step, step_idx, step_idx)

                bc_action_fn = _bc_action
                bc_reset_episode_fn = _bc_reset_episode

            offline_transitions = build_offline_transitions_from_expert_demos_rl(
                demo_root=offline_demo_root,
                suite_name=suite_name,
                task_name=task_name,
                pixel_key=pixel_key,
                include_eef_pos=args.include_eef_pos,
                low=low,
                high=high,
                state_coord_mode=args.state_coord_mode,
                max_demos=max_demos,
                terminal_reward=args.offline_terminal_reward,
                step_reward=args.offline_step_reward,
                base_action_mode=args.offline_base_action_mode,
                transition_action_mode="combined_base",
                bc_action_fn=bc_action_fn,
                bc_reset_episode_fn=bc_reset_episode_fn,
            )

        if residual_obs_mode == "image_only" and offline_transitions:
            sample_obs_raw = None
            for tr in offline_transitions:
                cur = tr.get("obs_raw")
                if isinstance(cur, dict):
                    sample_obs_raw = cur
                    break
            if sample_obs_raw is None:
                raise ValueError(
                    "image_only mode requires raw observation dicts in offline transitions, "
                    "but obs_raw is missing."
                )
            try:
                _sample_img, offline_image_key_used = _extract_image_from_obs(
                    obs=sample_obs_raw,
                    pixel_key=pixel_key,
                    image_obs_key=image_obs_key,
                    target_hw=image_hw,
                )
                if not image_key_used:
                    image_key_used = str(offline_image_key_used)
            except Exception as exc:
                raise ValueError(
                    "image_only mode requires image observations in offline data. "
                    f"offline_source={offline_source} did not provide usable image keys "
                    f"(requested={image_obs_key or '<auto>'}, pixel_key={pixel_key})."
                ) from exc

        if offline_transitions:
            offline_rb = _build_replay_buffer(
                max_size=max(len(offline_transitions), 1),
                batch_size=max(offline_batch_size, 1),
            )
            for tr in offline_transitions:
                _add_transition_to_rb(
                    offline_rb,
                    state_vec=tr["obs"],
                    next_state_vec=tr["next_obs"],
                    obs_raw=tr.get("obs_raw"),
                    next_obs_raw=tr.get("next_obs_raw"),
                    base_action_raw=tr.get("base_action"),
                    next_base_action_raw=tr.get("next_base_action"),
                    combined_action_raw=tr["action"],
                    reward=float(tr["reward"]),
                    done=bool(tr["done"]),
                    action_normalizer=action_normalizer,
                    residual_obs_mode=residual_obs_mode,
                    camera_key=args.camera_key,
                    dummy_image_size=args.dummy_image_size,
                    pixel_key=pixel_key,
                    image_obs_key=image_obs_key,
                    image_hw=image_hw,
                    image_only_prop_dim=prop_dim,
                )
            offline_transition_count = len(offline_rb)
        else:
            offline_fraction = 0.0
            offline_batch_size = 0
            online_batch_size = int(args.batch_size)

    if bool(args.strict_resfit):
        if offline_fraction <= 0.0 or offline_rb is None or len(offline_rb) <= 0:
            raise ValueError(
                "[strict-resfit] offline buffer is empty after initialization; "
                "strict mode requires effective offline_fraction > 0 with valid offline transitions."
            )
        if abs(float(offline_fraction) - 0.5) > 1e-12:
            raise ValueError(
                "[strict-resfit] effective offline_fraction drifted from 0.5 "
                f"(effective={offline_fraction})"
            )

    run_meta = {
        "backend": "resfit_qagent",
        "resfit_root": str(resfit_root),
        "bc_weight": str(Path(args.bc_weight).resolve()),
        "suite": suite_name,
        "task_desc": task_desc,
        "task_name": task_name,
        "pixel_key": pixel_key,
        "state_coord_mode": str(args.state_coord_mode),
        "residual_obs": {
            "mode": residual_obs_mode,
            "camera_key": str(args.camera_key),
            "image_obs_key_requested": image_obs_key,
            "image_obs_key_used": image_key_used,
            "image_hw": list(image_hw) if image_hw is not None else None,
            "obs_shape": list(obs_shape),
            "prop_dim": int(prop_dim),
        },
        "device": device,
        "low": low.tolist(),
        "high": high.tolist(),
        "offline": {
            "fraction_requested": requested_offline_fraction,
            "fraction_applied": offline_fraction,
            "fraction_schedule_enabled": bool(offline_fraction_schedule_enabled),
            "fraction_schedule_start": float(offline_fraction_start),
            "fraction_schedule_end": float(offline_fraction_end),
            "fraction_schedule_decay_steps": int(offline_fraction_decay_steps),
            "source": offline_source,
            "demo_root": str(offline_demo_root) if offline_demo_root is not None else "",
            "base_action_mode": args.offline_base_action_mode,
            "transitions": int(offline_transition_count),
            "offline_batch_size": int(offline_batch_size),
            "online_batch_size": int(online_batch_size),
            "success_only": bool(args.offline_success_only),
            "rollout_stats": offline_rollout_stats,
        },
        "online_bucket": {
            "enabled": bool(online_bucket_sampling),
            "success_batch_fraction": float(online_success_batch_fraction),
            "fail_max_steps": int(online_fail_max_steps),
            "fail_stride": int(online_fail_stride),
        },
        "replay": {
            "sampling_strategy_requested": requested_sampling_strategy,
            "sampling_strategy_effective": effective_sampling_strategy,
            "priority_alpha": float(alpha),
            "priority_beta": float(beta),
        },
        "args": vars(args),
    }
    dump_json(run_dir / "run_meta.json", run_meta)
    if wandb_run is not None:
        try:
            wandb_run.config.update(run_meta, allow_val_change=True)
        except Exception as exc:
            print(f"[wandb] config update failed: {exc}")

    stddev_schedule = f"linear({args.stddev_max},{args.stddev_min},{args.stddev_step})"
    update_every_n_steps = max(1, int(args.update_every_n_steps))
    num_updates_per_iteration = max(1, int(args.num_updates_per_iteration))
    actor_updates_per_iteration = max(1, int(args.actor_updates_per_iteration))
    actor_update_cadence = max(1, num_updates_per_iteration // actor_updates_per_iteration)

    train_csv = run_dir / "train_log.csv"
    eval_csv = run_dir / "eval_log.csv"
    eval_video_enabled = bool(args.eval_save_video)
    eval_video_dir: Path | None = None
    eval_video_every_ep = max(1, int(args.eval_video_online_every_episodes))
    last_online_video_episode = -1
    if eval_video_enabled:
        if imageio is None:
            print("[eval-video] disabled because imageio is not available")
            eval_video_enabled = False
        else:
            if str(args.eval_video_dir).strip():
                eval_video_dir = Path(args.eval_video_dir).expanduser().resolve()
            else:
                eval_video_dir = (run_dir / "eval_videos_rl").resolve()
            eval_video_dir.mkdir(parents=True, exist_ok=True)
            print(
                "[eval-video] enabled "
                f"dir={eval_video_dir} fps={int(args.eval_video_fps)} "
                f"render_size={int(args.eval_video_render_size)} "
                f"online_every_episodes={eval_video_every_ep}"
            )

    def _offline_fraction_at_step(step: int) -> float:
        if offline_rb is None or len(offline_rb) <= 0:
            return 0.0
        if not offline_fraction_schedule_enabled:
            return float(offline_fraction)

        t = float(np.clip(step, 0, offline_fraction_decay_steps)) / float(offline_fraction_decay_steps)
        frac = float(offline_fraction_start + (offline_fraction_end - offline_fraction_start) * t)
        return float(np.clip(frac, 0.0, 1.0))

    episode_return = 0.0
    episode_len = 0
    episode_idx = 0
    success_flag = 0
    episode_transition_cache: list[dict[str, Any]] = []
    cached_base_action_dict = base_action_dict
    diag_sums = {
        "clip_rate_dim": 0.0,
        "clip_any": 0.0,
        "residual_induced_clip_rate_dim": 0.0,
        "base_norm_l2": 0.0,
        "residual_norm_l2": 0.0,
        "res_over_base": 0.0,
        "clip_delta_l1_mean": 0.0,
        "clip_delta_linf": 0.0,
        "exec_clip_raw_delta": 0.0,
    }
    diag_exec_clip_raw_delta_max = 0.0
    diag_count = 0

    # Offline phase preview: save exactly one eval video before online rollout starts.
    if eval_video_enabled and eval_video_dir is not None:
        offline_eval_metrics = _evaluate(
            eval_env=eval_env,
            eval_base=eval_base,
            q_agent=q_agent,
            resfit_utils=resfit_utils,
            action_normalizer=action_normalizer,
            pixel_key=pixel_key,
            include_eef_pos=args.include_eef_pos,
            state_coord_mode=args.state_coord_mode,
            residual_obs_mode=residual_obs_mode,
            camera_key=args.camera_key,
            dummy_image_size=args.dummy_image_size,
            image_obs_key=image_obs_key,
            image_hw=image_hw,
            image_only_prop_dim=prop_dim,
            episodes=int(args.eval_episodes),
            device=device,
            video_dir=eval_video_dir,
            video_fps=int(args.eval_video_fps),
            video_render_size=int(args.eval_video_render_size),
            video_max_episodes=1,
            video_tag=args.eval_video_tag,
            video_phase="offline",
            step=0,
            online_episode_idx=-1,
            residual_gripper_mode=args.residual_gripper_mode,
            residual_gripper_scale=float(args.residual_gripper_scale),
        )
        offline_eval_row = {"step": 0, **offline_eval_metrics}
        _write_csv_row(eval_csv, offline_eval_row)
        print(json.dumps(offline_eval_row))
        if wandb_mod is not None:
            wandb_mod.log({k: v for k, v in offline_eval_row.items() if k != "step"}, step=0)

    for global_step in range(1, int(args.steps) + 1):
        base_action_dict = cached_base_action_dict
        base_action_raw = np.asarray(train_env.point2action(base_action_dict), dtype=np.float32).reshape(7)
        state_vec = observation_to_state(
            obs=obs,
            pixel_key=pixel_key,
            base_action_7d=base_action_raw,
            include_eef_pos=args.include_eef_pos,
            state_coord_mode=args.state_coord_mode,
        )

        base_action_norm = action_normalizer.normalize(base_action_raw)
        if global_step <= int(args.warmup_steps):
            residual_norm = np.random.uniform(
                low=-float(args.random_action_noise_scale),
                high=float(args.random_action_noise_scale),
                size=(7,),
            ).astype(np.float32)
        else:
            stddev = float(resfit_utils.schedule(stddev_schedule, global_step))
            agent_obs = _build_agent_obs_batched(
                residual_obs_mode=residual_obs_mode,
                state_vec=state_vec,
                obs_raw=obs,
                base_action_raw=base_action_raw,
                action_normalizer=action_normalizer,
                camera_key=args.camera_key,
                dummy_image_size=args.dummy_image_size,
                pixel_key=pixel_key,
                image_obs_key=image_obs_key,
                image_hw=image_hw,
                image_only_prop_dim=prop_dim,
                device=device,
            )
            with resfit_utils.eval_mode(q_agent):
                residual_norm = (
                    q_agent.act(agent_obs, eval_mode=False, stddev=stddev, cpu=True)
                    .squeeze(0)
                    .numpy()
                    .astype(np.float32)
                )
        residual_norm = _apply_residual_gripper_mode(
            residual_norm,
            mode=args.residual_gripper_mode,
            scale=float(args.residual_gripper_scale),
        )

        combined_norm = np.clip(base_action_norm + residual_norm, -1.0, 1.0).astype(np.float32)
        env_action_raw = action_normalizer.denormalize(combined_norm)
        action_diag = _action_diag(base_action_norm, residual_norm)
        env_action_exec = clip_action(env_action_raw, low=low, high=high)
        exec_clip_raw_delta = float(np.max(np.abs(env_action_exec - env_action_raw)))
        action_diag["exec_clip_raw_delta"] = exec_clip_raw_delta
        for _k, _v in action_diag.items():
            if _k in diag_sums:
                diag_sums[_k] += float(_v)
        diag_exec_clip_raw_delta_max = max(diag_exec_clip_raw_delta_max, exec_clip_raw_delta)
        diag_count += 1

        forced_terminal_due_env_error = False
        try:
            ts_next = train_env.step(env_action_exec)
            next_obs = ts_next.observation
            done = bool(ts_next.last())
        except ValueError as exc:
            if "terminated episode" not in str(exc):
                raise
            # Some LIBERO/robosuite wrappers can report terminal one step late.
            # Convert this to a terminal transition and continue training.
            forced_terminal_due_env_error = True
            next_obs = obs
            done = True
            print(f"[warn] forced terminal transition due to env step error: {exc}")

        goal_achieved = bool(next_obs.get("goal_achieved", False))
        # Sparse terminal reward policy:
        # - success terminal(done & goal): 1
        # - failure/truncation/non-terminal: 0
        reward = 1.0 if (done and goal_achieved) else 0.0
        episode_return += reward
        episode_len += 1
        if goal_achieved:
            success_flag = 1

        if done:
            next_base_action_raw = np.zeros((7,), dtype=np.float32)
            cached_base_action_dict = None
        else:
            next_step_in_ep = step_in_ep + 1
            next_base_action_dict = train_base.act(next_obs, next_step_in_ep, global_step)
            next_base_action_raw = np.asarray(
                train_env.point2action(next_base_action_dict), dtype=np.float32
            ).reshape(7)
            cached_base_action_dict = next_base_action_dict

        next_state_vec = observation_to_state(
            obs=next_obs,
            pixel_key=pixel_key,
            base_action_7d=next_base_action_raw,
            include_eef_pos=args.include_eef_pos,
            state_coord_mode=args.state_coord_mode,
        )

        _add_transition_to_rb(
            online_rb,
            state_vec=state_vec,
            next_state_vec=next_state_vec,
            obs_raw=obs,
            next_obs_raw=next_obs,
            base_action_raw=base_action_raw,
            next_base_action_raw=next_base_action_raw,
            combined_action_raw=env_action_exec,
            reward=reward,
            done=done,
            action_normalizer=action_normalizer,
            residual_obs_mode=residual_obs_mode,
            camera_key=args.camera_key,
            dummy_image_size=args.dummy_image_size,
            pixel_key=pixel_key,
            image_obs_key=image_obs_key,
            image_hw=image_hw,
            image_only_prop_dim=prop_dim,
        )
        episode_transition_cache.append(
            {
                "state_vec": state_vec.copy(),
                "next_state_vec": next_state_vec.copy(),
                "obs_raw": obs,
                "next_obs_raw": next_obs,
                "base_action_raw": base_action_raw.copy(),
                "next_base_action_raw": next_base_action_raw.copy(),
                "combined_action_raw": env_action_exec.copy(),
                "reward": float(reward),
                "done": bool(done),
            }
        )
        if done and online_bucket_sampling and online_success_rb is not None and online_fail_rb is not None:
            episode_success = bool(success_flag > 0)
            selected_transitions = _select_online_bucket_transitions(
                episode_transition_cache,
                episode_success=episode_success,
                fail_max_steps=online_fail_max_steps,
                fail_stride=online_fail_stride,
            )
            target_rb = online_success_rb if episode_success else online_fail_rb
            for tr in selected_transitions:
                _add_transition_to_rb(
                    target_rb,
                    state_vec=tr["state_vec"],
                    next_state_vec=tr["next_state_vec"],
                    obs_raw=tr["obs_raw"],
                    next_obs_raw=tr["next_obs_raw"],
                    base_action_raw=tr["base_action_raw"],
                    next_base_action_raw=tr["next_base_action_raw"],
                    combined_action_raw=tr["combined_action_raw"],
                    reward=float(tr["reward"]),
                    done=bool(tr["done"]),
                    action_normalizer=action_normalizer,
                    residual_obs_mode=residual_obs_mode,
                    camera_key=args.camera_key,
                    dummy_image_size=args.dummy_image_size,
                    pixel_key=pixel_key,
                    image_obs_key=image_obs_key,
                    image_hw=image_hw,
                    image_only_prop_dim=prop_dim,
                )

        metrics: dict[str, float] = {}
        used_online_batch = 0
        used_offline_batch = 0
        num_updates_done = 0
        current_offline_fraction = _offline_fraction_at_step(global_step)
        current_offline_batch_size = int(args.batch_size * current_offline_fraction)
        if current_offline_fraction > 0.0 and current_offline_batch_size == 0:
            current_offline_batch_size = 1
        current_online_batch_size = int(args.batch_size - current_offline_batch_size)

        if (
            len(online_rb) >= int(args.warmup_steps)
            and (global_step % update_every_n_steps == 0 or global_step == 1)
        ):
            stddev_for_update = float(resfit_utils.schedule(stddev_schedule, global_step))
            for i in range(num_updates_per_iteration):
                batch, cur_online, cur_offline = _sample_mixed_batch(
                    online_rb=online_rb,
                    offline_rb=offline_rb,
                    online_batch_size=current_online_batch_size,
                    offline_batch_size=current_offline_batch_size,
                    fallback_batch_size=int(args.batch_size),
                    device=device,
                    online_success_rb=online_success_rb,
                    online_fail_rb=online_fail_rb,
                    online_success_batch_fraction=online_success_batch_fraction,
                )
                if batch is None:
                    break

                if bool(args.strict_resfit):
                    _assert_tensor_range(
                        _get_batch_tensor(batch, ("action",)),
                        lo=-1.01,
                        hi=1.01,
                        name="batch.action(norm)",
                    )
                    _assert_tensor_range(
                        _get_batch_tensor(batch, ("obs", "observation.base_action")),
                        lo=-1.01,
                        hi=1.01,
                        name="batch.obs.observation.base_action(norm)",
                    )

                update_actor = ((i + 1) % actor_update_cadence) == 0
                metrics = q_agent.update(
                    batch=batch,
                    stddev=stddev_for_update,
                    update_actor=update_actor,
                    bc_batch=None,
                    ref_agent=q_agent,
                )

                if effective_sampling_strategy == "prioritized_replay" and "_td_errors" in metrics:
                    batch["_priority"] = metrics["_td_errors"]
                    if current_offline_fraction > 0.0 and offline_rb is not None:
                        if cur_online > 0:
                            online_rb.update_tensordict_priority(batch[:cur_online])
                        if cur_offline > 0:
                            offline_rb.update_tensordict_priority(batch[cur_online : cur_online + cur_offline])
                    else:
                        online_rb.update_tensordict_priority(batch)

                used_online_batch = cur_online
                used_offline_batch = cur_offline
                num_updates_done += 1

        if global_step % int(args.log_every) == 0:
            diag_den = max(diag_count, 1)
            row = {
                "step": global_step,
                "online_buffer_size": len(online_rb),
                "offline_buffer_size": len(offline_rb) if offline_rb is not None else 0,
                "online_success_buffer_size": len(online_success_rb) if online_success_rb is not None else 0,
                "online_fail_buffer_size": len(online_fail_rb) if online_fail_rb is not None else 0,
                "episode_return": episode_return,
                "episode_len": episode_len,
                "success": success_flag,
                "offline_fraction": current_offline_fraction,
                "num_updates_done": num_updates_done,
                "used_online_batch": used_online_batch,
                "used_offline_batch": used_offline_batch,
                "critic_loss": float(metrics.get("train/critic_loss", 0.0)),
                "critic_qt": float(metrics.get("train/critic_qt", 0.0)),
                "actor_loss": float(metrics.get("train/actor_loss_total", 0.0)),
                "actor_loss_base": float(metrics.get("train/actor_loss_base", 0.0)),
                "actor_l2_penalty": float(metrics.get("train/actor_l2_penalty", 0.0)),
                "critic_grad_norm": float(metrics.get("train/critic_grad_norm", 0.0)),
                "actor_grad_norm": float(metrics.get("train/actor_grad_norm", 0.0)),
                "encoder_grad_norm": float(metrics.get("train/encoder_grad_norm", 0.0)),
                "importance_weights_mean": float(metrics.get("train/importance_weights_mean", 0.0)),
                "importance_weights_std": float(metrics.get("train/importance_weights_std", 0.0)),
                "importance_weights_min": float(metrics.get("train/importance_weights_min", 0.0)),
                "importance_weights_max": float(metrics.get("train/importance_weights_max", 0.0)),
                "batch_reward": float(metrics.get("data/batch_R", 0.0)),
                "diag_samples": int(diag_count),
                "diag_clip_rate_dim": float(diag_sums["clip_rate_dim"] / diag_den),
                "diag_clip_any_rate": float(diag_sums["clip_any"] / diag_den),
                "diag_residual_induced_clip_rate_dim": float(
                    diag_sums["residual_induced_clip_rate_dim"] / diag_den
                ),
                "diag_base_norm_l2": float(diag_sums["base_norm_l2"] / diag_den),
                "diag_residual_norm_l2": float(diag_sums["residual_norm_l2"] / diag_den),
                "diag_res_over_base": float(diag_sums["res_over_base"] / diag_den),
                "diag_clip_delta_l1_mean": float(diag_sums["clip_delta_l1_mean"] / diag_den),
                "diag_clip_delta_linf": float(diag_sums["clip_delta_linf"] / diag_den),
                "diag_exec_clip_raw_delta_mean": float(diag_sums["exec_clip_raw_delta"] / diag_den),
                "diag_exec_clip_raw_delta_max": float(diag_exec_clip_raw_delta_max),
            }
            _write_csv_row(train_csv, row)
            print(json.dumps(row))
            if wandb_mod is not None:
                wandb_mod.log({f"train/{k}": v for k, v in row.items() if k != "step"}, step=global_step)
            for _k in diag_sums.keys():
                diag_sums[_k] = 0.0
            diag_exec_clip_raw_delta_max = 0.0
            diag_count = 0

        if int(args.eval_every) > 0 and global_step % int(args.eval_every) == 0:
            capture_online_video = False
            if eval_video_enabled and eval_video_dir is not None:
                if episode_idx == 0 and last_online_video_episode < 0:
                    capture_online_video = True
                elif episode_idx > 0 and episode_idx % eval_video_every_ep == 0 and episode_idx != last_online_video_episode:
                    capture_online_video = True
                if capture_online_video:
                    last_online_video_episode = int(episode_idx)

            eval_metrics = _evaluate(
                eval_env=eval_env,
                eval_base=eval_base,
                q_agent=q_agent,
                resfit_utils=resfit_utils,
                action_normalizer=action_normalizer,
                pixel_key=pixel_key,
                include_eef_pos=args.include_eef_pos,
                state_coord_mode=args.state_coord_mode,
                residual_obs_mode=residual_obs_mode,
                camera_key=args.camera_key,
                dummy_image_size=args.dummy_image_size,
                image_obs_key=image_obs_key,
                image_hw=image_hw,
                image_only_prop_dim=prop_dim,
                episodes=int(args.eval_episodes),
                device=device,
                video_dir=eval_video_dir if capture_online_video else None,
                video_fps=int(args.eval_video_fps),
                video_render_size=int(args.eval_video_render_size),
                video_max_episodes=max(1, int(args.eval_video_max_episodes)) if capture_online_video else 0,
                video_tag=args.eval_video_tag,
                video_phase="online",
                step=int(global_step),
                online_episode_idx=int(episode_idx),
                residual_gripper_mode=args.residual_gripper_mode,
                residual_gripper_scale=float(args.residual_gripper_scale),
            )
            eval_row = {"step": global_step, **eval_metrics}
            _write_csv_row(eval_csv, eval_row)
            print(json.dumps(eval_row))
            if wandb_mod is not None:
                wandb_mod.log({k: v for k, v in eval_row.items() if k != "step"}, step=global_step)

        if global_step % int(args.save_every) == 0:
            payload = {
                "step": global_step,
                "agent": q_agent.state_dict(),
                "suite": suite_name,
                "task_name": task_name,
                "pixel_key": pixel_key,
                "camera_key": args.camera_key,
                "dummy_image_size": int(args.dummy_image_size),
                "residual_obs_mode": residual_obs_mode,
                "image_obs_key_requested": image_obs_key,
                "image_obs_key_used": image_key_used,
                "image_hw": list(image_hw) if image_hw is not None else None,
                "obs_shape": list(obs_shape),
                "prop_dim": int(prop_dim),
                "action_normalizer": {
                    "low": action_normalizer.low.tolist(),
                    "high": action_normalizer.high.tolist(),
                },
                "args": vars(args),
            }
            torch.save(payload, ckpt_dir / f"{global_step}.pt")
            torch.save(payload, ckpt_dir / "latest.pt")

        if done:
            ep_row = {
                "step": global_step,
                "episode": episode_idx,
                "episode_return": episode_return,
                "episode_len": episode_len,
                "success": success_flag,
                "online_buffer_size": len(online_rb),
                "offline_buffer_size": len(offline_rb) if offline_rb is not None else 0,
                "offline_fraction": current_offline_fraction,
            }
            _write_csv_row(train_csv, ep_row)
            if wandb_mod is not None:
                wandb_mod.log({f"episode/{k}": v for k, v in ep_row.items() if k not in {"step", "episode"}}, step=global_step)

            episode_idx += 1
            episode_return = 0.0
            episode_len = 0
            success_flag = 0
            episode_transition_cache = []

            ts = train_env.reset()
            obs = ts.observation
            train_base.reset_episode()
            step_in_ep = 0
            cached_base_action_dict = train_base.act(obs, step_in_ep, global_step)
            if forced_terminal_due_env_error:
                print("[info] env reset completed after forced terminal transition")
        else:
            obs = next_obs
            step_in_ep += 1

    final_payload = {
        "step": int(args.steps),
        "agent": q_agent.state_dict(),
        "suite": suite_name,
        "task_name": task_name,
        "pixel_key": pixel_key,
        "camera_key": args.camera_key,
        "dummy_image_size": int(args.dummy_image_size),
        "action_normalizer": {
            "low": action_normalizer.low.tolist(),
            "high": action_normalizer.high.tolist(),
        },
        "args": vars(args),
    }
    torch.save(final_payload, ckpt_dir / "final.pt")
    torch.save(final_payload, ckpt_dir / "latest.pt")
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
