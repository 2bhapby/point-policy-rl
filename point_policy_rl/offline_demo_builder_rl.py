from __future__ import annotations

import pickle as pkl
from pathlib import Path
from typing import Any, Callable

import numpy as np

from point_policy_rl.env_bridge_rl import observation_to_state


def _resolve_demo_file(
    demo_root: Path,
    suite_name: str,
    task_name: str,
) -> Path:
    if demo_root.is_file():
        return demo_root

    candidate_paths = [
        demo_root / f"{task_name}.pkl",
        demo_root / suite_name / f"{task_name}.pkl",
        demo_root / suite_name.replace("libero_", "") / f"{task_name}.pkl",
    ]
    for path in candidate_paths:
        if path.exists():
            return path

    recursive = sorted(demo_root.rglob(f"{task_name}.pkl"))
    if recursive:
        return recursive[0]

    raise FileNotFoundError(
        f"Offline demo pkl not found for task '{task_name}' under: {demo_root}"
    )


def _stack_observation_sequence(
    obs_seq: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    stacked: dict[str, np.ndarray] = {}
    if len(obs_seq) == 0:
        return stacked

    keys = set()
    for obs in obs_seq:
        if isinstance(obs, dict):
            keys.update(obs.keys())

    for key in sorted(keys):
        values = []
        valid = True
        for obs in obs_seq:
            if not isinstance(obs, dict) or key not in obs:
                valid = False
                break
            values.append(np.asarray(obs[key]))
        if not valid or len(values) == 0:
            continue
        try:
            stacked[key] = np.stack(values, axis=0)
        except Exception:
            continue
    return stacked


def _convert_rlds_episodes_to_observations(
    episodes: list[Any],
    low: np.ndarray,
    high: np.ndarray,
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    low = np.asarray(low, dtype=np.float32).reshape(7)
    high = np.asarray(high, dtype=np.float32).reshape(7)

    for episode in episodes:
        if not isinstance(episode, dict):
            continue
        steps = episode.get("steps")
        if not isinstance(steps, list) or len(steps) < 2:
            continue

        obs_seq: list[dict[str, Any]] = []
        actions: list[np.ndarray] = []
        has_all_actions = True

        for step in steps:
            if not isinstance(step, dict):
                continue
            obs = step.get("observation")
            if not isinstance(obs, dict):
                continue
            obs_seq.append(obs)

            raw_action = step.get("action", None)
            if raw_action is None:
                has_all_actions = False
                continue
            action_arr = np.asarray(raw_action, dtype=np.float32).reshape(-1)
            if action_arr.size < 7:
                has_all_actions = False
                continue
            actions.append(np.clip(action_arr[:7], low, high).astype(np.float32))

        if len(obs_seq) < 2:
            continue

        demo_obs = _stack_observation_sequence(obs_seq)
        if has_all_actions and len(actions) == len(obs_seq):
            demo_obs["actions"] = np.stack(actions, axis=0).astype(np.float32)

        converted.append(demo_obs)

    return converted


def _slice_observation_step(demo_obs: dict[str, Any], t: int) -> dict[str, Any]:
    step_obs: dict[str, Any] = {}
    for key, value in demo_obs.items():
        arr = np.asarray(value)
        if arr.ndim == 0 or arr.shape[0] <= t:
            continue
        step_obs[key] = arr[t]
    return step_obs


def _to_gripper_command(raw_value: float) -> float:
    if raw_value > 0.25:
        return 1.0
    if raw_value < -0.25:
        return -1.0
    return float(np.clip(raw_value, -1.0, 1.0))


def _estimate_base_action_from_demo(
    demo_obs: dict[str, Any],
    t: int,
    pixel_key: str,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    delta_pos = np.zeros((3,), dtype=np.float32)
    features = np.asarray(
        demo_obs.get("features", np.zeros((0,), dtype=np.float32)),
        dtype=np.float32,
    )
    if features.ndim == 2 and features.shape[0] > (t + 1) and features.shape[1] >= 3:
        delta_pos = (features[t + 1, :3] - features[t, :3]).astype(np.float32)
    else:
        robot_key = f"robot_tracks_3d_{pixel_key}"
        point_key = f"point_tracks_{pixel_key}"
        if robot_key in demo_obs:
            robot_pts = np.asarray(demo_obs[robot_key], dtype=np.float32)
            if robot_pts.ndim == 3 and robot_pts.shape[0] > (t + 1) and robot_pts.shape[1] > 0:
                delta_pos = (robot_pts[t + 1, 0, :3] - robot_pts[t, 0, :3]).astype(np.float32)
        elif point_key in demo_obs:
            point_tracks = np.asarray(demo_obs[point_key], dtype=np.float32)
            if (
                point_tracks.ndim == 3
                and point_tracks.shape[0] > (t + 1)
                and point_tracks.shape[1] > 0
            ):
                delta_pos = (point_tracks[t + 1, 0, :3] - point_tracks[t, 0, :3]).astype(np.float32)

    gripper_states = np.asarray(
        demo_obs.get("gripper_states", np.zeros((0,), dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1)
    gripper_value = -1.0
    if gripper_states.shape[0] > (t + 1):
        gripper_value = float(gripper_states[t + 1])
    elif features.ndim == 2 and features.shape[0] > (t + 1) and features.shape[1] >= 1:
        gripper_value = float(features[t + 1, -1])
    gripper_cmd = _to_gripper_command(gripper_value)

    action = np.concatenate(
        [
            delta_pos.astype(np.float32),
            np.zeros((3,), dtype=np.float32),
            np.array([gripper_cmd], dtype=np.float32),
        ],
        axis=0,
    )
    return np.clip(action, low, high).astype(np.float32)


def _estimate_base_action_from_bc_action_dict(
    obs_t: dict[str, Any],
    bc_action_dict: dict[str, Any],
    pixel_key: str,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    point_key = f"point_tracks_{pixel_key}"
    action_key = f"future_tracks_{pixel_key}"
    if action_key not in bc_action_dict:
        raise KeyError(f"{action_key} missing in BC action dict")
    if point_key not in obs_t:
        raise KeyError(f"{point_key} missing in observation dict")

    current_tracks = np.asarray(obs_t[point_key], dtype=np.float32)
    future_tracks = np.asarray(bc_action_dict[action_key], dtype=np.float32)
    if future_tracks.ndim == 3:
        future_tracks = future_tracks[0]

    if current_tracks.ndim != 2 or current_tracks.shape[0] < 1 or current_tracks.shape[1] < 3:
        delta_pos = np.zeros((3,), dtype=np.float32)
    elif future_tracks.ndim != 2 or future_tracks.shape[0] < 1 or future_tracks.shape[1] < 3:
        delta_pos = np.zeros((3,), dtype=np.float32)
    else:
        delta_pos = (future_tracks[0, :3] - current_tracks[0, :3]).astype(np.float32)

    if "gripper" in bc_action_dict:
        gripper_value = float(np.asarray(bc_action_dict["gripper"]).reshape(-1)[0])
    elif "future_gripper_states" in bc_action_dict:
        gripper_value = float(np.asarray(bc_action_dict["future_gripper_states"]).reshape(-1)[0])
    else:
        features = np.asarray(obs_t.get("features", np.array([0.0], dtype=np.float32)), dtype=np.float32).reshape(-1)
        gripper_value = float(features[-1]) if features.size > 0 else -1.0
    gripper_cmd = _to_gripper_command(gripper_value)

    action = np.concatenate(
        [
            delta_pos.astype(np.float32),
            np.zeros((3,), dtype=np.float32),
            np.array([gripper_cmd], dtype=np.float32),
        ],
        axis=0,
    )
    return np.clip(action, low, high).astype(np.float32)


def _extract_demo_full_actions(
    demo_obs: dict[str, Any],
    horizon: int,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray | None:
    for key in ("actions", "action"):
        if key not in demo_obs:
            continue
        arr = np.asarray(demo_obs[key], dtype=np.float32)
        if arr.ndim == 1:
            if arr.size < 7:
                continue
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] < 7 or arr.shape[0] < 1:
            continue
        idx = np.clip(np.arange(horizon), 0, arr.shape[0] - 1)
        return np.clip(arr[idx, :7], low, high).astype(np.float32)
    return None


def build_offline_transitions_from_expert_demos_rl(
    demo_root: Path,
    suite_name: str,
    task_name: str,
    pixel_key: str,
    include_eef_pos: bool,
    low: np.ndarray,
    high: np.ndarray,
    max_demos: int | None = None,
    terminal_reward: float = 1.0,
    step_reward: float = 0.0,
    base_action_mode: str = "demo_delta",
    transition_action_mode: str = "residual_zero",
    bc_action_fn: Callable[[dict[str, Any], int], dict[str, Any]] | None = None,
    bc_reset_episode_fn: Callable[[], None] | None = None,
) -> list[dict[str, Any]]:
    demo_root = Path(demo_root).expanduser().resolve()
    demo_pkl_path = _resolve_demo_file(
        demo_root=demo_root,
        suite_name=suite_name,
        task_name=task_name,
    )

    with demo_pkl_path.open("rb") as f:
        payload = pkl.load(f)

    low = np.asarray(low, dtype=np.float32).reshape(7)
    high = np.asarray(high, dtype=np.float32).reshape(7)
    point_key = f"point_tracks_{pixel_key}"

    observations: list[dict[str, Any]] | None = None
    if isinstance(payload, dict) and isinstance(payload.get("observations"), list):
        observations = payload["observations"]
    elif isinstance(payload, dict) and isinstance(payload.get("episodes"), list):
        observations = _convert_rlds_episodes_to_observations(
            episodes=payload["episodes"],
            low=low,
            high=high,
        )
    else:
        raise ValueError(f"Unsupported offline demo format: {demo_pkl_path}")

    if not isinstance(observations, list):
        raise ValueError(f"'observations' must be a list in {demo_pkl_path}")

    mode = str(base_action_mode).strip().lower()
    if mode not in {"demo_delta", "bc_track_delta"}:
        raise ValueError(f"Unsupported base_action_mode: {base_action_mode}")
    if mode == "bc_track_delta" and bc_action_fn is None:
        raise ValueError("bc_action_fn is required when base_action_mode='bc_track_delta'")

    action_mode = str(transition_action_mode).strip().lower()
    if action_mode not in {"residual_zero", "combined_base"}:
        raise ValueError(f"Unsupported transition_action_mode: {transition_action_mode}")

    if max_demos is None or max_demos <= 0:
        demo_limit = len(observations)
    else:
        demo_limit = min(len(observations), int(max_demos))

    transitions: list[dict[str, Any]] = []
    for demo_idx in range(demo_limit):
        demo_obs = observations[demo_idx]
        if not isinstance(demo_obs, dict) or point_key not in demo_obs:
            continue

        point_tracks = np.asarray(demo_obs[point_key], dtype=np.float32)
        if point_tracks.ndim != 3 or point_tracks.shape[0] < 2:
            continue
        horizon = int(point_tracks.shape[0])
        demo_full_actions = _extract_demo_full_actions(
            demo_obs=demo_obs,
            horizon=horizon,
            low=low,
            high=high,
        )
        step_observations = [_slice_observation_step(demo_obs, t) for t in range(horizon)]

        base_actions = []
        if mode == "demo_delta":
            for t in range(horizon):
                if t == (horizon - 1):
                    base_actions.append(np.zeros((7,), dtype=np.float32))
                else:
                    base_actions.append(
                        _estimate_base_action_from_demo(
                            demo_obs=demo_obs,
                            t=t,
                            pixel_key=pixel_key,
                            low=low,
                            high=high,
                        )
                    )
        else:
            if bc_reset_episode_fn is not None:
                bc_reset_episode_fn()
            for t in range(horizon):
                bc_action_dict = bc_action_fn(step_observations[t], t)
                base_actions.append(
                    _estimate_base_action_from_bc_action_dict(
                        obs_t=step_observations[t],
                        bc_action_dict=bc_action_dict,
                        pixel_key=pixel_key,
                        low=low,
                        high=high,
                    )
                )

        for t in range(horizon - 1):
            obs_t = step_observations[t]
            obs_tp1 = step_observations[t + 1]
            base_action = base_actions[t]

            done = bool(t == (horizon - 2))
            if done:
                next_base_action = np.zeros((7,), dtype=np.float32)
                reward = float(terminal_reward)
            else:
                next_base_action = base_actions[t + 1]
                reward = float(step_reward)

            state = observation_to_state(
                obs=obs_t,
                pixel_key=pixel_key,
                base_action_7d=base_action,
                include_eef_pos=include_eef_pos,
            )
            next_state = observation_to_state(
                obs=obs_tp1,
                pixel_key=pixel_key,
                base_action_7d=next_base_action,
                include_eef_pos=include_eef_pos,
            )

            if action_mode == "residual_zero":
                transition_action = np.zeros((7,), dtype=np.float32)
            elif demo_full_actions is not None:
                transition_action = demo_full_actions[t].copy()
            else:
                transition_action = base_action.copy()

            transitions.append(
                {
                    "obs": state,
                    "action": transition_action,
                    "reward": reward,
                    "next_obs": next_state,
                    "done": done,
                    "base_action": base_action.copy(),
                    "next_base_action": next_base_action.copy(),
                }
            )

    return transitions
