"""Reference point->action pipeline for external validation.

This module mirrors the core rule-based logic in `libero_spatial.py::point2action`
without simulator dependencies. It can be reused in separate projects to verify
that point-level predictions are converted to robot actions as intended.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R

from robot_utils.franka.utils import rigid_transform_3D


@dataclass
class Point2ActionConfig:
    pose_solve_mode: str = "rigid"  # {"rigid", "kp0"}
    pose_delta_gain: float = 1.0
    max_delta_pos: float = 0.05
    max_delta_rot: float = 0.25

    # Gripper command interpretation
    gripper_label_mode: str = "command"  # {"command", "distance"}
    gripper_close_threshold: float = 0.2
    gripper_open_threshold: float = -0.2
    gripper_bias: float = 0.0

    # Distance-mode params
    gripper_distance_close_threshold: float = 0.07
    gripper_distance_open_threshold: float = 0.08
    gripper_distance_control_mode: str = "threshold"  # {"threshold", "target"}
    gripper_distance_value_source: str = "pred_points"  # {"token", "pred_points"}
    gripper_distance_deadband: float = 0.004

    # Gates / hysteresis / smoothing
    gripper_cmd_slew_rate: float = 0.15
    gripper_close_position_gate: bool = True
    gripper_close_position_threshold: float = 0.02
    gripper_close_object_gate: bool = False
    gripper_close_object_threshold: float = 0.06
    gripper_command_fallback_to_tip_distance: bool = False
    gripper_reopen_tip_gate: bool = False
    gripper_tip_close_threshold: float = 0.07
    gripper_tip_open_threshold: float = 0.08
    gripper_min_hold_steps: int = 12
    gripper_close_debounce_steps: int = 0
    gripper_open_debounce_steps: int = 0
    gripper_force_close_on_align: bool = False
    gripper_force_close_align_steps: int = 0

    # Optional close approach compensation
    close_downward_offset: float = 0.0
    close_approach_offset: float = 0.0
    close_approach_steps: int = 0


@dataclass
class Point2ActionState:
    prev_gripper_state: float = -1.0
    gripper_hold_remaining: int = 0
    gripper_close_pending: int = 0
    gripper_open_pending: int = 0
    gripper_align_close_pending: int = 0
    close_approach_remaining: int = 0


def build_rigid_local_indices(
    num_robot_points: int,
    tip_left_local_idx: Optional[int],
    tip_right_local_idx: Optional[int],
) -> Iterable[int]:
    rigid = [
        idx
        for idx in range(int(num_robot_points))
        if idx not in {tip_left_local_idx, tip_right_local_idx}
    ]
    if len(rigid) == 0:
        rigid = list(range(int(num_robot_points)))
    return rigid


def tip_distance_from_points(
    robot_points: np.ndarray,
    tip_left_local_idx: Optional[int],
    tip_right_local_idx: Optional[int],
) -> Optional[float]:
    if tip_left_local_idx is None or tip_right_local_idx is None:
        return None
    left = int(tip_left_local_idx)
    right = int(tip_right_local_idx)
    if robot_points.shape[0] <= max(left, right):
        return None
    return float(np.linalg.norm(robot_points[left] - robot_points[right]))


def compute_non_tip_alignment_error(
    current_robot_points: np.ndarray,
    target_robot_points: np.ndarray,
    rigid_local_indices: Iterable[int],
) -> Optional[float]:
    valid = [
        int(idx)
        for idx in rigid_local_indices
        if int(idx) < current_robot_points.shape[0] and int(idx) < target_robot_points.shape[0]
    ]
    if len(valid) == 0:
        return None
    return float(
        np.linalg.norm(current_robot_points[valid] - target_robot_points[valid], axis=-1).mean()
    )


def solve_delta_pose_from_points(
    current_robot_points: np.ndarray,
    target_robot_points: np.ndarray,
    current_eef_pos: np.ndarray,
    current_eef_rot: np.ndarray,
    rigid_local_indices: Iterable[int],
    cfg: Point2ActionConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    current_robot_points = np.asarray(current_robot_points, dtype=np.float32).reshape(-1, 3)
    target_robot_points = np.asarray(target_robot_points, dtype=np.float32).reshape(-1, 3)
    current_eef_pos = np.asarray(current_eef_pos, dtype=np.float32).reshape(3)
    current_eef_rot = np.asarray(current_eef_rot, dtype=np.float32).reshape(3, 3)

    if str(cfg.pose_solve_mode).strip().lower() == "kp0":
        delta_pos = target_robot_points[0] - current_robot_points[0]
        delta_rot = np.zeros((3,), dtype=np.float32)
    else:
        try:
            valid = [
                int(idx)
                for idx in rigid_local_indices
                if int(idx) < current_robot_points.shape[0] and int(idx) < target_robot_points.shape[0]
            ]
            src_points = current_robot_points
            dst_points = target_robot_points
            if len(valid) >= 3:
                src_points = current_robot_points[valid]
                dst_points = target_robot_points[valid]
            rot, trans = rigid_transform_3D(src_points, dst_points)
            rot = np.asarray(rot, dtype=np.float32)
            trans = np.asarray(trans, dtype=np.float32).reshape(3)

            target_eef_pos = rot @ current_eef_pos + trans
            delta_pos = target_eef_pos - current_eef_pos
            target_eef_rot = rot @ current_eef_rot
            delta_rot = R.from_matrix(target_eef_rot @ current_eef_rot.T).as_rotvec()
        except Exception:
            delta_pos = target_robot_points[0] - current_eef_pos
            delta_rot = np.zeros((3,), dtype=np.float32)

    delta_pos = np.asarray(delta_pos, dtype=np.float32) * float(cfg.pose_delta_gain)
    delta_pos = np.clip(delta_pos, -float(cfg.max_delta_pos), float(cfg.max_delta_pos))
    delta_rot = np.asarray(delta_rot, dtype=np.float32)
    delta_rot = np.clip(delta_rot, -float(cfg.max_delta_rot), float(cfg.max_delta_rot))
    delta_rot = np.nan_to_num(delta_rot, nan=0.0, posinf=0.0, neginf=0.0)
    return delta_pos.astype(np.float32), delta_rot.astype(np.float32)


def _apply_gripper_debounce(
    prev_cmd: float,
    desired_cmd: float,
    state: Point2ActionState,
    cfg: Point2ActionConfig,
) -> float:
    if desired_cmd > prev_cmd + 1e-6:
        state.gripper_open_pending = 0
        state.gripper_close_pending += 1
        if (
            int(cfg.gripper_close_debounce_steps) > 0
            and state.gripper_close_pending < int(cfg.gripper_close_debounce_steps)
        ):
            return prev_cmd
        state.gripper_close_pending = 0
        return 1.0
    if desired_cmd < prev_cmd - 1e-6:
        state.gripper_close_pending = 0
        state.gripper_open_pending += 1
        if (
            int(cfg.gripper_open_debounce_steps) > 0
            and state.gripper_open_pending < int(cfg.gripper_open_debounce_steps)
        ):
            return prev_cmd
        state.gripper_open_pending = 0
        return -1.0
    state.gripper_close_pending = 0
    state.gripper_open_pending = 0
    return prev_cmd


def point2action_step(
    *,
    current_robot_points: np.ndarray,
    target_robot_points: np.ndarray,
    current_eef_pos: np.ndarray,
    current_eef_rot: np.ndarray,
    raw_gripper_value: float,
    state: Point2ActionState,
    cfg: Point2ActionConfig,
    rigid_local_indices: Iterable[int],
    tip_left_local_idx: Optional[int] = None,
    tip_right_local_idx: Optional[int] = None,
    current_gripper_distance: Optional[float] = None,
    bowl_distance: Optional[float] = None,
) -> Tuple[np.ndarray, Point2ActionState, Dict[str, float]]:
    """Convert predicted robot points + gripper value into 7D action.

    Returns:
      - robot_action: np.ndarray shape (7,)
      - state: updated Point2ActionState
      - debug: intermediate scalar values
    """
    current_robot_points = np.asarray(current_robot_points, dtype=np.float32).reshape(-1, 3)
    target_robot_points = np.asarray(target_robot_points, dtype=np.float32).reshape(-1, 3)

    delta_pos, delta_rot = solve_delta_pose_from_points(
        current_robot_points=current_robot_points,
        target_robot_points=target_robot_points,
        current_eef_pos=current_eef_pos,
        current_eef_rot=current_eef_rot,
        rigid_local_indices=rigid_local_indices,
        cfg=cfg,
    )

    gripper_value = float(raw_gripper_value)
    prev_cmd = 1.0 if float(state.prev_gripper_state) >= 0.0 else -1.0
    desired_cmd = prev_cmd
    pred_tip_distance = tip_distance_from_points(
        target_robot_points, tip_left_local_idx, tip_right_local_idx
    )

    if str(cfg.gripper_label_mode).strip().lower() == "distance":
        if str(cfg.gripper_distance_value_source).strip().lower() == "pred_points":
            if pred_tip_distance is not None:
                gripper_value = float(pred_tip_distance)

        if str(cfg.gripper_distance_control_mode).strip().lower() == "target":
            if current_gripper_distance is None:
                current_gripper_distance = gripper_value
            close_gate = gripper_value <= float(cfg.gripper_distance_close_threshold)
            if prev_cmd < 0.0:
                if close_gate and current_gripper_distance > (
                    gripper_value + float(cfg.gripper_distance_deadband)
                ):
                    desired_cmd = 1.0
                else:
                    desired_cmd = -1.0
            else:
                if (
                    gripper_value >= float(cfg.gripper_distance_close_threshold)
                    and current_gripper_distance
                    < (gripper_value - float(cfg.gripper_distance_deadband))
                ):
                    desired_cmd = -1.0
                elif close_gate and current_gripper_distance > (
                    gripper_value + float(cfg.gripper_distance_deadband)
                ):
                    desired_cmd = 1.0
                else:
                    desired_cmd = prev_cmd
        else:
            if prev_cmd < 0.0 and gripper_value <= float(cfg.gripper_distance_close_threshold):
                desired_cmd = 1.0
            elif prev_cmd > 0.0 and gripper_value >= float(cfg.gripper_distance_open_threshold):
                desired_cmd = -1.0
    else:
        gripper_value = gripper_value + float(cfg.gripper_bias)
        if prev_cmd < 0.0 and gripper_value > float(cfg.gripper_close_threshold):
            desired_cmd = 1.0
        elif prev_cmd > 0.0 and gripper_value < float(cfg.gripper_open_threshold):
            desired_cmd = -1.0

        if bool(cfg.gripper_command_fallback_to_tip_distance) and pred_tip_distance is not None:
            if prev_cmd < 0.0 and pred_tip_distance <= float(cfg.gripper_tip_close_threshold):
                desired_cmd = 1.0
            elif prev_cmd > 0.0 and pred_tip_distance >= float(cfg.gripper_tip_open_threshold):
                desired_cmd = -1.0

        if bool(cfg.gripper_reopen_tip_gate) and prev_cmd > 0.0 and desired_cmd < 0.0:
            if pred_tip_distance is None or pred_tip_distance < float(cfg.gripper_tip_open_threshold):
                desired_cmd = 1.0

    align_err = compute_non_tip_alignment_error(
        current_robot_points=current_robot_points,
        target_robot_points=target_robot_points,
        rigid_local_indices=rigid_local_indices,
    )

    if bool(cfg.gripper_force_close_on_align) and prev_cmd < 0.0:
        if align_err is not None and align_err <= float(cfg.gripper_close_position_threshold):
            state.gripper_align_close_pending += 1
        else:
            state.gripper_align_close_pending = 0
        required = max(1, int(cfg.gripper_force_close_align_steps))
        if state.gripper_align_close_pending >= required:
            desired_cmd = 1.0
    else:
        state.gripper_align_close_pending = 0

    desired_cmd = _apply_gripper_debounce(prev_cmd, desired_cmd, state, cfg)

    if state.gripper_hold_remaining > 0:
        gripper_cmd = prev_cmd
        state.gripper_hold_remaining -= 1
    else:
        gripper_cmd = desired_cmd
        if int(cfg.gripper_min_hold_steps) > 0 and abs(gripper_cmd - prev_cmd) > 1e-6:
            state.gripper_hold_remaining = int(cfg.gripper_min_hold_steps)

    if bool(cfg.gripper_close_position_gate) and gripper_cmd > 0.0:
        if align_err is not None and align_err > float(cfg.gripper_close_position_threshold):
            gripper_cmd = -1.0
            state.gripper_hold_remaining = 0

    if bool(cfg.gripper_close_object_gate) and gripper_cmd > 0.0 and prev_cmd < 0.0:
        if bowl_distance is None or bowl_distance > float(cfg.gripper_close_object_threshold):
            gripper_cmd = -1.0
            state.gripper_hold_remaining = 0

    prev_cmd_cont = float(state.prev_gripper_state)
    if float(cfg.gripper_cmd_slew_rate) < 2.0:
        delta_cmd = float(gripper_cmd) - prev_cmd_cont
        delta_cmd = float(
            np.clip(delta_cmd, -float(cfg.gripper_cmd_slew_rate), float(cfg.gripper_cmd_slew_rate))
        )
        gripper_cmd = prev_cmd_cont + delta_cmd

    gripper_cmd = float(np.clip(gripper_cmd, -1.0, 1.0))
    state.prev_gripper_state = gripper_cmd

    if prev_cmd < 0.0 and gripper_cmd > 0.0 and int(cfg.close_approach_steps) > 0:
        state.close_approach_remaining = int(cfg.close_approach_steps)
    if state.close_approach_remaining > 0:
        if float(cfg.close_approach_offset) > 0.0 and target_robot_points.shape[0] > 4:
            approach_dir = target_robot_points[4] - target_robot_points[0]
            norm = float(np.linalg.norm(approach_dir))
            if norm > 1e-6:
                delta_pos = delta_pos + (approach_dir / norm) * float(cfg.close_approach_offset)
        if float(cfg.close_downward_offset) > 0.0:
            delta_pos[2] = delta_pos[2] - float(cfg.close_downward_offset)
        state.close_approach_remaining -= 1
        delta_pos = np.clip(delta_pos, -float(cfg.max_delta_pos), float(cfg.max_delta_pos))

    robot_action = np.zeros((7,), dtype=np.float32)
    robot_action[:3] = np.asarray(delta_pos, dtype=np.float32)
    robot_action[3:6] = np.asarray(delta_rot, dtype=np.float32)
    robot_action[6] = float(gripper_cmd)

    debug = {
        "raw_gripper_value": float(raw_gripper_value),
        "gripper_value_after_bias_or_source": float(gripper_value),
        "prev_cmd": float(prev_cmd),
        "desired_cmd": float(desired_cmd),
        "gripper_cmd_final": float(gripper_cmd),
        "align_err": float(align_err) if align_err is not None else np.nan,
        "pred_tip_distance": float(pred_tip_distance) if pred_tip_distance is not None else np.nan,
        "current_gripper_distance": (
            float(current_gripper_distance)
            if current_gripper_distance is not None
            else np.nan
        ),
        "bowl_distance": float(bowl_distance) if bowl_distance is not None else np.nan,
    }
    return robot_action, state, debug

