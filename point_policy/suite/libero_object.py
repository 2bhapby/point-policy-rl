import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, NamedTuple, Sequence

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

try:
    import dm_env
    from dm_env import StepType, TimeStep, specs
except ModuleNotFoundError:
    from suite.dm_env_compat import dm_env, StepType, TimeStep, specs

from robot_utils.franka.utils import rigid_transform_3D

_LIBERO_IMPORT_ERROR = None
try:
    from libero.libero import get_libero_path
    from libero.libero.benchmark import get_benchmark
    from libero.libero.envs import OffScreenRenderEnv
except Exception as exc:
    _LIBERO_IMPORT_ERROR = exc
    repo_root = Path(__file__).resolve().parents[2]
    local_libero_root = repo_root / "LIBERO"
    if local_libero_root.exists():
        sys.path.insert(0, str(local_libero_root))
        try:
            from libero.libero import get_libero_path
            from libero.libero.benchmark import get_benchmark
            from libero.libero.envs import OffScreenRenderEnv
            _LIBERO_IMPORT_ERROR = None
        except Exception as local_exc:
            _LIBERO_IMPORT_ERROR = local_exc
            get_libero_path = None
            get_benchmark = None
            OffScreenRenderEnv = None
    else:
        get_libero_path = None
        get_benchmark = None
        OffScreenRenderEnv = None

try:
    from robosuite.utils.camera_utils import (
        get_camera_transform_matrix,
        project_points_from_world_to_camera,
    )
except Exception:
    get_camera_transform_matrix = None
    project_points_from_world_to_camera = None


ROBOT_SITE_SUFFIXES = (
    "kp0_wrist",
    "kp1_tip_left",
    "kp2_tip_right",
    "kp3_body_r1_left",
    "kp4_body_r1_center",
    "kp5_body_r1_right",
    "kp6_body_r2_left",
    "kp7_body_r2_center",
    "kp8_body_r2_right",
)

TARGET_OBJECT_SITE_NAMES = ("bottom_site", "top_site", "horizontal_radius_site")
TARGET_OBJECT_DEFAULT_LOCAL_POINTS = np.array(
    [[0.0, 0.0, -0.04], [0.0, 0.0, 0.04], [0.025, 0.025, 0.0]], dtype=np.float32
)

def _to_str_name(name):
    if isinstance(name, bytes):
        return name.decode("utf-8")
    return str(name)


def _parse_vec(attr_value, expected_dim, default_value):
    if attr_value is None:
        return np.asarray(default_value, dtype=np.float32)
    vec = np.fromstring(attr_value, sep=" ", dtype=np.float32)
    if vec.shape[0] != expected_dim:
        return np.asarray(default_value, dtype=np.float32)
    return vec


def _quat_wxyz_to_xyzw(quat_wxyz):
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float32).reshape(-1)
    return np.array(
        [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32
    )


def _instance_to_category(instance_name):
    return re.sub(r"_[0-9]+$", "", str(instance_name))


def _to_bool_flag(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _parse_index_list(indices, max_size):
    if indices is None:
        return None
    if isinstance(indices, str):
        text = indices.strip()
        if text == "" or text.lower() in {"none", "null"}:
            return None
        text = text.strip("[]")
        if text == "":
            return []
        parsed = [int(token.strip()) for token in text.split(",") if token.strip() != ""]
    elif isinstance(indices, (list, tuple, np.ndarray)):
        parsed = [int(v) for v in indices]
    elif hasattr(indices, "__iter__"):
        parsed = [int(v) for v in indices]
    else:
        raise TypeError(
            "robot_point_indices must be None, string, list, tuple, or numpy array"
        )

    if len(parsed) != len(set(parsed)):
        raise ValueError(f"robot_point_indices must be unique, got {parsed}")
    invalid = [idx for idx in parsed if idx < 0 or idx >= int(max_size)]
    if invalid:
        raise ValueError(
            f"robot_point_indices out of range [0, {int(max_size) - 1}]: {invalid}"
        )
    return parsed


class RGBArrayAsObservationWrapper(dm_env.Environment):
    def __init__(
        self,
        env,
        task_name,
        init_states,
        max_episode_len=300,
        max_state_dim=3,
        pixel_keys=("pixels1", "pixels2"),
        use_robot_points=True,
        num_robot_points=9,
        robot_point_indices=None,
        use_object_points=True,
        num_object_points=7,
        max_delta_pos=0.05,
        max_delta_rot=0.25,
        gripper_label_mode="command",
        gripper_threshold=0.0,
        gripper_close_threshold=-0.3,
        gripper_open_threshold=0.6,
        gripper_bias=0.0,
        gripper_qpos_close_threshold=0.02,
        gripper_distance_close_threshold=0.07,
        gripper_distance_open_threshold=0.08,
        gripper_distance_control_mode="target",
        gripper_distance_value_source="token",
        gripper_distance_deadband=0.003,
        gripper_cmd_slew_rate=2.0,
        gripper_close_position_gate=False,
        gripper_close_position_threshold=0.02,
        real_deploy_mode=False,
        gripper_command_fallback_to_tip_distance=False,
        gripper_tip_close_threshold=0.07,
        gripper_tip_open_threshold=0.08,
        gripper_min_hold_steps=0,
        gripper_close_debounce_steps=0,
        gripper_open_debounce_steps=0,
        gripper_force_close_on_align=False,
        gripper_force_close_align_steps=0,
        close_downward_offset=0.0,
        close_approach_offset=0.0,
        close_approach_steps=0,
        seed=0,
    ):
        self._env = env
        self._task_name = task_name
        self._init_states = init_states
        self._max_episode_len = max_episode_len
        self._max_state_dim = max(1, int(max_state_dim))
        self._pixel_keys = list(pixel_keys)
        self._use_robot_points = bool(use_robot_points)
        self._num_robot_points = int(num_robot_points)
        self._robot_point_indices = _parse_index_list(
            robot_point_indices, len(ROBOT_SITE_SUFFIXES)
        )
        self._use_object_points = bool(use_object_points)
        self._num_object_points = int(num_object_points)
        self._max_delta_pos = float(max_delta_pos)
        self._max_delta_rot = float(max_delta_rot)
        self._gripper_label_mode = str(gripper_label_mode).strip().lower()
        if self._gripper_label_mode not in {"command", "distance"}:
            raise ValueError(
                "gripper_label_mode must be 'command' or 'distance', "
                f"got {gripper_label_mode}"
            )
        self._gripper_threshold = float(gripper_threshold)
        self._gripper_close_threshold = float(gripper_close_threshold)
        self._gripper_open_threshold = float(gripper_open_threshold)
        self._gripper_bias = float(gripper_bias)
        self._gripper_qpos_close_threshold = float(gripper_qpos_close_threshold)
        self._gripper_distance_close_threshold = float(gripper_distance_close_threshold)
        self._gripper_distance_open_threshold = float(gripper_distance_open_threshold)
        self._gripper_distance_control_mode = (
            str(gripper_distance_control_mode).strip().lower()
        )
        if self._gripper_distance_control_mode not in {"threshold", "target"}:
            raise ValueError(
                "gripper_distance_control_mode must be 'threshold' or 'target', "
                f"got {gripper_distance_control_mode}"
            )
        self._gripper_distance_value_source = (
            str(gripper_distance_value_source).strip().lower()
        )
        if self._gripper_distance_value_source not in {"token", "pred_points"}:
            raise ValueError(
                "gripper_distance_value_source must be 'token' or 'pred_points', "
                f"got {gripper_distance_value_source}"
            )
        self._gripper_distance_deadband = float(gripper_distance_deadband)
        self._gripper_cmd_slew_rate = float(gripper_cmd_slew_rate)
        self._gripper_close_position_gate = _to_bool_flag(gripper_close_position_gate)
        self._gripper_close_position_threshold = float(gripper_close_position_threshold)
        self._real_deploy_mode = _to_bool_flag(real_deploy_mode)
        self._gripper_command_fallback_to_tip_distance = _to_bool_flag(
            gripper_command_fallback_to_tip_distance
        )
        self._gripper_tip_close_threshold = float(gripper_tip_close_threshold)
        self._gripper_tip_open_threshold = float(gripper_tip_open_threshold)
        self._gripper_min_hold_steps = max(0, int(gripper_min_hold_steps))
        self._gripper_close_debounce_steps = max(0, int(gripper_close_debounce_steps))
        self._gripper_open_debounce_steps = max(0, int(gripper_open_debounce_steps))
        self._gripper_force_close_on_align = _to_bool_flag(gripper_force_close_on_align)
        self._gripper_force_close_align_steps = max(
            0, int(gripper_force_close_align_steps)
        )
        self._gripper_hold_remaining = 0
        self._gripper_close_pending = 0
        self._gripper_open_pending = 0
        self._gripper_align_close_pending = 0
        self._close_downward_offset = float(close_downward_offset)
        self._close_approach_offset = float(close_approach_offset)
        self._close_approach_steps = max(0, int(close_approach_steps))
        self._close_approach_remaining = 0
        self._rng = np.random.default_rng(seed)
        self._step = 0
        self._prev_gripper_state = -1.0
        self._current_raw_obs = None
        self._last_robot_action = np.zeros((7,), dtype=np.float32)
        self._warned_missing_eef_quat = False

        if self._use_robot_points:
            if self._robot_point_indices is None:
                if self._num_robot_points != len(ROBOT_SITE_SUFFIXES):
                    raise ValueError(
                        f"Expected num_robot_points={len(ROBOT_SITE_SUFFIXES)} "
                        "for kp0~kp8 when robot_point_indices is not set, "
                        f"got {self._num_robot_points}"
                    )
                self._robot_point_indices = list(range(len(ROBOT_SITE_SUFFIXES)))
            if len(self._robot_point_indices) != self._num_robot_points:
                raise ValueError(
                    "robot_point_indices length must match num_robot_points. "
                    f"len={len(self._robot_point_indices)} "
                    f"num_robot_points={self._num_robot_points}"
                )
        else:
            self._robot_point_indices = []

        self._tip_left_local_idx = None
        self._tip_right_local_idx = None
        if 1 in self._robot_point_indices and 2 in self._robot_point_indices:
            self._tip_left_local_idx = self._robot_point_indices.index(1)
            self._tip_right_local_idx = self._robot_point_indices.index(2)

        self._rigid_local_indices = [
            idx
            for idx in range(self._num_robot_points)
            if idx not in {self._tip_left_local_idx, self._tip_right_local_idx}
        ]
        if len(self._rigid_local_indices) == 0:
            self._rigid_local_indices = list(range(self._num_robot_points))

        if self._gripper_label_mode == "distance" and (
            self._tip_left_local_idx is None or self._tip_right_local_idx is None
        ):
            raise ValueError(
                "gripper_label_mode='distance' requires robot_point_indices to include "
                "kp1_tip_left(index 1) and kp2_tip_right(index 2)."
            )
        if (
            self._gripper_command_fallback_to_tip_distance
            and (
                self._tip_left_local_idx is None
                or self._tip_right_local_idx is None
            )
        ):
            print(
                "[libero_object] warning: "
                "gripper_command_fallback_to_tip_distance=true but tip indices "
                "(1,2) are not present in robot_point_indices. Disabling fallback."
            )
            self._gripper_command_fallback_to_tip_distance = False

        if self._use_object_points and self._num_object_points != 7:
            raise ValueError(
                f"Expected num_object_points=7 (basket4 + object3), got {self._num_object_points}"
            )

        self._pixel_to_camera = self._make_pixel_to_camera_map(self._pixel_keys)

        self._site_names = None
        self._body_names = None
        self._site_name_to_id = None
        self._body_name_to_id = None
        self._robot_site_ids = None

        self._obj_of_interest = list(getattr(self._env, "obj_of_interest", []))
        self._target_object_name = None
        self._basket_object_name = None
        self._target_body_id = None
        self._basket_body_id = None
        self._target_local_points = None
        self._basket_local_corners = None

        initial_obs = self.reset()
        self._obs_spec = {}
        for key in self._pixel_keys:
            self._obs_spec[key] = specs.BoundedArray(
                shape=initial_obs[key].shape,
                dtype=np.uint8,
                minimum=0,
                maximum=255,
                name=key,
            )
        self._obs_spec["proprioceptive"] = specs.BoundedArray(
            shape=(self._max_state_dim,),
            dtype=np.float32,
            minimum=-np.inf,
            maximum=np.inf,
            name="proprioceptive",
        )
        self._obs_spec["features"] = specs.BoundedArray(
            shape=(self._max_state_dim,),
            dtype=np.float32,
            minimum=-np.inf,
            maximum=np.inf,
            name="features",
        )

        self._action_spec = specs.Array(shape=(7,), dtype=np.float32, name="action")

    def _make_pixel_to_camera_map(self, pixel_keys: Sequence[str]) -> Dict[str, str]:
        camera_defaults = ["agentview_image", "robot0_eye_in_hand_image"]
        mapping = {}
        for idx, key in enumerate(pixel_keys):
            mapping[key] = camera_defaults[min(idx, len(camera_defaults) - 1)]
        return mapping

    def _select_init_state(self):
        idx = int(self._rng.integers(0, len(self._init_states)))
        init_state = self._init_states[idx]
        if isinstance(init_state, torch.Tensor):
            init_state = init_state.detach().cpu().numpy()
        return init_state

    def _extract_gripper_command_from_qpos(self, raw_obs):
        """
        Return gripper command in action space:
        +1.0 = closed, -1.0 = open.
        """
        value = raw_obs.get("robot0_gripper_qpos", np.array([0.0], dtype=np.float32))
        value = np.asarray(value).reshape(-1)
        if len(value) == 0:
            return 0.0
        qpos = float(value[0])
        # Panda finger qpos: near 0 => closed, larger magnitude => open.
        return 1.0 if abs(qpos) <= self._gripper_qpos_close_threshold else -1.0

    def _extract_gripper_distance(self):
        robot_points = self._get_robot_points_3d()
        if (
            self._tip_left_local_idx is None
            or self._tip_right_local_idx is None
        ):
            return 0.0
        left = int(self._tip_left_local_idx)
        right = int(self._tip_right_local_idx)
        if robot_points.shape[0] <= max(left, right):
            return 0.0
        return float(np.linalg.norm(robot_points[left] - robot_points[right]))

    def _tip_distance_from_points(self, robot_points):
        if (
            self._tip_left_local_idx is None
            or self._tip_right_local_idx is None
        ):
            return None
        left = int(self._tip_left_local_idx)
        right = int(self._tip_right_local_idx)
        if robot_points.shape[0] <= max(left, right):
            return None
        return float(np.linalg.norm(robot_points[left] - robot_points[right]))

    def _extract_gripper_feature(self, raw_obs):
        if self._gripper_label_mode == "distance":
            return self._extract_gripper_distance()
        return self._extract_gripper_command_from_qpos(raw_obs)

    def _compute_non_tip_alignment_error(self, current_robot_points, target_robot_points):
        rigid_ids = list(self._rigid_local_indices)
        valid_ids = [
            idx
            for idx in rigid_ids
            if idx < target_robot_points.shape[0] and idx < current_robot_points.shape[0]
        ]
        if len(valid_ids) == 0:
            return None
        return float(
            np.linalg.norm(
                current_robot_points[valid_ids] - target_robot_points[valid_ids],
                axis=-1,
            ).mean()
        )

    def _apply_gripper_debounce(self, prev_cmd, desired_cmd):
        if desired_cmd > prev_cmd + 1e-6:
            self._gripper_open_pending = 0
            self._gripper_close_pending += 1
            if (
                self._gripper_close_debounce_steps > 0
                and self._gripper_close_pending < self._gripper_close_debounce_steps
            ):
                return prev_cmd
            self._gripper_close_pending = 0
            return 1.0
        if desired_cmd < prev_cmd - 1e-6:
            self._gripper_close_pending = 0
            self._gripper_open_pending += 1
            if (
                self._gripper_open_debounce_steps > 0
                and self._gripper_open_pending < self._gripper_open_debounce_steps
            ):
                return prev_cmd
            self._gripper_open_pending = 0
            return -1.0
        self._gripper_close_pending = 0
        self._gripper_open_pending = 0
        return prev_cmd

    def _extract_eef_pos(self, raw_obs):
        if "robot0_eef_pos" not in raw_obs:
            raise KeyError(
                "robot0_eef_pos not found in observation. Check LIBERO environment keys."
            )
        return np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)[:3]

    def _extract_eef_rotmat(self, raw_obs):
        if "robot0_eef_quat" in raw_obs:
            quat = np.asarray(raw_obs["robot0_eef_quat"], dtype=np.float32).reshape(-1)[:4]
            return R.from_quat(quat).as_matrix().astype(np.float32)

        if "ee_ori" in raw_obs:
            ori = np.asarray(raw_obs["ee_ori"], dtype=np.float32).reshape(-1)
            if ori.shape[0] >= 3:
                return R.from_rotvec(ori[:3]).as_matrix().astype(np.float32)

        if not self._warned_missing_eef_quat:
            print(
                "[libero_object] warning: missing eef orientation key, "
                "using identity rotation fallback."
            )
            self._warned_missing_eef_quat = True
        return np.eye(3, dtype=np.float32)

    def _build_features(self, raw_obs):
        features = np.zeros((self._max_state_dim,), dtype=np.float32)
        eef_pos = self._extract_eef_pos(raw_obs)
        num_pos = min(len(eef_pos), max(0, self._max_state_dim - 1))
        if num_pos > 0:
            features[:num_pos] = eef_pos[:num_pos]
        features[-1] = self._extract_gripper_feature(raw_obs)
        return features

    def _refresh_name_caches(self):
        if self._site_names is not None and self._body_names is not None:
            return
        model = self._env.sim.model
        self._site_names = [_to_str_name(name) for name in model.site_names]
        self._body_names = [_to_str_name(name) for name in model.body_names]
        self._site_name_to_id = {name: idx for idx, name in enumerate(self._site_names)}
        self._body_name_to_id = {name: idx for idx, name in enumerate(self._body_names)}

    def _match_site_suffix(self, suffix, preferred_prefix=None):
        self._refresh_name_caches()
        candidates = []
        for name in self._site_names:
            if name == suffix or name.endswith(f"_{suffix}") or name.endswith(suffix):
                candidates.append(name)
        if not candidates:
            return None

        def _rank(name):
            return (
                0 if preferred_prefix is not None and name.startswith(preferred_prefix) else 1,
                0 if preferred_prefix is not None and name == f"{preferred_prefix}{suffix}" else 1,
                len(name),
                name,
            )

        return sorted(candidates, key=_rank)[0]

    def _resolve_robot_site_ids(self):
        self._refresh_name_caches()
        resolved_names = [None for _ in ROBOT_SITE_SUFFIXES]
        required_global_ids = set(self._robot_point_indices)
        missing = []
        for global_idx, suffix in enumerate(ROBOT_SITE_SUFFIXES):
            if global_idx not in required_global_ids:
                continue
            name = self._match_site_suffix(suffix, preferred_prefix="gripper0_")
            if name is None:
                missing.append(suffix)
            else:
                resolved_names[global_idx] = name

        if missing:
            raise KeyError(
                "Missing robot keypoint sites in simulator model: "
                f"{missing}. Please ensure panda_gripper.xml with kp0~kp8 is loaded."
            )

        self._robot_site_ids = [
            self._site_name_to_id[resolved_names[idx]]
            for idx in self._robot_point_indices
        ]

    def _resolve_object_names(self, raw_obs):
        if self._target_object_name is not None and self._basket_object_name is not None:
            return

        obj_names = list(self._obj_of_interest)
        if raw_obs is not None:
            from_obs = sorted(
                {
                    key[: -len("_pos")]
                    for key in raw_obs.keys()
                    if key.endswith("_pos") and not key.startswith("robot0_")
                }
            )
            if len(obj_names) == 0:
                obj_names = from_obs

        basket_candidates = [name for name in obj_names if "basket" in name]
        target_candidates = [name for name in obj_names if "basket" not in name]

        if len(basket_candidates) > 0:
            self._basket_object_name = basket_candidates[0]
        if len(target_candidates) > 0:
            self._target_object_name = target_candidates[0]

        if self._target_object_name is None or self._basket_object_name is None:
            raise ValueError(
                f"Could not resolve target/basket object names from obj_of_interest={obj_names}"
            )

    def _resolve_body_id(self, object_name):
        self._refresh_name_caches()
        preferred = f"{object_name}_main"
        if preferred in self._body_name_to_id:
            return self._body_name_to_id[preferred]

        candidates = [
            name
            for name in self._body_names
            if name.startswith(f"{object_name}_") and name.endswith("_main")
        ]
        if len(candidates) == 0:
            raise KeyError(
                f"Could not find body for object '{object_name}'. "
                f"Expected '{preferred}' in model body names."
            )
        return self._body_name_to_id[candidates[0]]

    def _resolve_object_asset_xml(self, category):
        assets_root = Path(get_libero_path("assets"))
        candidates = [
            assets_root / "stable_hope_objects" / category / f"{category}.xml",
            assets_root / "stable_scanned_objects" / category / f"{category}.xml",
        ]
        for path in candidates:
            if path.exists():
                return path
        raise FileNotFoundError(
            f"Could not locate asset xml for category='{category}'. "
            f"Tried: {[str(path) for path in candidates]}"
        )

    def _load_target_object_local_points(self, target_category):
        xml_path = self._resolve_object_asset_xml(target_category)
        root = ET.parse(xml_path).getroot()
        points = []
        for site_name in TARGET_OBJECT_SITE_NAMES:
            site = root.find(f".//site[@name='{site_name}']")
            if site is None:
                return TARGET_OBJECT_DEFAULT_LOCAL_POINTS.copy()
            points.append(_parse_vec(site.get("pos"), 3, [0.0, 0.0, 0.0]))
        return np.stack(points, axis=0).astype(np.float32)

    def _load_basket_local_corners(self, basket_category):
        xml_path = self._resolve_object_asset_xml(basket_category)
        root = ET.parse(xml_path).getroot()
        contain_site = root.find(".//site[@name='contain_region']")
        if contain_site is None:
            raise KeyError(
                f"contain_region site not found in basket xml: {xml_path}"
            )

        contain_pos = _parse_vec(contain_site.get("pos"), 3, [0.0, 0.0, 0.0])
        contain_size = _parse_vec(contain_site.get("size"), 3, [0.05, 0.05, 0.05])
        contain_quat_wxyz = _parse_vec(
            contain_site.get("quat"), 4, [1.0, 0.0, 0.0, 0.0]
        )
        contain_quat_xyzw = _quat_wxyz_to_xyzw(contain_quat_wxyz)
        contain_rot = R.from_quat(contain_quat_xyzw).as_matrix().astype(np.float32)

        sx, sy, sz = contain_size
        offsets = np.array(
            [[sx, sy, sz], [sx, -sy, sz], [-sx, sy, sz], [-sx, -sy, sz]],
            dtype=np.float32,
        )
        corners = contain_pos[None, :] + (offsets @ contain_rot.T)
        return corners.astype(np.float32)

    def _subtree_body_ids(self, root_body_id):
        body_parentid = np.asarray(self._env.sim.model.body_parentid, dtype=np.int32).reshape(
            -1
        )
        subtree = []
        for body_id in range(len(body_parentid)):
            cur = int(body_id)
            while cur >= 0:
                if cur == int(root_body_id):
                    subtree.append(body_id)
                    break
                parent = int(body_parentid[cur])
                if parent == cur:
                    break
                cur = parent
        return np.asarray(sorted(set(subtree)), dtype=np.int32)

    def _collect_object_geom_ids(self, body_id):
        geom_bodyid = np.asarray(self._env.sim.model.geom_bodyid, dtype=np.int32).reshape(-1)
        subtree_ids = self._subtree_body_ids(body_id)
        geom_ids = np.where(np.isin(geom_bodyid, subtree_ids))[0].astype(np.int32)
        if geom_ids.size == 0:
            raise ValueError(f"No geoms found for body subtree rooted at id={body_id}")
        return geom_ids

    def _target_axis_points_from_geoms(self, geom_points_world):
        geom_points = np.asarray(geom_points_world, dtype=np.float32).reshape(-1, 3)
        if geom_points.shape[0] == 0:
            raise ValueError("No target geoms available to compute axis points.")
        if geom_points.shape[0] == 1:
            p = geom_points[0]
            return np.stack([p, p, p], axis=0).astype(np.float32)

        center = geom_points.mean(axis=0, keepdims=True)
        centered = geom_points - center
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        axis = np.asarray(vh[0], dtype=np.float32).reshape(3)
        dominant = int(np.argmax(np.abs(axis)))
        if axis[dominant] < 0:
            axis = -axis

        proj = centered @ axis
        min_proj = float(np.min(proj))
        max_proj = float(np.max(proj))
        center_1d = center.reshape(3)
        bottom = center_1d + min_proj * axis
        top = center_1d + max_proj * axis
        mid = center_1d
        return np.stack([bottom, top, mid], axis=0).astype(np.float32)

    def _ensure_geometry_metadata(self, raw_obs):
        if self._use_robot_points and self._robot_site_ids is None:
            self._resolve_robot_site_ids()

        if self._use_object_points and (
            self._target_local_points is None or self._basket_local_corners is None
        ):
            self._resolve_object_names(raw_obs)
            target_category = _instance_to_category(self._target_object_name)
            basket_category = _instance_to_category(self._basket_object_name)
            self._target_local_points = self._load_target_object_local_points(
                target_category
            )
            self._basket_local_corners = self._load_basket_local_corners(
                basket_category
            )
            self._target_body_id = self._resolve_body_id(self._target_object_name)
            self._basket_body_id = self._resolve_body_id(self._basket_object_name)

    def _transform_local_points(self, body_pos, body_quat_xyzw, local_points):
        body_rot = R.from_quat(body_quat_xyzw).as_matrix().astype(np.float32)
        return body_pos[None, :] + (local_points @ body_rot.T)

    def _extract_body_pose(self, raw_obs, object_name, body_id):
        # Keep object pose source consistent with dataset conversion:
        # use simulator body pose (xpos + xquat[wxyz]) and convert to xyzw.
        pos = np.asarray(self._env.sim.data.body_xpos[body_id], dtype=np.float32).reshape(-1)[:3]
        quat_wxyz = np.asarray(
            self._env.sim.data.body_xquat[body_id], dtype=np.float32
        ).reshape(-1)[:4]
        quat = _quat_wxyz_to_xyzw(quat_wxyz)
        return pos, quat

    def _get_robot_points_3d(self):
        if not self._use_robot_points:
            return np.zeros((0, 3), dtype=np.float32)
        if self._robot_site_ids is None:
            self._resolve_robot_site_ids()
        points = np.asarray(
            self._env.sim.data.site_xpos[self._robot_site_ids], dtype=np.float32
        )
        points = points.reshape(-1, 3)
        if points.shape[0] != self._num_robot_points:
            raise ValueError(
                "Robot point count mismatch: "
                f"expected {self._num_robot_points}, got {points.shape[0]}"
            )
        return points

    def _get_object_points_3d(self, raw_obs):
        if not self._use_object_points:
            return np.zeros((0, 3), dtype=np.float32)
        self._ensure_geometry_metadata(raw_obs)

        target_pos, target_quat = self._extract_body_pose(
            raw_obs, self._target_object_name, self._target_body_id
        )
        basket_pos, basket_quat = self._extract_body_pose(
            raw_obs, self._basket_object_name, self._basket_body_id
        )

        target_points = self._transform_local_points(
            target_pos, target_quat, self._target_local_points
        )
        basket_points = self._transform_local_points(
            basket_pos, basket_quat, self._basket_local_corners
        )
        points = np.concatenate([basket_points, target_points], axis=0).astype(np.float32)
        if points.shape[0] != self._num_object_points:
            raise ValueError(
                f"Object point count mismatch: expected {self._num_object_points}, got {points.shape[0]}"
            )
        return points

    def _build_observation(self, raw_obs, goal_achieved):
        self._ensure_geometry_metadata(raw_obs)

        observation = {}
        robot_tracks = self._get_robot_points_3d()
        object_tracks = self._get_object_points_3d(raw_obs)
        tracks = np.concatenate([robot_tracks, object_tracks], axis=0).astype(np.float32)

        for pixel_key in self._pixel_keys:
            camera_key = self._pixel_to_camera[pixel_key]
            if camera_key not in raw_obs:
                raise KeyError(
                    f"{camera_key} not found in observation for pixel_key={pixel_key}"
                )
            observation[pixel_key] = raw_obs[camera_key]
            observation[f"point_tracks_{pixel_key}"] = tracks.copy()

        features = self._build_features(raw_obs)
        observation["proprioceptive"] = features
        observation["features"] = features
        observation["goal_achieved"] = bool(goal_achieved)
        return observation

    def reset(self, **kwargs):
        self._step = 0
        raw_obs = self._env.reset()
        raw_obs = self._env.set_init_state(self._select_init_state())

        # settle simulation for a few frames
        for _ in range(5):
            raw_obs, _, _, _ = self._env.step(np.zeros((7,), dtype=np.float32))

        self._current_raw_obs = raw_obs
        if self._gripper_label_mode == "distance":
            init_distance = self._extract_gripper_distance()
            if init_distance <= self._gripper_distance_close_threshold:
                self._prev_gripper_state = 1.0
            elif init_distance >= self._gripper_distance_open_threshold:
                self._prev_gripper_state = -1.0
            else:
                # In hysteresis gap, fallback to observed qpos-based command.
                self._prev_gripper_state = self._extract_gripper_command_from_qpos(raw_obs)
        else:
            self._prev_gripper_state = self._extract_gripper_command_from_qpos(raw_obs)
        self._close_approach_remaining = 0
        self._gripper_hold_remaining = 0
        self._gripper_close_pending = 0
        self._gripper_open_pending = 0
        self._gripper_align_close_pending = 0
        observation = self._build_observation(raw_obs, goal_achieved=False)
        self.observation = observation
        return observation

    def point2action(self, action):
        pixel_key = self._pixel_keys[0]
        action_key = f"future_tracks_{pixel_key}"
        if action_key not in action:
            raise KeyError(f"{action_key} missing from policy action")

        future_tracks = np.asarray(action[action_key], dtype=np.float32)
        if future_tracks.ndim == 3:
            future_tracks = future_tracks[0]
        if future_tracks.ndim != 2 or future_tracks.shape[1] < 3:
            raise ValueError(
                f"{action_key} must have shape (N, >=3), got {future_tracks.shape}"
            )

        current_robot_points = self._get_robot_points_3d()
        target_robot_points = future_tracks[: self._num_robot_points, :3]
        if target_robot_points.shape[0] != self._num_robot_points:
            raise ValueError(
                f"Insufficient robot points in action: expected {self._num_robot_points}, "
                f"got {target_robot_points.shape[0]}"
            )

        current_eef_pos = self._extract_eef_pos(self._current_raw_obs)
        current_eef_rot = self._extract_eef_rotmat(self._current_raw_obs)

        try:
            # Use rigid body keypoints for pose solve; tip points (kp1, kp2) include
            # gripper articulation and can corrupt rigid transform estimation.
            rigid_ids = list(self._rigid_local_indices)
            valid_ids = [
                idx
                for idx in rigid_ids
                if idx < current_robot_points.shape[0]
                and idx < target_robot_points.shape[0]
            ]
            src_points = current_robot_points
            dst_points = target_robot_points
            if len(valid_ids) >= 3:
                src_points = current_robot_points[valid_ids]
                dst_points = target_robot_points[valid_ids]
            rot, trans = rigid_transform_3D(src_points, dst_points)
            rot = np.asarray(rot, dtype=np.float32)
            trans = np.asarray(trans, dtype=np.float32).reshape(3)
            target_eef_pos = rot @ current_eef_pos + trans
            delta_pos = target_eef_pos - current_eef_pos
            target_eef_rot = rot @ current_eef_rot
            delta_rot = R.from_matrix(target_eef_rot @ current_eef_rot.T).as_rotvec()
        except Exception:
            # Fallback to kp0 translation only if rigid transform fails
            target_pos = target_robot_points[0]
            delta_pos = target_pos - current_eef_pos
            delta_rot = np.zeros((3,), dtype=np.float32)

        delta_pos = np.clip(delta_pos, -self._max_delta_pos, self._max_delta_pos)
        delta_rot = np.clip(delta_rot, -self._max_delta_rot, self._max_delta_rot)
        delta_rot = np.nan_to_num(delta_rot, nan=0.0, posinf=0.0, neginf=0.0)

        if "gripper" in action:
            raw_gripper_value = float(np.asarray(action["gripper"]).reshape(-1)[0])
        elif "future_gripper_states" in action:
            raw_gripper_value = float(
                np.asarray(action["future_gripper_states"]).reshape(-1)[0]
            )
        else:
            raw_gripper_value = self._prev_gripper_state

        gripper_value = raw_gripper_value
        prev_cmd = 1.0 if float(self._prev_gripper_state) >= 0.0 else -1.0
        desired_cmd = prev_cmd
        if self._gripper_label_mode == "distance":
            if (
                self._gripper_distance_value_source == "pred_points"
            ):
                pred_tip_distance = self._tip_distance_from_points(target_robot_points)
                if pred_tip_distance is not None:
                    gripper_value = pred_tip_distance
            # Predicted value is finger-tip distance in meters.
            if self._gripper_distance_control_mode == "target":
                # Track predicted distance with deadband, but gate switching with
                # close/open thresholds to avoid premature closure.
                current_distance = self._extract_gripper_distance()
                close_gate = gripper_value <= self._gripper_distance_close_threshold
                open_gate = gripper_value >= self._gripper_distance_open_threshold
                if prev_cmd < 0.0:
                    if (
                        close_gate
                        and current_distance
                        > gripper_value + self._gripper_distance_deadband
                    ):
                        desired_cmd = 1.0
                    else:
                        desired_cmd = -1.0
                else:
                    # Be permissive for reopening when predicted target distance
                    # is larger than the close threshold to avoid sticky closure.
                    if (
                        gripper_value >= self._gripper_distance_close_threshold
                        and current_distance
                        < gripper_value - self._gripper_distance_deadband
                    ):
                        desired_cmd = -1.0
                    elif (
                        close_gate
                        and current_distance
                        > gripper_value + self._gripper_distance_deadband
                    ):
                        desired_cmd = 1.0
                    else:
                        desired_cmd = prev_cmd
            else:
                if (
                    prev_cmd < 0.0
                    and gripper_value <= self._gripper_distance_close_threshold
                ):
                    desired_cmd = 1.0
                elif (
                    prev_cmd > 0.0
                    and gripper_value >= self._gripper_distance_open_threshold
                ):
                    desired_cmd = -1.0
        else:
            # Predicted value is command-like scalar.
            gripper_value = gripper_value + self._gripper_bias
            if prev_cmd < 0.0 and gripper_value > self._gripper_close_threshold:
                desired_cmd = 1.0
            elif prev_cmd > 0.0 and gripper_value < self._gripper_open_threshold:
                desired_cmd = -1.0
            if (
                self._gripper_command_fallback_to_tip_distance
            ):
                pred_tip_distance = self._tip_distance_from_points(target_robot_points)
                if pred_tip_distance is not None:
                    if (
                        prev_cmd < 0.0
                        and pred_tip_distance <= self._gripper_tip_close_threshold
                    ):
                        desired_cmd = 1.0
                    elif (
                        prev_cmd > 0.0
                        and pred_tip_distance >= self._gripper_tip_open_threshold
                    ):
                        desired_cmd = -1.0

        align_err = self._compute_non_tip_alignment_error(
            current_robot_points, target_robot_points
        )
        if self._gripper_force_close_on_align and prev_cmd < 0.0:
            if (
                align_err is not None
                and align_err <= self._gripper_close_position_threshold
            ):
                self._gripper_align_close_pending += 1
            else:
                self._gripper_align_close_pending = 0

            required_align_steps = max(1, self._gripper_force_close_align_steps)
            if self._gripper_align_close_pending >= required_align_steps:
                desired_cmd = 1.0
        else:
            self._gripper_align_close_pending = 0

        desired_cmd = self._apply_gripper_debounce(prev_cmd, desired_cmd)

        # Keep command for a minimum number of steps after switching to prevent
        # rapid open/close chattering around the decision boundary.
        if self._gripper_hold_remaining > 0:
            gripper_cmd = prev_cmd
            self._gripper_hold_remaining -= 1
        else:
            gripper_cmd = desired_cmd
            if (
                self._gripper_min_hold_steps > 0
                and abs(gripper_cmd - prev_cmd) > 1e-6
            ):
                self._gripper_hold_remaining = self._gripper_min_hold_steps
        if self._gripper_close_position_gate and gripper_cmd > 0.0:
            if (
                align_err is not None
                and align_err > self._gripper_close_position_threshold
            ):
                gripper_cmd = -1.0
                self._gripper_hold_remaining = 0

        prev_cmd_cont = float(self._prev_gripper_state)
        if self._gripper_cmd_slew_rate < 2.0:
            delta_cmd = float(gripper_cmd) - prev_cmd_cont
            delta_cmd = float(
                np.clip(delta_cmd, -self._gripper_cmd_slew_rate, self._gripper_cmd_slew_rate)
            )
            gripper_cmd = prev_cmd_cont + delta_cmd
        gripper_cmd = float(np.clip(gripper_cmd, -1.0, 1.0))
        self._prev_gripper_state = gripper_cmd

        # Optional grasp compensation: when switching to close, move slightly
        # along the gripper forward direction and downward for a few steps.
        if prev_cmd < 0.0 and gripper_cmd > 0.0 and self._close_approach_steps > 0:
            self._close_approach_remaining = self._close_approach_steps
        if self._close_approach_remaining > 0:
            if self._close_approach_offset > 0.0 and target_robot_points.shape[0] > 4:
                approach_dir = target_robot_points[4] - target_robot_points[0]
                norm = float(np.linalg.norm(approach_dir))
                if norm > 1e-6:
                    delta_pos = delta_pos + (
                        approach_dir / norm
                    ) * self._close_approach_offset
            if self._close_downward_offset > 0.0:
                delta_pos[2] = delta_pos[2] - self._close_downward_offset
            self._close_approach_remaining -= 1
            delta_pos = np.clip(delta_pos, -self._max_delta_pos, self._max_delta_pos)

        robot_action = np.zeros((7,), dtype=np.float32)
        robot_action[:3] = delta_pos
        robot_action[3:6] = delta_rot
        robot_action[6] = gripper_cmd
        return robot_action

    def step(self, action):
        self._step += 1
        robot_action = self.point2action(action) if isinstance(action, dict) else action
        robot_action = np.asarray(robot_action, dtype=np.float32).reshape(-1)
        self._last_robot_action = robot_action.copy()

        raw_obs, reward, env_done, info = self._env.step(robot_action)
        self._current_raw_obs = raw_obs

        try:
            goal_achieved = bool(self._env.check_success())
        except Exception:
            goal_achieved = bool(env_done)
        done = bool(goal_achieved) or (self._step >= self._max_episode_len)
        if isinstance(info, dict):
            info["env_done"] = bool(env_done)
            info["goal_achieved"] = bool(goal_achieved)

        observation = self._build_observation(raw_obs, goal_achieved=goal_achieved)
        self.observation = observation
        return observation, reward, done, info

    def render(self, mode="rgb_array", width=256, height=256):
        del mode, width, height
        if self._current_raw_obs is None:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        frame = self._current_raw_obs[self._pixel_to_camera[self._pixel_keys[0]]]
        return frame

    def project_world_points_to_pixel(self, points_world, pixel_key=None):
        if get_camera_transform_matrix is None or project_points_from_world_to_camera is None:
            raise RuntimeError(
                "robosuite camera projection utilities are unavailable in this environment."
            )

        points_world = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
        if points_world.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.int32)

        if pixel_key is None:
            pixel_key = self._pixel_keys[0]
        if pixel_key not in self._pixel_to_camera:
            raise KeyError(f"Unknown pixel_key={pixel_key}. Valid: {self._pixel_keys}")

        camera_obs_key = self._pixel_to_camera[pixel_key]
        camera_name = (
            camera_obs_key[:-6] if camera_obs_key.endswith("_image") else camera_obs_key
        )

        if self._current_raw_obs is not None and camera_obs_key in self._current_raw_obs:
            h, w = self._current_raw_obs[camera_obs_key].shape[:2]
        elif pixel_key in self.observation:
            h, w = self.observation[pixel_key].shape[:2]
        else:
            h, w = 128, 128

        world_to_camera = get_camera_transform_matrix(
            sim=self._env.sim,
            camera_name=camera_name,
            camera_height=h,
            camera_width=w,
        )
        pixels_hw = project_points_from_world_to_camera(
            points=points_world,
            world_to_camera_transform=world_to_camera,
            camera_height=h,
            camera_width=w,
        )
        return np.asarray(np.round(pixels_hw), dtype=np.int32)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._action_spec

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for _ in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += (time_step.reward or 0.0) * discount
            discount *= time_step.discount
            if time_step.last():
                break
        return time_step._replace(reward=reward, discount=discount)

    def reset(self, **kwargs):
        return self._env.reset(**kwargs)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def point2action(self, action):
        return self._env.point2action(action)

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        self._discount = 1.0
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.Array(
            shape=wrapped_action_spec.shape, dtype=dtype, name="action"
        )

    def step(self, action):
        observation, reward, done, info = self._env.step(action)
        step_type = StepType.LAST if done else StepType.MID
        return TimeStep(
            step_type=step_type,
            reward=reward,
            discount=self._discount,
            observation=observation,
        )

    def reset(self, **kwargs):
        obs = self._env.reset(**kwargs)
        return TimeStep(
            step_type=StepType.FIRST, reward=0.0, discount=self._discount, observation=obs
        )

    def point2action(self, action):
        return self._env.point2action(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env

    def reset(self, **kwargs):
        time_step = self._env.reset(**kwargs)
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)
        return ExtendedTimeStep(
            observation=time_step.observation,
            step_type=time_step.step_type,
            action=action,
            reward=time_step.reward or 0.0,
            discount=time_step.discount or 1.0,
        )

    def point2action(self, action):
        return self._env.point2action(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


def _normalize_tasks(task_name):
    if isinstance(task_name, str):
        return [name.strip() for name in task_name.split(",") if name.strip()]
    if isinstance(task_name, (list, tuple)):
        return [str(name).strip() for name in task_name if str(name).strip()]
    raise TypeError("task_name must be a string, list, or tuple")


def make(
    task_name,
    benchmark_name,
    task_order_index,
    action_repeat,
    seed,
    height,
    width,
    max_episode_len,
    max_state_dim,
    pixel_keys,
    eval,  # compatibility with existing suite signatures
    use_robot_points,
    num_robot_points,
    robot_point_indices,
    use_object_points,
    num_object_points,
    point_dim,
    max_delta_pos,
    max_delta_rot,
    gripper_label_mode,
    gripper_threshold,
    gripper_close_threshold,
    gripper_open_threshold,
    gripper_bias,
    gripper_qpos_close_threshold,
    gripper_distance_close_threshold,
    gripper_distance_open_threshold,
    gripper_distance_control_mode="target",
    gripper_distance_value_source="token",
    gripper_distance_deadband=0.003,
    gripper_cmd_slew_rate=2.0,
    gripper_close_position_gate=False,
    gripper_close_position_threshold=0.02,
    real_deploy_mode=False,
    gripper_command_fallback_to_tip_distance=False,
    gripper_tip_close_threshold=0.07,
    gripper_tip_open_threshold=0.08,
    gripper_min_hold_steps=0,
    gripper_close_debounce_steps=0,
    gripper_open_debounce_steps=0,
    gripper_force_close_on_align=False,
    gripper_force_close_align_steps=0,
    close_downward_offset=0.0,
    close_approach_offset=0.0,
    close_approach_steps=0,
):
    if get_benchmark is None or get_libero_path is None or OffScreenRenderEnv is None:
        raise RuntimeError(
            "LIBERO import failed for libero_object suite. "
            f"Import error: {_LIBERO_IMPORT_ERROR}"
        )

    if not use_robot_points:
        raise ValueError("libero_object suite currently requires use_robot_points=true")
    if int(point_dim) != 3:
        raise ValueError("libero_object suite currently supports point_dim=3 only")

    tasks = _normalize_tasks(task_name)
    if len(tasks) == 0:
        raise ValueError("No task name provided")

    benchmark = get_benchmark(benchmark_name)(task_order_index)
    task_lookup = {task.name: task for task in benchmark.tasks}
    missing = [name for name in tasks if name not in task_lookup]
    if missing:
        available = ", ".join(task_lookup.keys())
        raise KeyError(
            f"Unknown task(s): {missing}. Available in {benchmark_name}: {available}"
        )

    envs = []
    task_descriptions = []
    for idx, task_key in enumerate(tasks):
        task = task_lookup[task_key]

        bddl_file_name = os.path.join(
            get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
        )
        init_states_file = os.path.join(
            get_libero_path("init_states"), task.problem_folder, task.init_states_file
        )
        init_states = torch.load(init_states_file)

        base_env = OffScreenRenderEnv(
            bddl_file_name=bddl_file_name,
            gripper_types="PandaGripper",
            camera_heights=height,
            camera_widths=width,
            camera_names=["agentview", "robot0_eye_in_hand"],
            horizon=int(max_episode_len),
            ignore_done=bool(eval),
        )
        base_env.seed(seed + idx)

        env = RGBArrayAsObservationWrapper(
            base_env,
            task_name=task_key,
            init_states=init_states,
            max_episode_len=max_episode_len,
            max_state_dim=max_state_dim,
            pixel_keys=pixel_keys,
            use_robot_points=use_robot_points,
            num_robot_points=num_robot_points,
            robot_point_indices=robot_point_indices,
            use_object_points=use_object_points,
            num_object_points=num_object_points,
            max_delta_pos=max_delta_pos,
            max_delta_rot=max_delta_rot,
            gripper_label_mode=gripper_label_mode,
            gripper_threshold=gripper_threshold,
            gripper_close_threshold=gripper_close_threshold,
            gripper_open_threshold=gripper_open_threshold,
            gripper_bias=gripper_bias,
            gripper_qpos_close_threshold=gripper_qpos_close_threshold,
            gripper_distance_close_threshold=gripper_distance_close_threshold,
            gripper_distance_open_threshold=gripper_distance_open_threshold,
            gripper_distance_control_mode=gripper_distance_control_mode,
            gripper_distance_value_source=gripper_distance_value_source,
            gripper_distance_deadband=gripper_distance_deadband,
            gripper_cmd_slew_rate=gripper_cmd_slew_rate,
            gripper_close_position_gate=gripper_close_position_gate,
            gripper_close_position_threshold=gripper_close_position_threshold,
            real_deploy_mode=real_deploy_mode,
            gripper_command_fallback_to_tip_distance=gripper_command_fallback_to_tip_distance,
            gripper_tip_close_threshold=gripper_tip_close_threshold,
            gripper_tip_open_threshold=gripper_tip_open_threshold,
            gripper_min_hold_steps=gripper_min_hold_steps,
            gripper_close_debounce_steps=gripper_close_debounce_steps,
            gripper_open_debounce_steps=gripper_open_debounce_steps,
            gripper_force_close_on_align=gripper_force_close_on_align,
            gripper_force_close_align_steps=gripper_force_close_align_steps,
            close_downward_offset=close_downward_offset,
            close_approach_offset=close_approach_offset,
            close_approach_steps=close_approach_steps,
            seed=seed + idx,
        )
        env = ActionDTypeWrapper(env, np.float32)
        env = ActionRepeatWrapper(env, action_repeat)
        env = ExtendedTimeStepWrapper(env)
        envs.append(env)
        task_descriptions.append(task.language)

    return envs, task_descriptions
