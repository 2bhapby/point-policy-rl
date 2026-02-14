from __future__ import annotations

import copy
import importlib
import inspect
import sys
import types
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from point_policy_rl.utils_rl import add_point_policy_path


def _suite_make_module_name(suite_name: str) -> str:
    suite_name = str(suite_name).strip().lower()
    if suite_name in {"libero_spatial", "spatial"}:
        return "suite.libero_spatial"
    if suite_name in {"libero_object", "object"}:
        return "suite.libero_object"
    raise ValueError(f"Unsupported suite: {suite_name}")


def _ensure_libero_runtime_compat(repo_root: Path) -> None:
    local_libero_root = (Path(repo_root).resolve() / "LIBERO").resolve()
    if local_libero_root.exists() and str(local_libero_root) not in sys.path:
        sys.path.insert(0, str(local_libero_root))

    try:
        env_wrapper_mod = importlib.import_module("libero.libero.envs.env_wrapper")
        control_cls = getattr(env_wrapper_mod, "ControlEnv", None)
        if control_cls is not None and not bool(getattr(control_cls, "_pp_rl_seed_patched", False)):
            def _seed_compat(self, seed):
                seed_fn = getattr(self.env, "seed", None)
                if callable(seed_fn):
                    seed_fn(seed)
                    return
                np.random.seed(seed)
                if hasattr(self.env, "_seed"):
                    self.env._seed = seed

            control_cls.seed = _seed_compat
            control_cls._pp_rl_seed_patched = True
    except Exception:
        pass

    def _patch_robot_cls(module_name: str, class_name: str) -> None:
        try:
            mod = importlib.import_module(module_name)
            cls = getattr(mod, class_name)
        except Exception:
            return

        if bool(getattr(cls, "_pp_rl_robot_patched", False)):
            return

        if not hasattr(cls, "arms"):
            cls.arms = ["right"]

        # Override to robosuite>=1.5 API expectation regardless of inherited base
        # class property (which may raise NotImplementedError in ManipulatorModel).
        def _default_base(self):
            return getattr(self, "default_mount", None)

        cls.default_base = property(_default_base)

        for prop_name in ("default_gripper", "default_controller_config"):
            prop = getattr(cls, prop_name, None)
            if not isinstance(prop, property):
                continue
            orig_fget = prop.fget
            if orig_fget is None:
                continue

            def _make_dict_wrapper(fget):
                def _wrapped(self):
                    value = fget(self)
                    if isinstance(value, dict):
                        return value
                    return {"right": value}

                return _wrapped

            setattr(cls, prop_name, property(_make_dict_wrapper(orig_fget)))

        cls._pp_rl_robot_patched = True

    _patch_robot_cls("libero.libero.envs.robots.mounted_panda", "MountedPanda")
    _patch_robot_cls("libero.libero.envs.robots.on_the_ground_panda", "OnTheGroundPanda")


def _ensure_suite_runtime_compat(suite_mod) -> None:
    wrapper_cls = getattr(suite_mod, "RGBArrayAsObservationWrapper", None)
    if wrapper_cls is None or bool(getattr(wrapper_cls, "_pp_rl_site_fallback_patched", False)):
        return

    orig_resolve_robot_site_ids = getattr(wrapper_cls, "_resolve_robot_site_ids", None)
    if orig_resolve_robot_site_ids is None:
        return

    robot_suffixes = tuple(getattr(suite_mod, "ROBOT_SITE_SUFFIXES", ()))

    def _resolve_robot_site_ids_compat(self):
        try:
            return orig_resolve_robot_site_ids(self)
        except Exception as exc:
            warnings.warn(
                "Robot keypoint site resolution failed; "
                "fallback to pose-reconstructed robot points (cal_offset). "
                f"reason={exc}",
                RuntimeWarning,
            )
            self._robot_point_mode = "cal_offset"
            if getattr(self, "_robot_site_ids_full", None) is None:
                self._robot_site_ids_full = list(range(len(robot_suffixes)))
            self._robot_site_ids = []
            return None

    wrapper_cls._resolve_robot_site_ids = _resolve_robot_site_ids_compat
    wrapper_cls._pp_rl_site_fallback_patched = True


def _ensure_robosuite_single_arm_compat() -> None:
    """
    Provide a lightweight compatibility shim for environments that only ship
    robosuite>=1.5 (where single_arm_env module was removed).
    """
    module_name = "robosuite.environments.manipulation.single_arm_env"
    has_single_arm_env = True
    try:
        importlib.import_module(module_name)
    except Exception:
        has_single_arm_env = False

    try:
        robosuite_mod = importlib.import_module("robosuite")
        if not hasattr(robosuite_mod, "load_controller_config"):
            part_factory_mod = importlib.import_module("robosuite.controllers.parts.controller_factory")
            composite_factory_mod = importlib.import_module(
                "robosuite.controllers.composite.composite_controller_factory"
            )
            load_part_controller_config = getattr(part_factory_mod, "load_part_controller_config")
            refactor_composite_controller_config = getattr(
                composite_factory_mod, "refactor_composite_controller_config"
            )

            def _load_controller_config_compat(default_controller=None, custom_fpath=None):
                part_cfg = load_part_controller_config(
                    custom_fpath=custom_fpath,
                    default_controller=default_controller,
                )
                if isinstance(part_cfg, dict) and "type" in part_cfg and "body_parts" in part_cfg:
                    return part_cfg
                return refactor_composite_controller_config(
                    controller_config=part_cfg,
                    robot_type="Panda",
                    arms=["right"],
                )

            robosuite_mod.load_controller_config = _load_controller_config_compat
    except Exception:
        pass

    if not has_single_arm_env:
        try:
            manip_mod = importlib.import_module("robosuite.environments.manipulation.manipulation_env")
            manipulation_pkg = importlib.import_module("robosuite.environments.manipulation")
            ManipulationEnv = getattr(manip_mod, "ManipulationEnv")
        except Exception:
            return

        class SingleArmEnv(ManipulationEnv):
            def __init__(self, *args, mount_types="default", **kwargs):
                # robosuite<=1.4 used mount_types; >=1.5 uses base_types.
                kwargs.setdefault("base_types", mount_types)
                super().__init__(*args, **kwargs)

        shim = types.ModuleType(module_name)
        shim.SingleArmEnv = SingleArmEnv
        sys.modules[module_name] = shim
        setattr(manipulation_pkg, "single_arm_env", shim)

    # LIBERO also imports legacy robot class path:
    # `from robosuite.robots.single_arm import SingleArm`
    robot_module_name = "robosuite.robots.single_arm"
    if robot_module_name in sys.modules:
        return
    try:
        robots_pkg = importlib.import_module("robosuite.robots")
        fixed_base_mod = importlib.import_module("robosuite.robots.fixed_base_robot")
        FixedBaseRobot = getattr(fixed_base_mod, "FixedBaseRobot")
    except Exception:
        return

    class SingleArm(FixedBaseRobot):
        pass

    robot_shim = types.ModuleType(robot_module_name)
    robot_shim.SingleArm = SingleArm
    sys.modules[robot_module_name] = robot_shim
    setattr(robots_pkg, "single_arm", robot_shim)


def _extract_task_block(cfg: dict[str, Any]) -> dict[str, Any]:
    suite_cfg = cfg.get("suite", {}) if isinstance(cfg, dict) else {}
    task = suite_cfg.get("task", {}) if isinstance(suite_cfg, dict) else {}
    if not isinstance(task, dict):
        task = {}
    return task


def _extract_suite_block(cfg: dict[str, Any]) -> dict[str, Any]:
    suite_cfg = cfg.get("suite", {}) if isinstance(cfg, dict) else {}
    if not isinstance(suite_cfg, dict):
        raise ValueError("Missing suite block in config")
    return suite_cfg


def _build_make_kwargs(
    cfg: dict[str, Any],
    suite_override: str | None,
    task_override: str | None,
    seed: int,
    eval_mode: bool,
    max_episode_len: int,
) -> tuple[str, dict[str, Any]]:
    suite_cfg = copy.deepcopy(_extract_suite_block(cfg))
    task_cfg = copy.deepcopy(_extract_task_block(cfg))

    suite_name = str(suite_override or suite_cfg.get("name") or suite_cfg.get("suite") or "").strip()
    if suite_name == "":
        raise ValueError("Cannot infer suite name from config")

    task_name = task_override or task_cfg.get("task_name")
    if task_name is None:
        raise ValueError("task_name is required (missing in config and no override provided)")

    img_size = suite_cfg.get("img_size", [128, 128])
    width = int(img_size[0])
    height = int(img_size[1])

    num_robot_points = int(suite_cfg.get("num_robot_points", 9))
    point_dim = int(suite_cfg.get("point_dim", 3))
    max_state_dim = int(num_robot_points * point_dim)

    kwargs: dict[str, Any] = {
        "task_name": task_name,
        "benchmark_name": task_cfg.get("benchmark_name"),
        "task_order_index": int(task_cfg.get("task_order_index", 0)),
        "action_repeat": int(suite_cfg.get("action_repeat", 1)),
        "seed": int(seed),
        "height": height,
        "width": width,
        "max_episode_len": int(max_episode_len),
        "max_state_dim": max_state_dim,
        "pixel_keys": suite_cfg.get("pixel_keys", ["pixels1", "pixels2"]),
        "eval": bool(eval_mode),
    }

    # Backfill required suite.make args that may not exist in older BC hydra configs.
    kwargs.setdefault("robot_point_indices", None)
    kwargs.setdefault("object_point_mode", "geom_fps")
    kwargs.setdefault("pose_delta_gain", 1.0)
    kwargs.setdefault("pose_solve_mode", "rigid")

    # Copy every suite key as potential make arg; unknown ones are dropped by signature filter.
    for key, value in suite_cfg.items():
        kwargs.setdefault(key, value)

    # Copy task values as fallbacks for direct make args.
    for key, value in task_cfg.items():
        kwargs.setdefault(key, value)

    return suite_name, kwargs


def build_single_env_from_bc_config(
    cfg: dict[str, Any],
    repo_root: Path,
    suite_override: str | None,
    task_override: str | None,
    seed: int,
    eval_mode: bool,
    max_episode_len: int,
):
    _ensure_robosuite_single_arm_compat()
    add_point_policy_path(repo_root)
    _ensure_libero_runtime_compat(repo_root)

    suite_name, kwargs = _build_make_kwargs(
        cfg=cfg,
        suite_override=suite_override,
        task_override=task_override,
        seed=seed,
        eval_mode=eval_mode,
        max_episode_len=max_episode_len,
    )

    mod_name = _suite_make_module_name(suite_name)
    suite_mod = importlib.import_module(mod_name)
    _ensure_suite_runtime_compat(suite_mod)
    make_fn = getattr(suite_mod, "make")

    sig = inspect.signature(make_fn)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    envs, task_descriptions = make_fn(**filtered)
    if len(envs) == 0:
        raise RuntimeError("suite.make returned no environments")

    suite_cfg = _extract_suite_block(cfg)
    pixel_keys = list(suite_cfg.get("pixel_keys", ["pixels1", "pixels2"]))
    pixel_key = pixel_keys[0]

    max_delta_pos = float(suite_cfg.get("max_delta_pos", 0.05))
    max_delta_rot = float(suite_cfg.get("max_delta_rot", 0.25))
    low = np.array(
        [-max_delta_pos, -max_delta_pos, -max_delta_pos, -max_delta_rot, -max_delta_rot, -max_delta_rot, -1.0],
        dtype=np.float32,
    )
    high = np.array(
        [max_delta_pos, max_delta_pos, max_delta_pos, max_delta_rot, max_delta_rot, max_delta_rot, 1.0],
        dtype=np.float32,
    )

    env = envs[0]
    task_desc = task_descriptions[0] if len(task_descriptions) > 0 else ""
    return env, task_desc, pixel_key, low, high, suite_name


def observation_to_state(
    obs: dict[str, Any],
    pixel_key: str,
    base_action_7d: np.ndarray,
    include_eef_pos: bool = False,
) -> np.ndarray:
    point_key = f"point_tracks_{pixel_key}"
    if point_key not in obs:
        raise KeyError(f"Missing point key in observation: {point_key}")

    points = np.asarray(obs[point_key], dtype=np.float32).reshape(-1)
    features = np.asarray(obs.get("features", np.zeros((1,), dtype=np.float32)), dtype=np.float32).reshape(-1)
    if features.size == 0:
        gripper = np.array([0.0], dtype=np.float32)
        eef_pos = np.zeros((3,), dtype=np.float32)
    else:
        gripper = np.array([float(features[-1])], dtype=np.float32)
        if features.size >= 3:
            eef_pos = features[:3].astype(np.float32)
        else:
            eef_pos = np.zeros((3,), dtype=np.float32)

    base_action_7d = np.asarray(base_action_7d, dtype=np.float32).reshape(7)
    parts = [points, gripper]
    if include_eef_pos:
        parts.append(eef_pos)
    parts.append(base_action_7d)
    return np.concatenate(parts, axis=0).astype(np.float32)


def clip_action(action: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    action = np.asarray(action, dtype=np.float32)
    return np.clip(action, low, high).astype(np.float32)
