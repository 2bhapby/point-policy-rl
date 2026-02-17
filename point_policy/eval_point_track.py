#!/usr/bin/env python3

import warnings
import os
import re
import shutil

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from pathlib import Path

import hydra
import torch
import numpy as np
import cv2
from hydra.core.hydra_config import HydraConfig

import utils
from logger import Logger
from replay_buffer import make_expert_replay_loader
from video import VideoRecorder
from robot_utils.franka.utils import rigid_transform_3D

warnings.filterwarnings("ignore", category=DeprecationWarning)
torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    obs_shape = {}
    for key in cfg.suite.pixel_keys:
        obs_shape[key] = obs_spec[key].shape
    if cfg.use_proprio:
        obs_shape[cfg.suite.proprio_key] = obs_spec[cfg.suite.proprio_key].shape
    obs_shape[cfg.suite.feature_key] = obs_spec[cfg.suite.feature_key].shape
    cfg.agent.obs_shape = obs_shape
    cfg.agent.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg.agent)


class Workspace:
    def __init__(self, cfg):
        try:
            self.work_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
        except Exception:
            self.work_dir = Path.cwd().resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # load data
        dataset_iterable = hydra.utils.call(self.cfg.expert_dataset)
        self.expert_replay_loader = make_expert_replay_loader(
            dataset_iterable, self.cfg.batch_size
        )
        self.expert_replay_iter = iter(self.expert_replay_loader)

        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        dataset_episode_len = self.expert_replay_loader.dataset._max_episode_len
        override_episode_len = int(getattr(self.cfg, "eval_max_episode_len", -1))
        if override_episode_len > 0:
            self.cfg.suite.task_make_fn.max_episode_len = max(
                int(dataset_episode_len), override_episode_len
            )
        else:
            self.cfg.suite.task_make_fn.max_episode_len = int(dataset_episode_len)
        print(
            "[eval] max_episode_len="
            f"{self.cfg.suite.task_make_fn.max_episode_len} "
            f"(dataset={dataset_episode_len}, override={override_episode_len})"
        )
        self.cfg.suite.task_make_fn.max_state_dim = (
            self.expert_replay_loader.dataset._max_state_dim
        )

        try:
            if self.cfg.suite.use_object_points:
                import yaml

                cfg_path = f"{cfg.root_dir}/point_policy/cfgs/suite/points_cfg.yaml"
                with open(cfg_path) as stream:
                    try:
                        points_cfg = yaml.safe_load(stream)
                    except yaml.YAMLError as exc:
                        print(exc)
                    root_dir, dift_path, cotracker_checkpoint = (
                        points_cfg["root_dir"],
                        points_cfg["dift_path"],
                        points_cfg["cotracker_checkpoint"],
                    )
                    points_cfg["dift_path"] = f"{root_dir}/{dift_path}"
                    points_cfg[
                        "cotracker_checkpoint"
                    ] = f"{root_dir}/{cotracker_checkpoint}"
                self.cfg.suite.task_make_fn.points_cfg = points_cfg
        except:
            pass

        self.env, self.task_descriptions = hydra.utils.call(self.cfg.suite.task_make_fn)

        # create agent
        self.agent = make_agent(
            self.env[0].observation_spec(), self.env[0].action_spec(), cfg
        )

        self.envs_till_idx = len(self.env)
        self.expert_replay_loader.dataset.envs_till_idx = self.envs_till_idx
        self.expert_replay_iter = iter(self.expert_replay_loader)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None,
            render_size=int(getattr(self.cfg, "video_render_size", 256)),
            flip_vertical=bool(getattr(self.cfg, "video_flip_vertical", False)),
        )
        self.point_overlay = bool(getattr(self.cfg, "point_overlay", True))
        self.point_overlay_side_by_side = bool(
            getattr(self.cfg, "point_overlay_side_by_side", False)
        )
        self.point_overlay_radius = int(getattr(self.cfg, "point_overlay_radius", 3))
        self.point_debug = bool(getattr(self.cfg, "point_debug", False))
        self.point_debug_max_steps = int(getattr(self.cfg, "point_debug_max_steps", 20))
        self.contact_debug = bool(getattr(self.cfg, "contact_debug", False))
        self.contact_debug_max_steps = int(
            getattr(self.cfg, "contact_debug_max_steps", 200)
        )
        self.eval_rollout_steps = int(getattr(self.cfg, "eval_rollout_steps", -1))
        if self.eval_rollout_steps > 0:
            print(f"[eval] fixed rollout steps per episode={self.eval_rollout_steps}")
        self.video_save_per_episode = bool(
            getattr(self.cfg, "video_save_per_episode", False)
        )
        self.video_chunk_steps = int(getattr(self.cfg, "video_chunk_steps", -1))
        if self.video_chunk_steps <= 0 and self.eval_rollout_steps > 0:
            self.video_chunk_steps = int(self.eval_rollout_steps)
        if self.video_save_per_episode:
            print(
                "[eval] video_save_per_episode=true "
                f"(video_chunk_steps={self.video_chunk_steps})"
            )
        self.video_export_enabled = bool(
            getattr(self.cfg, "video_export_enabled", False)
        )
        video_export_dir = str(getattr(self.cfg, "video_export_dir", "")).strip()
        self.video_export_dir = None
        if self.video_export_enabled and len(video_export_dir) > 0:
            self.video_export_dir = Path(video_export_dir).expanduser().resolve()
            self.video_export_dir.mkdir(parents=True, exist_ok=True)
        self.video_export_tag = str(getattr(self.cfg, "video_export_tag", "")).strip()
        self.rigidify_pred_points = bool(
            getattr(self.cfg, "rigidify_pred_points", False)
        )
        self.rigidify_pred_points_all_robot = bool(
            getattr(self.cfg, "rigidify_pred_points_all_robot", True)
        )
        if self.rigidify_pred_points:
            print(
                "[eval] rigidify_pred_points=true "
                f"(all_robot={self.rigidify_pred_points_all_robot})"
            )

    @staticmethod
    def _to_name_list(names):
        out = []
        for n in names:
            if isinstance(n, bytes):
                out.append(n.decode("utf-8", errors="ignore"))
            else:
                out.append(str(n))
        return out

    @staticmethod
    def _resolve_sim_model_data(sim):
        if sim is None:
            return None, None
        model = getattr(sim, "model", None)
        data = getattr(sim, "data", None)
        if model is None:
            model = getattr(sim, "_model", None)
        if data is None:
            data = getattr(sim, "_data", None)
        return model, data

    @staticmethod
    def _safe_name_list_from_ids(model, n, id2name_fn_name, prefix):
        id2name_fn = getattr(model, id2name_fn_name, None)
        out = []
        for i in range(int(n)):
            name = None
            if callable(id2name_fn):
                try:
                    name = id2name_fn(i)
                except Exception:
                    name = None
            if name is None:
                name = f"{prefix}{i}"
            if isinstance(name, bytes):
                name = name.decode("utf-8", errors="ignore")
            out.append(str(name))
        return out

    def _build_contact_debug_ctx(self, env):
        sim = getattr(env, "sim", None)
        model, data = self._resolve_sim_model_data(sim)
        if model is None or data is None:
            return None

        if hasattr(model, "body_names"):
            body_names = self._to_name_list(model.body_names)
        else:
            body_names = self._safe_name_list_from_ids(
                model, getattr(model, "nbody", 0), "body_id2name", "body"
            )
        if hasattr(model, "geom_names"):
            geom_names = self._to_name_list(model.geom_names)
        else:
            geom_names = self._safe_name_list_from_ids(
                model, getattr(model, "ngeom", 0), "geom_id2name", "geom"
            )

        finger_body_ids = {
            idx
            for idx, name in enumerate(body_names)
            if (
                "leftfinger" in name
                or "rightfinger" in name
                or "finger_joint1_tip" in name
                or "finger_joint2_tip" in name
            )
        }
        bowl_body_ids = {
            idx for idx, name in enumerate(body_names) if "bowl" in name.lower()
        }
        plate_body_ids = {
            idx for idx, name in enumerate(body_names) if "plate" in name.lower()
        }
        object_body_ids = bowl_body_ids | plate_body_ids
        pad_geom_ids = {
            idx
            for idx, name in enumerate(geom_names)
            if ("finger1_pad_collision" in name or "finger2_pad_collision" in name)
        }

        return dict(
            sim=sim,
            model=model,
            data=data,
            body_names=body_names,
            geom_names=geom_names,
            finger_body_ids=finger_body_ids,
            bowl_body_ids=bowl_body_ids,
            plate_body_ids=plate_body_ids,
            object_body_ids=object_body_ids,
            pad_geom_ids=pad_geom_ids,
        )

    def _collect_contact_step_debug(self, env, ctx):
        if ctx is None:
            return None

        sim = ctx["sim"]
        model = ctx["model"]
        data = ctx["data"]
        body_names = ctx["body_names"]
        geom_names = ctx["geom_names"]
        finger_body_ids = ctx["finger_body_ids"]
        bowl_body_ids = ctx["bowl_body_ids"]
        plate_body_ids = ctx["plate_body_ids"]
        object_body_ids = ctx["object_body_ids"]
        pad_geom_ids = ctx["pad_geom_ids"]

        ncon = int(data.ncon)
        finger_object_contacts = 0
        finger_bowl_contacts = 0
        finger_plate_contacts = 0
        pad_object_contacts = 0
        pairs = []
        max_pairs = 5

        for ci in range(ncon):
            c = data.contact[ci]
            g1 = int(c.geom1)
            g2 = int(c.geom2)
            b1 = int(model.geom_bodyid[g1])
            b2 = int(model.geom_bodyid[g2])

            touches_finger_object = (
                (b1 in finger_body_ids and b2 in object_body_ids)
                or (b2 in finger_body_ids and b1 in object_body_ids)
            )
            if not touches_finger_object:
                continue

            finger_object_contacts += 1
            if (b1 in bowl_body_ids) or (b2 in bowl_body_ids):
                finger_bowl_contacts += 1
            if (b1 in plate_body_ids) or (b2 in plate_body_ids):
                finger_plate_contacts += 1

            touches_pad_object = (
                (g1 in pad_geom_ids and b2 in object_body_ids)
                or (g2 in pad_geom_ids and b1 in object_body_ids)
            )
            if touches_pad_object:
                pad_object_contacts += 1

            if len(pairs) < max_pairs:
                g1_name = geom_names[g1] if 0 <= g1 < len(geom_names) else f"geom{g1}"
                g2_name = geom_names[g2] if 0 <= g2 < len(geom_names) else f"geom{g2}"
                b1_name = body_names[b1] if 0 <= b1 < len(body_names) else f"body{b1}"
                b2_name = body_names[b2] if 0 <= b2 < len(body_names) else f"body{b2}"
                pairs.append(
                    dict(
                        g1=g1_name,
                        g2=g2_name,
                        b1=b1_name,
                        b2=b2_name,
                        dist=float(c.dist),
                        pad=int(touches_pad_object),
                    )
                )

        return dict(
            ncon=ncon,
            finger_object_contacts=finger_object_contacts,
            finger_bowl_contacts=finger_bowl_contacts,
            finger_plate_contacts=finger_plate_contacts,
            pad_object_contacts=pad_object_contacts,
            pairs=pairs,
        )

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.suite.action_repeat

    def _rigidify_pred_world(self, env, input_world, pred_world):
        """Project predicted robot keypoints onto a rigid transform manifold."""
        pred_world = np.asarray(pred_world, dtype=np.float32)
        input_world = np.asarray(input_world, dtype=np.float32)
        if pred_world.ndim != 2 or input_world.ndim != 2:
            return pred_world
        if pred_world.shape[1] < 3 or input_world.shape[1] < 3:
            return pred_world

        num_robot_points = int(getattr(env, "_num_robot_points", 0))
        if num_robot_points <= 0:
            return pred_world
        if pred_world.shape[0] < num_robot_points or input_world.shape[0] < num_robot_points:
            return pred_world

        curr_robot = input_world[:num_robot_points, :3]
        pred_robot = pred_world[:num_robot_points, :3]
        rigid_ids = list(getattr(env, "_rigid_local_indices", range(num_robot_points)))
        valid_ids = [
            idx
            for idx in rigid_ids
            if 0 <= int(idx) < num_robot_points
        ]
        if len(valid_ids) < 3:
            return pred_world

        try:
            rot, trans = rigid_transform_3D(
                curr_robot[valid_ids], pred_robot[valid_ids]
            )
            rot = np.asarray(rot, dtype=np.float32)
            trans = np.asarray(trans, dtype=np.float32).reshape(3)
            projected_robot = (rot @ curr_robot.T).T + trans
        except Exception:
            return pred_world

        rigidified = pred_world.copy()
        if self.rigidify_pred_points_all_robot:
            rigidified[:num_robot_points, :3] = projected_robot
        else:
            rigidified[valid_ids, :3] = projected_robot[valid_ids]
        return rigidified

    def _maybe_rigidify_action(self, env, action, observation):
        """Rigidify predicted world points in action for eval visualization/control."""
        if not self.rigidify_pred_points:
            return action
        if not isinstance(action, dict):
            return action

        updated = dict(action)
        for pixel_key in self.cfg.suite.pixel_keys:
            pred_key = f"future_tracks_{pixel_key}"
            obs_key = f"point_tracks_{pixel_key}"
            if pred_key not in updated or obs_key not in observation:
                continue

            pred_np = np.asarray(updated[pred_key], dtype=np.float32)
            batched = pred_np.ndim == 3
            pred_world = pred_np[0] if batched else pred_np
            if pred_world.ndim != 2:
                continue

            input_world = np.asarray(observation[obs_key], dtype=np.float32).reshape(-1, 3)
            rigidified_world = self._rigidify_pred_world(env, input_world, pred_world)
            if rigidified_world.shape != pred_world.shape:
                continue

            pred_out = pred_world.copy()
            pred_out[:, :3] = rigidified_world[:, :3]
            updated[pred_key] = pred_out[None] if batched else pred_out

        return updated

    def _overlay_points(self, frame, input_pixels_hw, pred_pixels_hw, num_robot_points):
        out = np.asarray(frame).copy()
        if out.dtype != np.uint8:
            out = np.clip(out, 0, 255).astype(np.uint8)
        if out.ndim != 3 or out.shape[2] != 3:
            return out

        h, w = out.shape[:2]
        r = max(1, self.point_overlay_radius)

        def _draw_filled(points_hw, color_rgb):
            for ph, pw in np.asarray(points_hw).reshape(-1, 2):
                y, x = int(ph), int(pw)
                if 0 <= y < h and 0 <= x < w:
                    cv2.circle(out, (x, y), r, color_rgb, -1)

        def _draw_ring(points_hw, color_rgb):
            for ph, pw in np.asarray(points_hw).reshape(-1, 2):
                y, x = int(ph), int(pw)
                if 0 <= y < h and 0 <= x < w:
                    cv2.circle(out, (x, y), r + 1, color_rgb, 1)

        input_pixels_hw = np.asarray(input_pixels_hw).reshape(-1, 2)
        pred_pixels_hw = np.asarray(pred_pixels_hw).reshape(-1, 2)

        input_robot = input_pixels_hw[: int(num_robot_points)]
        input_object = input_pixels_hw[int(num_robot_points) :]

        # RGB colors on the saved frame.
        _draw_filled(input_robot, (0, 255, 0))      # input robot: green
        _draw_filled(input_object, (0, 255, 255))   # input object: yellow
        _draw_ring(pred_pixels_hw, (255, 0, 0))     # predicted robot: red ring
        return out

    def _resize_panel(self, frame):
        out = np.asarray(frame)
        if out.dtype != np.uint8:
            out = np.clip(out, 0, 255).astype(np.uint8)
        if out.shape[0] != self.video_recorder.render_size or out.shape[1] != self.video_recorder.render_size:
            out = cv2.resize(
                out,
                dsize=(self.video_recorder.render_size, self.video_recorder.render_size),
                interpolation=cv2.INTER_CUBIC,
            )
        return out

    @staticmethod
    def _slugify(text):
        return re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("_")

    def _build_export_video_name(
        self,
        env_idx,
        status=None,
        success_count=None,
        num_episodes=None,
        episode_idx=None,
        chunk_idx=None,
    ):
        job_id = self._slugify(os.environ.get("SLURM_JOB_ID", "local"))
        parts = [f"job{job_id}", f"env{int(env_idx)}"]
        if episode_idx is not None:
            parts.append(f"ep{int(episode_idx)}")
        if chunk_idx is not None:
            parts.append(f"chunk{int(chunk_idx)}")
        if status is not None:
            parts.append(self._slugify(status))
        if success_count is not None and num_episodes is not None:
            parts.append(f"s{int(success_count)}of{int(num_episodes)}")
        return "__".join(parts) + ".mp4"

    def _export_video_copy(
        self,
        env_idx,
        local_video_name,
        status=None,
        success_count=None,
        num_episodes=None,
        episode_idx=None,
        chunk_idx=None,
    ):
        if (
            not self.video_export_enabled
            or self.video_export_dir is None
            or self.video_recorder.save_dir is None
        ):
            return
        src = self.video_recorder.save_dir / local_video_name
        if not src.exists():
            return

        dst = self.video_export_dir / self._build_export_video_name(
            env_idx,
            status=status,
            success_count=success_count,
            num_episodes=num_episodes,
            episode_idx=episode_idx,
            chunk_idx=chunk_idx,
        )
        if dst.exists():
            stem = dst.stem
            suffix = dst.suffix
            idx = 2
            while True:
                candidate = self.video_export_dir / f"{stem}_v{idx}{suffix}"
                if not candidate.exists():
                    dst = candidate
                    break
                idx += 1

        shutil.copy2(src, dst)
        print(f"[video-export] saved: {dst}")

    def _eval_offline_gt_points(self):
        """Run policy on GT point observations from PKL only (no env.step/action exec)."""
        dataset = self.expert_replay_loader.dataset
        debug_pixel_key = self.cfg.suite.pixel_keys[0]
        obs_point_key = f"point_tracks_{debug_pixel_key}"
        pred_point_key = f"future_tracks_{debug_pixel_key}"
        demo_idx_cfg = int(getattr(self.cfg, "offline_demo_index", 0))
        max_steps_cfg = int(getattr(self.cfg, "offline_max_steps", -1))

        for env_idx in range(self.envs_till_idx):
            episodes = dataset._episodes.get(env_idx, [])
            if len(episodes) == 0:
                print(f"[offline-point-debug] env={env_idx}: no episodes, skip")
                continue

            demo_idx = max(0, min(int(demo_idx_cfg), len(episodes) - 1))
            observations = episodes[demo_idx]["observation"]
            if obs_point_key not in observations:
                print(
                    f"[offline-point-debug] env={env_idx}: missing {obs_point_key}, skip"
                )
                continue

            traj_len = int(len(observations[obs_point_key]))
            history_pad = int(getattr(dataset, "_history_len", 0))
            if traj_len > history_pad:
                traj_len = traj_len - history_pad
            if max_steps_cfg > 0:
                traj_len = min(traj_len, max_steps_cfg)

            print(
                "[offline-point-debug] "
                f"env={env_idx} demo={demo_idx} steps={traj_len} "
                f"(history_pad={history_pad})"
            )

            self.agent.buffer_reset()

            point_debug_inputs = []
            point_debug_preds = []
            gripper_pred = []
            gripper_gt = []
            steps = []

            for step in range(traj_len):
                obs = {}
                for pixel_key in self.cfg.suite.pixel_keys:
                    k = f"point_tracks_{pixel_key}"
                    if k not in observations:
                        raise KeyError(f"{k} not found in offline episode observation")
                    obs[k] = np.asarray(observations[k][step], dtype=np.float32)

                if "features" in observations:
                    obs["features"] = np.asarray(
                        observations["features"][step], dtype=np.float32
                    )
                else:
                    obs["features"] = np.zeros((4,), dtype=np.float32)

                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(
                        obs,
                        self.expert_replay_loader.dataset.stats,
                        step,
                        self.global_step,
                        eval_mode=True,
                    )

                if pred_point_key in action:
                    pred_points = np.asarray(action[pred_point_key], dtype=np.float32)
                    if pred_points.ndim == 3:
                        pred_points = pred_points[0]
                    pred_points = pred_points.reshape(-1, pred_points.shape[-1])
                    point_debug_inputs.append(obs[obs_point_key].copy())
                    point_debug_preds.append(pred_points.copy())
                    steps.append(int(step))

                pred_g = np.nan
                if "gripper" in action:
                    pred_g = float(np.asarray(action["gripper"]).reshape(-1)[0])
                elif "future_gripper_states" in action:
                    pred_g = float(
                        np.asarray(action["future_gripper_states"]).reshape(-1)[0]
                    )
                gripper_pred.append(pred_g)

                gt_g = np.nan
                if "gripper_states" in observations:
                    gt_g = float(np.asarray(observations["gripper_states"][step]).reshape(-1)[0])
                gripper_gt.append(gt_g)

            if len(point_debug_inputs) > 0:
                point_debug_inputs_np = np.stack(point_debug_inputs, axis=0)
                point_debug_preds_np = np.stack(point_debug_preds, axis=0)
                debug_npz = self.work_dir / f"point_debug_offline_env{env_idx}.npz"
                np.savez_compressed(
                    debug_npz,
                    input_points=point_debug_inputs_np,
                    pred_points=point_debug_preds_np,
                    step=np.asarray(steps, dtype=np.int32),
                    gripper_pred=np.asarray(gripper_pred, dtype=np.float32),
                    gripper_gt=np.asarray(gripper_gt, dtype=np.float32),
                    env_idx=np.asarray([env_idx], dtype=np.int32),
                    demo_idx=np.asarray([demo_idx], dtype=np.int32),
                    pixel_key=np.asarray([debug_pixel_key]),
                )
                print(f"[offline-point-debug] saved: {debug_npz}")

                preview_txt = self.work_dir / f"point_debug_offline_env{env_idx}.txt"
                with preview_txt.open("w", encoding="utf-8") as fp:
                    num_robot_points = int(self.cfg.suite.num_robot_points)
                    total_input_points = int(point_debug_inputs_np.shape[1])
                    input_dim = int(point_debug_inputs_np.shape[2])
                    pred_points_count = int(point_debug_preds_np.shape[1])
                    pred_dim = int(point_debug_preds_np.shape[2])
                    num_object_points = max(0, total_input_points - num_robot_points)
                    fp.write(
                        f"mode=offline_gt_points_only\n"
                        f"env_idx={env_idx}\n"
                        f"demo_idx={demo_idx}\n"
                        f"pixel_key={debug_pixel_key}\n"
                        f"num_steps={len(point_debug_inputs)}\n"
                        f"input_points shape: [num_steps, {total_input_points}, {input_dim}] "
                        f"(robot {num_robot_points} + object {num_object_points})\n"
                        f"pred_points shape: [num_steps, {pred_points_count}, {pred_dim}] "
                        "(robot only)\n"
                        f"gripper_pred min/max: "
                        f"{float(np.nanmin(np.asarray(gripper_pred))):.6f}/"
                        f"{float(np.nanmax(np.asarray(gripper_pred))):.6f}\n"
                        f"gripper_gt min/max: "
                        f"{float(np.nanmin(np.asarray(gripper_gt))):.6f}/"
                        f"{float(np.nanmax(np.asarray(gripper_gt))):.6f}\n\n"
                    )
                    show_n = min(10, len(point_debug_inputs))
                    for idx in range(show_n):
                        fp.write(
                            f"[step {steps[idx]}] "
                            f"gripper_pred={gripper_pred[idx]:.6f} "
                            f"gripper_gt={gripper_gt[idx]:.6f}\n"
                        )
                        fp.write(
                            "input_robot_first3=\n"
                            f"{point_debug_inputs_np[idx, :3]}\n"
                            "pred_robot_first3=\n"
                            f"{point_debug_preds_np[idx, :3]}\n\n"
                        )
                print(f"[offline-point-debug] preview: {preview_txt}")

    def eval(self):
        self.agent.train(False)
        if bool(getattr(self.cfg, "offline_gt_points_only", False)):
            self._eval_offline_gt_points()
            self.agent.train(True)
            return

        episode_rewards = []
        successes = []
        episode_lengths = []
        for env_idx in range(self.envs_till_idx):
            print(f"evaluating env {env_idx}")
            episode, total_reward = 0, 0
            eval_until_episode = utils.Until(self.cfg.suite.num_eval_episodes)
            success = []
            point_debug_inputs = []
            point_debug_preds = []
            point_debug_episodes = []
            point_debug_steps = []
            gripper_debug_pred = []
            gripper_debug_cmd = []
            gripper_debug_obs = []
            gripper_debug_qpos = []
            gripper_debug_episodes = []
            gripper_debug_steps = []
            contact_debug_lines = []
            debug_pixel_key = self.cfg.suite.pixel_keys[0]
            obs_point_key = f"point_tracks_{debug_pixel_key}"
            pred_point_key = f"future_tracks_{debug_pixel_key}"
            contact_ctx = self._build_contact_debug_ctx(self.env[env_idx])

            while eval_until_episode(episode):
                print(episode)
                time_step = self.env[env_idx].reset()
                self.agent.buffer_reset()
                step = 0
                episode_success = False
                episode_video_chunks = []
                episode_chunk_idx = 0

                if self.video_save_per_episode:
                    self.video_recorder.init(enabled=True)
                elif episode == 0:
                    self.video_recorder.init(enabled=True)

                while True:
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        action = self.agent.act(
                            time_step.observation,
                            self.expert_replay_loader.dataset.stats,
                            step,
                            self.global_step,
                            eval_mode=True,
                        )
                    action = self._maybe_rigidify_action(
                        self.env[env_idx], action, time_step.observation
                    )
                    if (
                        self.point_debug
                        and len(point_debug_inputs) < self.point_debug_max_steps
                        and obs_point_key in time_step.observation
                        and pred_point_key in action
                    ):
                        input_points = np.asarray(
                            time_step.observation[obs_point_key], dtype=np.float32
                        )
                        pred_points = np.asarray(action[pred_point_key], dtype=np.float32)
                        point_debug_inputs.append(input_points.copy())
                        point_debug_preds.append(pred_points.copy())
                        point_debug_episodes.append(int(episode))
                        point_debug_steps.append(int(step))
                        if len(point_debug_inputs) == 1:
                            print(
                                "[point-debug] captured first sample: "
                                f"input={input_points.shape}, pred={pred_points.shape}"
                            )
                    pending_gripper_debug = None
                    if self.point_debug and len(gripper_debug_pred) < self.point_debug_max_steps:
                        pred_g = None
                        if "gripper" in action:
                            pred_g = float(np.asarray(action["gripper"]).reshape(-1)[0])
                        elif "future_gripper_states" in action:
                            pred_g = float(
                                np.asarray(action["future_gripper_states"]).reshape(-1)[0]
                            )
                        if pred_g is not None:
                            obs_g = np.nan
                            try:
                                obs_g = float(np.asarray(time_step.observation["features"]).reshape(-1)[-1])
                            except Exception:
                                pass
                            raw_qpos = np.nan
                            try:
                                raw_obs = getattr(self.env[env_idx], "_current_raw_obs", None)
                                if isinstance(raw_obs, dict) and "robot0_gripper_qpos" in raw_obs:
                                    raw_qpos = float(
                                        np.asarray(raw_obs["robot0_gripper_qpos"]).reshape(-1)[0]
                                    )
                            except Exception:
                                pass
                            pending_gripper_debug = (
                                pred_g,
                                obs_g,
                                raw_qpos,
                                int(episode),
                                int(step),
                            )

                    if (
                        self.point_overlay
                        and obs_point_key in time_step.observation
                        and pred_point_key in action
                    ):
                        raw_frame = self.env[env_idx].render()
                        input_world = np.asarray(
                            time_step.observation[obs_point_key], dtype=np.float32
                        ).reshape(-1, 3)
                        pred_world = np.asarray(action[pred_point_key], dtype=np.float32)
                        if pred_world.ndim == 3:
                            pred_world = pred_world[0]
                        pred_world = pred_world.reshape(-1, pred_world.shape[-1])[:, :3]
                        try:
                            input_pixels_hw = self.env[env_idx].project_world_points_to_pixel(
                                input_world, pixel_key=debug_pixel_key
                            )
                            pred_pixels_hw = self.env[env_idx].project_world_points_to_pixel(
                                pred_world, pixel_key=debug_pixel_key
                            )
                            vis_raw = np.asarray(raw_frame).copy()
                            if self.video_recorder.flip_vertical:
                                vis_raw = np.flipud(vis_raw).copy()
                            vis_overlay = self._overlay_points(
                                vis_raw,
                                input_pixels_hw,
                                pred_pixels_hw,
                                num_robot_points=self.cfg.suite.num_robot_points,
                            )
                            if self.point_overlay_side_by_side:
                                left = self._resize_panel(vis_raw)
                                right = self._resize_panel(vis_overlay)
                                composed = np.concatenate([left, right], axis=1)
                                self.video_recorder.record_frame(
                                    composed, apply_flip=False, apply_resize=False
                                )
                            else:
                                self.video_recorder.record_frame(
                                    vis_overlay,
                                    apply_flip=not self.video_recorder.flip_vertical,
                                    apply_resize=True,
                                )
                        except Exception as exc:
                            if not hasattr(self, "_overlay_warned"):
                                print(f"[point-overlay] projection failed once: {exc}")
                                self._overlay_warned = True
                            if self.point_overlay_side_by_side:
                                vis_raw = np.asarray(raw_frame).copy()
                                if self.video_recorder.flip_vertical:
                                    vis_raw = np.flipud(vis_raw).copy()
                                left = self._resize_panel(vis_raw)
                                right = self._resize_panel(vis_raw)
                                composed = np.concatenate([left, right], axis=1)
                                self.video_recorder.record_frame(
                                    composed, apply_flip=False, apply_resize=False
                                )
                            else:
                                self.video_recorder.record(self.env[env_idx])
                    else:
                        self.video_recorder.record(self.env[env_idx])

                    time_step = self.env[env_idx].step(action)
                    if pending_gripper_debug is not None:
                        cmd_g = np.nan
                        try:
                            last_action = getattr(self.env[env_idx], "_last_robot_action", None)
                            if last_action is not None:
                                cmd_g = float(np.asarray(last_action).reshape(-1)[6])
                        except Exception:
                            pass
                        pred_g, obs_g, raw_qpos, dbg_episode, dbg_step = pending_gripper_debug
                        gripper_debug_pred.append(pred_g)
                        gripper_debug_cmd.append(cmd_g)
                        gripper_debug_obs.append(obs_g)
                        gripper_debug_qpos.append(raw_qpos)
                        gripper_debug_episodes.append(dbg_episode)
                        gripper_debug_steps.append(dbg_step)
                    if (
                        self.contact_debug
                        and len(contact_debug_lines) < self.contact_debug_max_steps
                    ):
                        contact_info = self._collect_contact_step_debug(
                            self.env[env_idx], contact_ctx
                        )
                        if contact_info is not None:
                            cmd_g = np.nan
                            try:
                                last_action = getattr(
                                    self.env[env_idx], "_last_robot_action", None
                                )
                                if last_action is not None:
                                    cmd_g = float(np.asarray(last_action).reshape(-1)[6])
                            except Exception:
                                pass
                            line = (
                                f"ep={int(episode)} step={int(step)} "
                                f"cmd_gripper={cmd_g:.3f} "
                                f"ncon={contact_info['ncon']} "
                                f"finger_obj={contact_info['finger_object_contacts']} "
                                f"finger_bowl={contact_info['finger_bowl_contacts']} "
                                f"finger_plate={contact_info['finger_plate_contacts']} "
                                f"pad_obj={contact_info['pad_object_contacts']}"
                            )
                            if len(contact_info["pairs"]) > 0:
                                pair_chunks = []
                                for pair in contact_info["pairs"]:
                                    pair_chunks.append(
                                        f"({pair['g1']}|{pair['g2']} "
                                        f"{pair['b1']}|{pair['b2']} "
                                        f"dist={pair['dist']:.5f} pad={pair['pad']})"
                                    )
                                line = line + " pairs=" + "; ".join(pair_chunks)
                            contact_debug_lines.append(line)
                    total_reward += time_step.reward
                    step += 1
                    episode_success = episode_success or bool(
                        time_step.observation.get("goal_achieved", False)
                    )

                    if (
                        self.video_save_per_episode
                        and self.video_chunk_steps > 0
                        and step > 0
                        and step % self.video_chunk_steps == 0
                    ):
                        local_video_name = (
                            f"{self.global_frame}_env{env_idx}_ep{episode}_"
                            f"chunk{episode_chunk_idx}.mp4"
                        )
                        self.video_recorder.save(local_video_name)
                        episode_video_chunks.append((local_video_name, episode_chunk_idx))
                        episode_chunk_idx += 1
                        self.video_recorder.init(enabled=True)

                    if self.eval_rollout_steps > 0:
                        if step >= self.eval_rollout_steps:
                            break
                    else:
                        if time_step.last():
                            break

                episode += 1
                success.append(float(episode_success))
                episode_lengths.append(step)
                if self.video_save_per_episode:
                    if len(self.video_recorder.frames) > 0:
                        local_video_name = (
                            f"{self.global_frame}_env{env_idx}_ep{episode - 1}_"
                            f"chunk{episode_chunk_idx}.mp4"
                        )
                        self.video_recorder.save(local_video_name)
                        episode_video_chunks.append((local_video_name, episode_chunk_idx))
                    episode_status = "success" if episode_success else "fail"
                    for local_name, chunk_idx in episode_video_chunks:
                        self._export_video_copy(
                            env_idx,
                            local_name,
                            status=episode_status,
                            episode_idx=episode - 1,
                            chunk_idx=chunk_idx,
                        )
            local_video_name = f"{self.global_frame}_env{env_idx}.mp4"
            if not self.video_save_per_episode:
                self.video_recorder.save(local_video_name)
            env_success_count = int(np.sum(success))
            env_num_episodes = int(len(success))
            env_status = "success" if env_success_count > 0 else "fail"
            if not self.video_save_per_episode:
                self._export_video_copy(
                    env_idx,
                    local_video_name,
                    status=env_status,
                    success_count=env_success_count,
                    num_episodes=env_num_episodes,
                )

            if self.point_debug and len(point_debug_inputs) > 0:
                point_debug_inputs_np = np.stack(point_debug_inputs, axis=0)
                point_debug_preds_np = np.stack(point_debug_preds, axis=0)
                debug_npz = self.work_dir / f"point_debug_env{env_idx}.npz"
                np.savez_compressed(
                    debug_npz,
                    input_points=point_debug_inputs_np,
                    pred_points=point_debug_preds_np,
                    episode=np.asarray(point_debug_episodes, dtype=np.int32),
                    step=np.asarray(point_debug_steps, dtype=np.int32),
                    pixel_key=np.asarray([debug_pixel_key]),
                )
                preview_txt = self.work_dir / f"point_debug_env{env_idx}.txt"
                with preview_txt.open("w", encoding="utf-8") as fp:
                    num_robot_points = int(self.cfg.suite.num_robot_points)
                    total_input_points = int(point_debug_inputs_np.shape[1])
                    input_dim = int(point_debug_inputs_np.shape[2])
                    pred_points_count = int(point_debug_preds_np.shape[1])
                    pred_dim = int(point_debug_preds_np.shape[2])
                    num_object_points = max(0, total_input_points - num_robot_points)
                    fp.write(
                        f"pixel_key={debug_pixel_key}\n"
                        f"num_samples={len(point_debug_inputs)}\n"
                        f"input_points shape: [num_samples, {total_input_points}, {input_dim}] "
                        f"(robot {num_robot_points} + object {num_object_points})\n"
                        f"pred_points shape: [num_samples, {pred_points_count}, {pred_dim}] "
                        "(robot only)\n\n"
                    )
                    show_n = min(3, len(point_debug_inputs))
                    for idx in range(show_n):
                        object_preview_end = min(total_input_points, num_robot_points + 3)
                        fp.write(
                            f"[sample {idx}] episode={point_debug_episodes[idx]} "
                            f"step={point_debug_steps[idx]}\n"
                        )
                        fp.write(
                            "input_robot_first3=\n"
                            f"{point_debug_inputs_np[idx, :3]}\n"
                            "input_object_first3=\n"
                            f"{point_debug_inputs_np[idx, num_robot_points:object_preview_end]}\n"
                            "pred_robot_first3=\n"
                            f"{point_debug_preds_np[idx, :3]}\n\n"
                        )
                print(f"[point-debug] saved: {debug_npz}")
                print(f"[point-debug] preview: {preview_txt}")

            if self.point_debug and len(gripper_debug_pred) > 0:
                gripper_txt = self.work_dir / f"gripper_debug_env{env_idx}.txt"
                with gripper_txt.open("w", encoding="utf-8") as fp:
                    fp.write(
                        "pred_gripper: model output before threshold\n"
                        "cmd_gripper: final env action[6] after threshold/sign mapping\n\n"
                    )
                    for idx in range(len(gripper_debug_pred)):
                        fp.write(
                            f"ep={gripper_debug_episodes[idx]} "
                            f"step={gripper_debug_steps[idx]} "
                            f"obs_gripper={gripper_debug_obs[idx]:.6f} "
                            f"raw_qpos={gripper_debug_qpos[idx]:.6f} "
                            f"pred_gripper={gripper_debug_pred[idx]:.6f} "
                            f"cmd_gripper={gripper_debug_cmd[idx]:.1f}\n"
                        )
                print(f"[gripper-debug] saved: {gripper_txt}")
            if self.contact_debug and len(contact_debug_lines) > 0:
                contact_txt = self.work_dir / f"contact_debug_env{env_idx}.txt"
                with contact_txt.open("w", encoding="utf-8") as fp:
                    fp.write(
                        "ncon: total mujoco contacts\n"
                        "finger_obj: contacts where finger body touched bowl/plate body\n"
                        "pad_obj: subset where finger pad geom touched bowl/plate body\n\n"
                    )
                    for line in contact_debug_lines:
                        fp.write(line + "\n")
                print(f"[contact-debug] saved: {contact_txt}")

            episode_rewards.append(total_reward / episode)
            successes.append(np.mean(success))

        for _ in range(len(self.env) - self.envs_till_idx):
            episode_rewards.append(0)
            successes.append(0)

        with self.logger.log_and_dump_ctx(self.global_frame, ty="eval") as log:
            for env_idx, reward in enumerate(episode_rewards):
                log(f"episode_reward_env{env_idx}", reward)
                log(f"success_env{env_idx}", successes[env_idx])
            log("episode_reward", np.mean(episode_rewards[: self.envs_till_idx]))
            log("success", np.mean(successes))
            mean_episode_len = (
                float(np.mean(episode_lengths)) if len(episode_lengths) > 0 else 0.0
            )
            log("episode_length", mean_episode_len * self.cfg.suite.action_repeat)
            log("episode", self.global_episode)
            log("step", self.global_step)

        self.agent.train(True)

    def save_snapshot(self):
        snapshot = self.work_dir / "snapshot.pt"
        self.agent.clear_buffers()
        keys_to_save = ["timer", "_global_step", "_global_episode"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        payload.update(self.agent.save_snapshot())
        with snapshot.open("wb") as f:
            torch.save(payload, f)

        self.agent.buffer_reset()

    def load_snapshot(self, snapshots):
        # bc
        with snapshots["bc"].open("rb") as f:
            payload = torch.load(f)
        agent_payload = {}
        for k, v in payload.items():
            if k not in self.__dict__:
                agent_payload[k] = v
        self.agent.load_snapshot(agent_payload, eval=True)


@hydra.main(config_path="cfgs", config_name="config_eval", version_base=None)
def main(cfg):
    workspace = Workspace(cfg)

    # Load weights
    snapshots = {}
    # bc
    bc_snapshot = Path(cfg.bc_weight)
    if not bc_snapshot.exists():
        raise FileNotFoundError(f"bc weight not found: {bc_snapshot}")
    print(f"loading bc weight: {bc_snapshot}")
    snapshots["bc"] = bc_snapshot
    workspace.load_snapshot(snapshots)

    workspace.eval()


if __name__ == "__main__":
    main()
