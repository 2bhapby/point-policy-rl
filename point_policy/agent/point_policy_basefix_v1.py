import numpy as np
from collections import deque

import torch
from torch import nn
import torch.nn.functional as F

import utils
from agent.networks.policy_head import (
    DeterministicHead,
    DiffusionHead,
)
from agent.networks.mlp import MLP
from agent.networks.gpt import GPT, GPTConfig


class Actor(nn.Module):
    def __init__(
        self,
        repr_dim,
        act_dim,
        num_track_points,
        hidden_dim,
        policy_head="deterministic",
        device="cuda",
        pred_gripper=True,
    ):
        super().__init__()

        self._policy_head = policy_head
        self._repr_dim = repr_dim
        self._act_dim = act_dim
        # GPT context length must cover all point tokens (and optional gripper token).
        track_block_size = int(num_track_points) + (1 if bool(pred_gripper) else 0)
        block_size = max(20, track_block_size)

        self._policy = GPT(
            GPTConfig(
                block_size=block_size,
                input_dim=repr_dim,
                output_dim=hidden_dim,
                n_layer=4,
                n_head=2,
                n_embd=hidden_dim,
                dropout=0.1,
            )
        )

        if policy_head == "deterministic":
            self._action_head = DeterministicHead(
                hidden_dim, self._act_dim, hidden_size=hidden_dim, num_layers=2
            )
        elif policy_head == "diffusion":
            obs_horizon = num_track_points if not pred_gripper else num_track_points + 1
            pred_horizon = (
                num_track_points if not pred_gripper else num_track_points + 1
            )
            self._action_head = DiffusionHead(
                input_size=hidden_dim,
                output_size=self._act_dim,
                obs_horizon=obs_horizon,
                pred_horizon=pred_horizon,
                hidden_size=hidden_dim,
                num_layers=2,
                device=device,
            )

        self.apply(utils.weight_init)

    def forward(
        self,
        past_tracks,
        stddev,
        target=None,
        mask=None,
        return_features=False,
    ):
        features = self._policy(past_tracks)

        pred_action = self._action_head(
            features,
            stddev,
            **{
                "action_seq": target if target is not None else None,
            },
        )

        if target is None:
            if return_features:
                return pred_action, features
            return pred_action
        else:
            loss = self._action_head.loss_fn(
                pred_action,
                target,
                mask,
                reduction="mean",
            )
            loss = loss[0] if isinstance(loss, tuple) else loss
            if return_features:
                return pred_action, loss, features
            return pred_action, loss


class BCAgent:
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        lr,
        hidden_dim,
        stddev_schedule,
        use_tb,
        policy_head,
        pixel_keys,
        history,
        history_len,
        eval_history_len,
        temporal_agg,
        max_episode_len,
        num_queries,
        use_robot_points,
        num_robot_points,
        use_object_points,
        num_object_points,
        point_dim,
        pred_gripper,
        gripper_loss_weight=1.0,
        gripper_agg_mode="avg",
        condition_on_gripper_state=True,
        separate_gripper_head=False,
        gripper_bce_weight=1.0,
        gripper_label_mode="command",
        gripper_head_use_transformer_context=False,
        temporal_weight_mode="exp",
        temporal_weight_k=0.01,
        emit_temporal_debug=False,
    ):
        self.device = device
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.stddev_schedule = stddev_schedule
        self.use_tb = use_tb
        self.policy_head = policy_head
        self.history_len = history_len if history else 1
        self.eval_history_len = eval_history_len if history else 1
        self.pred_gripper = pred_gripper
        self.gripper_loss_weight = float(gripper_loss_weight)
        self.gripper_agg_mode = str(gripper_agg_mode).strip().lower()
        self.condition_on_gripper_state = bool(condition_on_gripper_state)
        self.separate_gripper_head = bool(separate_gripper_head)
        self.gripper_bce_weight = float(gripper_bce_weight)
        self.gripper_label_mode = str(gripper_label_mode).strip().lower()
        self.gripper_head_use_transformer_context = bool(
            gripper_head_use_transformer_context
        )
        if self.gripper_agg_mode not in {"avg", "latest", "min"}:
            raise ValueError(
                "gripper_agg_mode must be one of {'avg','latest','min'}, "
                f"got {gripper_agg_mode}"
            )
        if self.separate_gripper_head and not self.pred_gripper:
            raise ValueError("separate_gripper_head=true requires pred_gripper=true")
        if self.separate_gripper_head and self.gripper_label_mode != "command":
            raise ValueError(
                "separate_gripper_head with BCE currently supports "
                "gripper_label_mode='command' only"
            )
        self._use_gripper_token = self.pred_gripper and (not self.separate_gripper_head)

        self._use_robot_points = use_robot_points
        self._num_robot_points = num_robot_points
        self._use_object_points = use_object_points
        self._num_object_points = num_object_points
        self.num_track_points = (num_robot_points if use_robot_points else 0) + (
            num_object_points if use_object_points else 0
        )

        # actor parameters
        self._act_dim = point_dim
        assert self._act_dim in [2, 3], "Only 2D or 3D actions are supported"

        # keys
        self.pixel_keys = pixel_keys

        # action chunking params
        self.temporal_agg = temporal_agg
        self.max_episode_len = max_episode_len
        self.num_queries = num_queries if self.temporal_agg else 1
        self.temporal_weight_mode = str(temporal_weight_mode).strip().lower()
        self.temporal_weight_k = float(temporal_weight_k)
        self.emit_temporal_debug = bool(emit_temporal_debug)
        if self.temporal_weight_mode not in {"exp", "uniform", "latest"}:
            raise ValueError(
                "temporal_weight_mode must be one of {'exp','uniform','latest'}, "
                f"got {temporal_weight_mode}"
            )

        # observation params
        self.repr_dim = 512  # representation dim for transformer input
        obs_shape = obs_shape[self.pixel_keys[0]]

        # Track model size
        model_size = 0

        # projector for points and patches
        self.point_projector = MLP(
            self._act_dim * self.history_len, hidden_channels=[self.repr_dim]
        ).to(device)
        self.point_projector.apply(utils.weight_init)
        model_size += sum(
            p.numel() for p in self.point_projector.parameters() if p.requires_grad
        )

        # actor
        action_dim = (
            self._act_dim * self.num_queries if self.temporal_agg else self._act_dim
        )
        self.actor = Actor(
            self.repr_dim,
            action_dim,
            self.num_track_points,
            hidden_dim,
            self.policy_head,
            device,
            self._use_gripper_token,
        ).to(device)
        model_size += sum(p.numel() for p in self.actor.parameters() if p.requires_grad)

        self.gripper_head = None
        if self.pred_gripper and self.separate_gripper_head:
            gripper_head_input_dim = (
                hidden_dim
                if self.gripper_head_use_transformer_context
                else self.repr_dim
            )
            self.gripper_head = MLP(
                gripper_head_input_dim, hidden_channels=[hidden_dim, 1]
            ).to(device)
            self.gripper_head.apply(utils.weight_init)
            model_size += sum(
                p.numel() for p in self.gripper_head.parameters() if p.requires_grad
            )

        # optimizers
        # point projector
        params = list(self.point_projector.parameters())
        self.point_opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        # actor
        self.actor_opt = torch.optim.AdamW(
            self.actor.parameters(), lr=lr, weight_decay=1e-4
        )
        self.gripper_opt = None
        if self.gripper_head is not None:
            self.gripper_opt = torch.optim.AdamW(
                self.gripper_head.parameters(), lr=lr, weight_decay=1e-4
            )

        self.train()
        self.buffer_reset()

    def __repr__(self):
        return "bc"

    def train(self, training=True):
        self.training = training
        if training:
            self.point_projector.train(training)
            self.actor.train(training)
            if self.gripper_head is not None:
                self.gripper_head.train(training)
        else:
            self.point_projector.eval()
            self.actor.eval()
            if self.gripper_head is not None:
                self.gripper_head.eval()

    def buffer_reset(self):
        self.observation_buffer = {}
        for key in self.pixel_keys:
            self.observation_buffer[f"past_tracks_{key}"] = deque(
                maxlen=self.eval_history_len
            )  # since point track history concatenated
        if (
            self.pred_gripper
            and self._use_gripper_token
            and self.condition_on_gripper_state
        ):
            self.observation_buffer["past_gripper_states"] = deque(
                maxlen=self.eval_history_len
            )

        # temporal aggregation
        if self.temporal_agg:
            self.all_time_actions = {}
            for pixel_key in self.pixel_keys:
                gripper_points = (
                    1 if self._use_gripper_token else 0
                )
                self.all_time_actions[pixel_key] = torch.zeros(
                    [
                        self.max_episode_len,
                        self.max_episode_len + self.num_queries,
                        self._act_dim * (self._num_robot_points + gripper_points),
                    ]
                ).to(self.device)
        self._last_temporal_debug = None

    def clear_buffers(self):
        del self.observation_buffer
        if self.temporal_agg:
            del self.all_time_actions
        self._last_temporal_debug = None

    def consume_temporal_debug(self):
        debug_payload = self._last_temporal_debug
        self._last_temporal_debug = None
        return debug_payload

    def act(self, obs, norm_stats, step, global_step, eval_mode=False, **kwargs):
        self._last_temporal_debug = None
        if norm_stats is not None:
            preprocess = {
                "past_tracks": lambda x: (x - norm_stats["past_tracks"]["min"])
                / (
                    norm_stats["past_tracks"]["max"]
                    - norm_stats["past_tracks"]["min"]
                    + 1e-5
                ),
                "gripper_states": lambda x: (x - norm_stats["gripper_states"]["min"])
                / (
                    norm_stats["gripper_states"]["max"]
                    - norm_stats["gripper_states"]["min"]
                    + 1e-5
                ),
            }
            post_process = {
                "future_tracks": lambda x: x
                * (norm_stats["past_tracks"]["max"] - norm_stats["past_tracks"]["min"])
                + norm_stats["past_tracks"]["min"],
                "gripper_states": lambda x: x
                * (
                    norm_stats["gripper_states"]["max"]
                    - norm_stats["gripper_states"]["min"]
                )
                + norm_stats["gripper_states"]["min"],
            }

        use_gripper_token = self._use_gripper_token
        use_gripper_condition = (
            use_gripper_token and self.condition_on_gripper_state
        )

        past_tracks = []
        for key in self.pixel_keys:
            point_tracks = preprocess["past_tracks"](obs[f"point_tracks_{key}"])
            self.observation_buffer[f"past_tracks_{key}"].append(point_tracks)
            while len(self.observation_buffer[f"past_tracks_{key}"]) < self.history_len:
                self.observation_buffer[f"past_tracks_{key}"].append(point_tracks)
            past_tracks.append(
                np.stack(self.observation_buffer[f"past_tracks_{key}"], axis=0)
            )
        if use_gripper_token:
            if use_gripper_condition:
                gripper_state = preprocess["gripper_states"](obs["features"][-1])
                self.observation_buffer["past_gripper_states"].append(gripper_state)
                while (
                    len(self.observation_buffer["past_gripper_states"])
                    < self.history_len
                ):
                    self.observation_buffer["past_gripper_states"].append(gripper_state)
                past_gripper_states = np.stack(
                    self.observation_buffer["past_gripper_states"], axis=0
                )
            else:
                # Keep a fixed neutral token so gripper does not condition motion.
                past_gripper_states = np.zeros((self.history_len,), dtype=np.float32)

        # convert to tensor
        past_tracks = torch.as_tensor(np.array(past_tracks), device=self.device).float()
        if use_gripper_token:
            past_gripper_states = torch.as_tensor(
                np.array(past_gripper_states), device=self.device
            ).float()

        # reshape past_tracks
        shape = past_tracks.shape
        past_tracks = past_tracks.transpose(1, 2).reshape(shape[0], shape[2], -1)
        if use_gripper_token:
            past_gripper_states = past_gripper_states[None, None].repeat(
                past_tracks.shape[0], 1, self._act_dim
            )
            past_tracks = torch.cat([past_tracks, past_gripper_states], dim=1)

        # encode past tracks
        past_tracks = self.point_projector(past_tracks)

        stddev = 0.1
        need_transformer_features = (
            self.pred_gripper
            and self.separate_gripper_head
            and self.gripper_head_use_transformer_context
        )

        if need_transformer_features:
            future_tracks, transformer_features = self.actor(
                past_tracks, stddev, return_features=True
            )
        else:
            future_tracks = self.actor(past_tracks, stddev)

        if self.policy_head == "deterministic":
            future_tracks = future_tracks.mean  # .cpu().numpy() # for deterministic

        gripper_value_np = None
        if self.pred_gripper and self.separate_gripper_head:
            if self.gripper_head_use_transformer_context:
                # Use contextualized transformer features so gripper prediction can
                # depend on both robot and object point context.
                gripper_feat = transformer_features.mean(dim=1)
            else:
                gripper_feat = past_tracks[:, : self._num_robot_points].mean(dim=1)
            gripper_logits = self.gripper_head(gripper_feat)
            gripper_prob = torch.sigmoid(gripper_logits)
            # Aggregate camera branches into a single gripper command.
            gripper_prob = gripper_prob.mean(dim=0, keepdim=True)
            gripper_value_np = post_process["gripper_states"](
                gripper_prob[:, :1].detach().cpu().numpy()
            )

        # extract robot and gripper points
        robot_points = future_tracks[:, : self._num_robot_points]
        if use_gripper_token:
            gripper_points = future_tracks[:, -1:]
            robot_points = torch.cat([robot_points, gripper_points], dim=1)
        future_tracks = robot_points

        return_dict = {}
        if not self.temporal_agg:
            for idx in range(len(future_tracks)):
                return_dict[f"future_tracks_{self.pixel_keys[idx]}"] = post_process[
                    "future_tracks"
                ](
                    future_tracks[idx, : self._num_robot_points, : self._act_dim]
                    .detach()
                    .cpu()
                    .numpy()
                )
                if self.pred_gripper:
                    if self.separate_gripper_head:
                        return_dict["gripper"] = gripper_value_np
                    else:
                        return_dict["future_gripper_states"] = post_process[
                            "gripper_states"
                        ](future_tracks[idx, -1:, :1].detach().cpu().numpy())
        else:
            for idx in range(len(future_tracks)):
                pixel_key = self.pixel_keys[idx]
                track = future_tracks[idx]
                track = track.view(-1, self.num_queries, self._act_dim)
                # consider only robot points
                start_idx = 0
                end_idx = (
                    start_idx + self._num_robot_points + (1 if use_gripper_token else 0)
                )
                track = track[start_idx:end_idx]
                # convert to proper shape
                track = track.transpose(0, 1).reshape(self.num_queries, -1)[None]
                self.all_time_actions[pixel_key][
                    [step],
                    step : step + self.num_queries,
                ] = track[-1:]
                tracks_for_curr_step = self.all_time_actions[pixel_key][:, step]
                tracks_populated = torch.all(tracks_for_curr_step != 0.0, dim=-1)
                tracks_for_curr_step = tracks_for_curr_step[tracks_populated]
                if len(tracks_for_curr_step) == 0:
                    tracks_for_curr_step = self.all_time_actions[pixel_key][
                        step : step + 1, step
                    ]
                if self.temporal_weight_mode == "latest":
                    weights_np = np.zeros((len(tracks_for_curr_step),), dtype=np.float32)
                    weights_np[-1] = 1.0
                elif self.temporal_weight_mode == "uniform":
                    weights_np = np.ones((len(tracks_for_curr_step),), dtype=np.float32)
                else:  # exp
                    weights_np = np.exp(
                        -self.temporal_weight_k * np.arange(len(tracks_for_curr_step))
                    ).astype(np.float32)
                weights_np = weights_np / max(weights_np.sum(), 1e-8)
                weights = torch.from_numpy(weights_np).to(self.device).unsqueeze(dim=1)
                track = (tracks_for_curr_step * weights).sum(dim=0, keepdim=True)
                track = track.detach().cpu().numpy()[0].reshape(-1, self._act_dim)
                return_dict[f"future_tracks_{pixel_key}"] = post_process[
                    "future_tracks"
                ](track[: self._num_robot_points])
                if self.emit_temporal_debug and idx == 0:
                    self._last_temporal_debug = {
                        "step": int(step),
                        "pixel_key": str(pixel_key),
                        "num_candidates": int(len(tracks_for_curr_step)),
                        "weight_mode": self.temporal_weight_mode,
                        "weights": weights_np.tolist(),
                    }
                if self.pred_gripper and use_gripper_token:
                    gripper_row_idx = self._num_robot_points
                    if self.gripper_agg_mode == "avg":
                        gripper_value = track[gripper_row_idx : gripper_row_idx + 1, :1]
                    else:
                        num_points_total = self._num_robot_points + 1
                        per_step = (
                            tracks_for_curr_step.detach()
                            .cpu()
                            .numpy()
                            .reshape(-1, num_points_total, self._act_dim)
                        )
                        gripper_series = per_step[:, gripper_row_idx, 0]
                        if self.gripper_agg_mode == "latest":
                            selected = float(gripper_series[-1])
                        else:  # "min"
                            selected = float(np.min(gripper_series))
                        gripper_value = np.array([[selected]], dtype=np.float32)
                    return_dict["gripper"] = post_process["gripper_states"](
                        gripper_value
                    )
            if self.pred_gripper and self.separate_gripper_head:
                return_dict["gripper"] = gripper_value_np

        return return_dict

    def update(self, expert_replay_iter, step, **kwargs):
        metrics = dict()

        batch = next(expert_replay_iter)
        data = utils.to_torch(batch, self.device)

        past_tracks = data["past_tracks"].float()
        future_tracks = data["future_tracks"].float()
        action_masks = data["action_mask"].float()
        if self.pred_gripper:
            past_gripper_states = data["past_gripper_states"].float()
            future_gripper_states = data["future_gripper_states"].float()
            if (
                (not self.condition_on_gripper_state)
                or self.separate_gripper_head
            ):
                past_gripper_states = torch.zeros_like(past_gripper_states)

        use_gripper_token = self._use_gripper_token
        if use_gripper_token:
            # Add a dimension to action masks for tokenized gripper regression.
            gripper_mask = (
                torch.ones_like(action_masks)[:, :1] * self.gripper_loss_weight
            )
            action_masks = torch.cat([action_masks, gripper_mask], dim=1)

        # reshape for training
        shape = past_tracks.shape
        past_tracks = past_tracks.transpose(1, 2).reshape(shape[0], shape[2], -1)
        future_tracks = future_tracks[:, 0]

        if use_gripper_token:
            past_gripper_states = past_gripper_states[:, None]
            future_gripper_states = future_gripper_states[:, :1]

            # Make last dim of gripper_states same as that of tracks
            past_gripper_states = past_gripper_states.repeat(1, 1, self._act_dim)
            future_gripper_states = future_gripper_states.repeat(1, 1, self._act_dim)

            # add gripper states as (n+1)-th track point
            past_tracks = torch.cat([past_tracks, past_gripper_states], dim=1)
            future_tracks = torch.cat([future_tracks, future_gripper_states], dim=1)

        # encode past tracks
        past_tracks = self.point_projector(past_tracks)

        # actor loss
        stddev = utils.schedule(self.stddev_schedule, step)
        need_transformer_features = (
            self.pred_gripper
            and self.separate_gripper_head
            and self.gripper_head_use_transformer_context
        )

        if need_transformer_features:
            pred_action, actor_loss, transformer_features = self.actor(
                past_tracks,
                stddev,
                future_tracks,
                action_masks,
                return_features=True,
                **kwargs,
            )
        else:
            pred_action, actor_loss = self.actor(
                past_tracks,
                stddev,
                future_tracks,
                action_masks,
                **kwargs,
            )

        if self.pred_gripper and self.separate_gripper_head:
            # Use current-step target (command label, normalized to [0,1]).
            gripper_target = future_gripper_states[:, 0, 0]
            gripper_target = (gripper_target >= 0.5).float()
            if self.gripper_head_use_transformer_context:
                gripper_feat = transformer_features.mean(dim=1)
            else:
                gripper_feat = past_tracks[:, : self._num_robot_points].mean(dim=1)
            gripper_logits = self.gripper_head(gripper_feat).squeeze(-1)
            gripper_bce = F.binary_cross_entropy_with_logits(
                gripper_logits, gripper_target, reduction="mean"
            )
            actor_loss["gripper_bce_loss"] = gripper_bce
            actor_loss["actor_loss"] = (
                actor_loss["actor_loss"] + self.gripper_bce_weight * gripper_bce
            )
            actor_loss["gripper_prob_mean"] = torch.sigmoid(gripper_logits).mean()
            actor_loss["gripper_target_mean"] = gripper_target.mean()

        # optimize
        self.point_opt.zero_grad(set_to_none=True)
        self.actor_opt.zero_grad(set_to_none=True)
        if self.gripper_opt is not None:
            self.gripper_opt.zero_grad(set_to_none=True)
        actor_loss["actor_loss"].backward()
        self.point_opt.step()
        self.actor_opt.step()
        if self.gripper_opt is not None:
            self.gripper_opt.step()

        if self.policy_head == "diffusion" and step % 10 == 0:
            self.actor._action_head.net.ema_step()

        if self.use_tb:
            for key, value in actor_loss.items():
                metrics[key] = value.item()

        return metrics

    def save_snapshot(self):
        model_keys = ["actor", "point_projector"]
        opt_keys = ["actor_opt", "point_opt"]
        if self.gripper_head is not None:
            model_keys.append("gripper_head")
        if self.gripper_opt is not None:
            opt_keys.append("gripper_opt")
        # models
        payload = {
            k: self.__dict__[k].state_dict() for k in model_keys if k != "encoder"
        }
        # optimizers
        payload.update({k: self.__dict__[k] for k in opt_keys})

        others = ["max_episode_len"]
        payload.update({k: self.__dict__[k] for k in others})
        return payload

    def load_snapshot(self, payload, eval=False):
        # models
        model_keys = ["actor", "point_projector"]
        if self.gripper_head is not None:
            model_keys.append("gripper_head")
        for k in model_keys:
            if k not in payload:
                if k == "gripper_head":
                    raise KeyError(
                        "gripper_head missing in checkpoint while "
                        "separate_gripper_head=true"
                    )
                continue
            self.__dict__[k].load_state_dict(payload[k])

        if eval:
            self.train(False)
            return
