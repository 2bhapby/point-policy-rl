#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# Allow `python point_policy_rl/train_residual_td3_rl.py` execution.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from point_policy_rl.base_policy_adapter_rl import FrozenPointPolicyBaseRL
from point_policy_rl.env_bridge_rl import (
    build_single_env_from_bc_config,
    clip_action,
    observation_to_state,
)
from point_policy_rl.offline_demo_builder_rl import build_offline_transitions_from_expert_demos_rl
from point_policy_rl.replay_buffer_rl import ReplayBufferRL
from point_policy_rl.td3_agent_rl import ResidualTD3AgentRL, TD3ConfigRL
from point_policy_rl.utils_rl import build_run_dir, coerce_device, dump_json, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Residual TD3 for Point-Policy (isolated _rl path)")
    parser.add_argument("--bc-weight", type=str, required=True)
    parser.add_argument("--suite", type=str, default=None, choices=["libero_spatial", "libero_object"])
    parser.add_argument("--task-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--warmup-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=300000)

    parser.add_argument("--actor-hidden-dim", type=int, default=256)
    parser.add_argument("--critic-hidden-dim", type=int, default=256)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--policy-noise", type=float, default=0.2)
    parser.add_argument("--noise-clip", type=float, default=0.5)
    parser.add_argument("--actor-update-freq", type=int, default=2)
    parser.add_argument("--exploration-std", type=float, default=0.1)

    parser.add_argument("--residual-pos-scale", type=float, default=0.02)
    parser.add_argument("--residual-rot-scale", type=float, default=0.10)
    parser.add_argument("--residual-gripper-scale", type=float, default=0.50)

    parser.add_argument("--eval-every", type=int, default=5000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=10000)
    parser.add_argument("--log-every", type=int, default=200)

    parser.add_argument("--include-eef-pos", action="store_true")
    parser.add_argument("--env-max-episode-len", type=int, default=3000)

    parser.add_argument("--offline-fraction", type=float, default=0.0)
    parser.add_argument("--offline-demo-root", type=str, default="")
    parser.add_argument("--offline-max-demos", type=int, default=0)
    parser.add_argument("--offline-step-reward", type=float, default=0.0)
    parser.add_argument("--offline-terminal-reward", type=float, default=1.0)
    parser.add_argument(
        "--offline-base-action-mode",
        type=str,
        default="demo_delta",
        choices=["demo_delta", "bc_track_delta"],
    )

    parser.add_argument("--num-updates-per-step", type=int, default=1)
    parser.add_argument("--update-every-n-steps", type=int, default=1)

    parser.add_argument("--output-root", type=str, default="point_policy/exp_local_rl")
    parser.add_argument("--run-name", type=str, default="")

    return parser.parse_args()


def _evaluate(
    env,
    base_policy: FrozenPointPolicyBaseRL,
    td3_agent: ResidualTD3AgentRL,
    episodes: int,
    pixel_key: str,
    low: np.ndarray,
    high: np.ndarray,
    include_eef_pos: bool,
) -> dict[str, float]:
    episode_returns = []
    successes = []

    for _ in range(episodes):
        time_step = env.reset()
        obs = time_step.observation
        base_policy.reset_episode()
        step_in_ep = 0

        done = False
        ep_ret = 0.0
        success = 0.0

        while not done:
            base_action_dict = base_policy.act(obs, step_in_ep, step_in_ep)
            base_action = np.asarray(env.point2action(base_action_dict), dtype=np.float32).reshape(7)

            state = observation_to_state(
                obs=obs,
                pixel_key=pixel_key,
                base_action_7d=base_action,
                include_eef_pos=include_eef_pos,
            )
            residual = td3_agent.select_action(state, exploration_std=0.0, deterministic=True)
            env_action = clip_action(base_action + residual, low=low, high=high)

            time_step = env.step(env_action)
            obs = time_step.observation
            done = bool(time_step.last())
            ep_ret += float(time_step.reward)
            success = 1.0 if bool(obs.get("goal_achieved", False)) else success
            step_in_ep += 1

        episode_returns.append(ep_ret)
        successes.append(success)

    return {
        "eval/episode_return": float(np.mean(episode_returns)) if episode_returns else 0.0,
        "eval/success": float(np.mean(successes)) if successes else 0.0,
    }


def _write_csv_row(csv_path: Path, row: dict[str, float | int]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _concat_batches(parts: list[dict]):
    if len(parts) == 1:
        return parts[0]

    import torch

    out = {}
    for key in parts[0].keys():
        out[key] = torch.cat([part[key] for part in parts], dim=0)
    return out


def _sample_mixed_batch(
    online_replay: ReplayBufferRL,
    offline_replay: ReplayBufferRL | None,
    online_batch_size: int,
    offline_batch_size: int,
    fallback_batch_size: int,
    device: str,
):
    batch_parts = []
    used_online_batch = 0
    used_offline_batch = 0

    if online_batch_size > 0 and len(online_replay) >= online_batch_size:
        batch_parts.append(online_replay.sample(batch_size=online_batch_size, device=device))
        used_online_batch = online_batch_size
    if (
        offline_replay is not None
        and offline_batch_size > 0
        and len(offline_replay) >= offline_batch_size
    ):
        batch_parts.append(offline_replay.sample(batch_size=offline_batch_size, device=device))
        used_offline_batch = offline_batch_size

    if not batch_parts:
        if len(online_replay) >= fallback_batch_size:
            batch_parts.append(online_replay.sample(batch_size=fallback_batch_size, device=device))
            used_online_batch = fallback_batch_size
        elif offline_replay is not None and len(offline_replay) >= fallback_batch_size:
            batch_parts.append(offline_replay.sample(batch_size=fallback_batch_size, device=device))
            used_offline_batch = fallback_batch_size

    if not batch_parts:
        return None, 0, 0

    return _concat_batches(batch_parts), used_online_batch, used_offline_batch


def main() -> None:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    output_root = (repo_root / args.output_root).resolve()
    device = coerce_device(args.device)
    set_seed(args.seed)

    # Build base policy adapters (separate train/eval buffers).
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

    run_name = args.run_name or "default"
    run_dir = build_run_dir(output_root=output_root, suite_name=suite_name, run_name=run_name)
    ckpt_dir = run_dir / "snapshot"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    task_name = args.task_name or train_base.cfg["suite"]["task"]["task_name"]

    run_meta = {
        "bc_weight": str(Path(args.bc_weight).resolve()),
        "suite": suite_name,
        "task_desc": task_desc,
        "task_name": task_name,
        "pixel_key": pixel_key,
        "device": device,
        "low": low.tolist(),
        "high": high.tolist(),
        "args": vars(args),
    }

    residual_scale = np.array(
        [
            args.residual_pos_scale,
            args.residual_pos_scale,
            args.residual_pos_scale,
            args.residual_rot_scale,
            args.residual_rot_scale,
            args.residual_rot_scale,
            args.residual_gripper_scale,
        ],
        dtype=np.float32,
    )

    # Bootstrap state dim from first observation.
    time_step = train_env.reset()
    obs = time_step.observation
    train_base.reset_episode()
    step_in_ep = 0
    base_action_dict = train_base.act(obs, step_in_ep, 0)
    base_action = np.asarray(train_env.point2action(base_action_dict), dtype=np.float32).reshape(7)
    state = observation_to_state(
        obs=obs,
        pixel_key=pixel_key,
        base_action_7d=base_action,
        include_eef_pos=args.include_eef_pos,
    )
    obs_dim = int(state.shape[0])

    td3_cfg = TD3ConfigRL(
        obs_dim=obs_dim,
        action_dim=7,
        device=device,
        actor_hidden_dim=args.actor_hidden_dim,
        critic_hidden_dim=args.critic_hidden_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        tau=args.tau,
        policy_noise=args.policy_noise,
        noise_clip=args.noise_clip,
        actor_update_freq=args.actor_update_freq,
    )
    td3_agent = ResidualTD3AgentRL(cfg=td3_cfg, action_scale=residual_scale)
    replay = ReplayBufferRL(obs_dim=obs_dim, action_dim=7, capacity=args.buffer_size)

    num_updates_per_step = max(1, int(args.num_updates_per_step))
    update_every_n_steps = max(1, int(args.update_every_n_steps))
    if num_updates_per_step != int(args.num_updates_per_step):
        print(
            f"[train] clipped num_updates_per_step from {args.num_updates_per_step} "
            f"to {num_updates_per_step}"
        )
    if update_every_n_steps != int(args.update_every_n_steps):
        print(
            f"[train] clipped update_every_n_steps from {args.update_every_n_steps} "
            f"to {update_every_n_steps}"
        )

    requested_offline_fraction = float(args.offline_fraction)
    offline_fraction = float(np.clip(requested_offline_fraction, 0.0, 1.0))
    if abs(offline_fraction - requested_offline_fraction) > 1e-8:
        print(
            f"[offline] clipped offline_fraction from {requested_offline_fraction} "
            f"to {offline_fraction}"
        )
    offline_batch_size = int(args.batch_size * offline_fraction)
    if offline_fraction > 0.0 and offline_batch_size == 0:
        offline_batch_size = 1
    online_batch_size = int(args.batch_size - offline_batch_size)
    offline_replay: ReplayBufferRL | None = None
    offline_transition_count = 0

    if args.offline_demo_root.strip():
        offline_demo_root = Path(args.offline_demo_root).expanduser().resolve()
    else:
        offline_demo_root = (repo_root / "expert_demos" / suite_name).resolve()

    if offline_fraction > 0.0:
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

            def _bc_action(obs_step: dict, step_idx: int) -> dict:
                return offline_base.act(obs_step, step_idx, step_idx)

            bc_reset_episode_fn = _bc_reset_episode
            bc_action_fn = _bc_action

        try:
            offline_transitions = build_offline_transitions_from_expert_demos_rl(
                demo_root=offline_demo_root,
                suite_name=suite_name,
                task_name=task_name,
                pixel_key=pixel_key,
                include_eef_pos=args.include_eef_pos,
                low=low,
                high=high,
                max_demos=max_demos,
                terminal_reward=args.offline_terminal_reward,
                step_reward=args.offline_step_reward,
                base_action_mode=args.offline_base_action_mode,
                bc_action_fn=bc_action_fn,
                bc_reset_episode_fn=bc_reset_episode_fn,
            )
        except Exception as exc:
            print(
                f"[offline] failed to build transitions from {offline_demo_root}: {exc}. "
                "Fallback to online-only updates."
            )
            offline_transitions = []

        if offline_transitions:
            offline_replay = ReplayBufferRL(
                obs_dim=obs_dim,
                action_dim=7,
                capacity=max(len(offline_transitions), 1),
            )
            for transition in offline_transitions:
                offline_replay.add(
                    obs=transition["obs"],
                    action=transition["action"],
                    reward=float(transition["reward"]),
                    next_obs=transition["next_obs"],
                    done=bool(transition["done"]),
                )
            offline_transition_count = len(offline_replay)
            print(
                json.dumps(
                    {
                        "offline_enabled": True,
                        "offline_demo_root": str(offline_demo_root),
                        "offline_transition_count": offline_transition_count,
                        "offline_batch_size": offline_batch_size,
                        "online_batch_size": online_batch_size,
                    }
                )
            )
        else:
            offline_fraction = 0.0
            offline_batch_size = 0
            online_batch_size = args.batch_size
            print(
                json.dumps(
                    {
                        "offline_enabled": False,
                        "offline_reason": "no_transitions",
                        "offline_demo_root": str(offline_demo_root),
                    }
                )
            )

    run_meta["offline"] = {
        "fraction_requested": requested_offline_fraction,
        "fraction_applied": offline_fraction,
        "demo_root": str(offline_demo_root),
        "base_action_mode": args.offline_base_action_mode,
        "transitions": int(offline_transition_count),
        "offline_batch_size": int(offline_batch_size),
        "online_batch_size": int(online_batch_size),
        "offline_max_demos": int(args.offline_max_demos),
        "offline_step_reward": float(args.offline_step_reward),
        "offline_terminal_reward": float(args.offline_terminal_reward),
    }
    run_meta["update_schedule"] = {
        "warmup_steps": int(args.warmup_steps),
        "num_updates_per_step": int(num_updates_per_step),
        "update_every_n_steps": int(update_every_n_steps),
    }
    dump_json(run_dir / "run_meta.json", run_meta)

    train_csv = run_dir / "train_log.csv"
    eval_csv = run_dir / "eval_log.csv"

    episode_return = 0.0
    episode_len = 0
    episode_idx = 0
    success_flag = 0

    # Precompute first base action dict to keep BC history consistent.
    cached_base_action_dict = base_action_dict

    for global_step in range(1, args.steps + 1):
        # Current base action from cache.
        base_action_dict = cached_base_action_dict
        base_action = np.asarray(train_env.point2action(base_action_dict), dtype=np.float32).reshape(7)

        state = observation_to_state(
            obs=obs,
            pixel_key=pixel_key,
            base_action_7d=base_action,
            include_eef_pos=args.include_eef_pos,
        )

        if global_step <= args.warmup_steps:
            residual_action = np.random.uniform(-residual_scale, residual_scale).astype(np.float32)
        else:
            residual_action = td3_agent.select_action(
                state,
                exploration_std=args.exploration_std,
                deterministic=False,
            )

        env_action = clip_action(base_action + residual_action, low=low, high=high)
        time_step_next = train_env.step(env_action)
        next_obs = time_step_next.observation
        reward = float(time_step_next.reward)
        done = bool(time_step_next.last())

        episode_return += reward
        episode_len += 1
        if bool(next_obs.get("goal_achieved", False)):
            success_flag = 1

        if done:
            next_base_action = np.zeros((7,), dtype=np.float32)
            cached_base_action_dict = None
        else:
            next_step_in_ep = step_in_ep + 1
            next_base_action_dict = train_base.act(next_obs, next_step_in_ep, global_step)
            next_base_action = np.asarray(
                train_env.point2action(next_base_action_dict), dtype=np.float32
            ).reshape(7)
            cached_base_action_dict = next_base_action_dict

        next_state = observation_to_state(
            obs=next_obs,
            pixel_key=pixel_key,
            base_action_7d=next_base_action,
            include_eef_pos=args.include_eef_pos,
        )

        replay.add(
            obs=state,
            action=residual_action,
            reward=reward,
            next_obs=next_state,
            done=done,
        )

        metrics = {}
        used_online_batch = 0
        used_offline_batch = 0
        num_updates_done = 0
        if (
            global_step > args.warmup_steps
            and (global_step % update_every_n_steps == 0 or global_step == 1)
        ):
            for _ in range(num_updates_per_step):
                mixed_batch, cur_online, cur_offline = _sample_mixed_batch(
                    online_replay=replay,
                    offline_replay=offline_replay,
                    online_batch_size=online_batch_size,
                    offline_batch_size=offline_batch_size,
                    fallback_batch_size=args.batch_size,
                    device=device,
                )
                if mixed_batch is None:
                    break
                metrics = td3_agent.update_from_batch(mixed_batch)
                used_online_batch = cur_online
                used_offline_batch = cur_offline
                num_updates_done += 1

        if global_step % args.log_every == 0:
            row = {
                "step": global_step,
                "buffer_size": len(replay),
                "offline_buffer_size": len(offline_replay) if offline_replay is not None else 0,
                "episode_return": episode_return,
                "episode_len": episode_len,
                "success": success_flag,
                "offline_fraction": offline_fraction,
                "num_updates_done": num_updates_done,
                "used_online_batch": used_online_batch,
                "used_offline_batch": used_offline_batch,
                "critic_loss": metrics.get("critic_loss", 0.0),
                "actor_loss": metrics.get("actor_loss", 0.0),
                "q1_mean": metrics.get("q1_mean", 0.0),
                "q2_mean": metrics.get("q2_mean", 0.0),
            }
            _write_csv_row(train_csv, row)
            print(json.dumps(row))

        if args.eval_every > 0 and global_step % args.eval_every == 0:
            eval_metrics = _evaluate(
                env=eval_env,
                base_policy=eval_base,
                td3_agent=td3_agent,
                episodes=args.eval_episodes,
                pixel_key=pixel_key,
                low=low,
                high=high,
                include_eef_pos=args.include_eef_pos,
            )
            eval_row = {"step": global_step, **eval_metrics}
            _write_csv_row(eval_csv, eval_row)
            print(json.dumps(eval_row))

        if global_step % args.save_every == 0:
            import torch

            payload = {
                "step": global_step,
                "td3": td3_agent.state_dict(),
                "obs_dim": obs_dim,
                "action_dim": 7,
                "residual_scale": residual_scale.tolist(),
                "low": low.tolist(),
                "high": high.tolist(),
                "include_eef_pos": bool(args.include_eef_pos),
                "bc_weight": str(Path(args.bc_weight).resolve()),
                "suite": suite_name,
                "task_name": task_name,
                "pixel_key": pixel_key,
                "offline_fraction": offline_fraction,
                "offline_demo_root": str(offline_demo_root),
                "offline_base_action_mode": args.offline_base_action_mode,
                "num_updates_per_step": num_updates_per_step,
                "update_every_n_steps": update_every_n_steps,
                "args": vars(args),
            }
            ckpt_path = ckpt_dir / f"{global_step}.pt"
            torch.save(payload, ckpt_path)
            torch.save(payload, ckpt_dir / "latest.pt")

        if done:
            ep_row = {
                "step": global_step,
                "episode": episode_idx,
                "episode_return": episode_return,
                "episode_len": episode_len,
                "success": success_flag,
                "buffer_size": len(replay),
                "offline_buffer_size": len(offline_replay) if offline_replay is not None else 0,
                "offline_fraction": offline_fraction,
                "num_updates_done": num_updates_done,
                "used_online_batch": used_online_batch,
                "used_offline_batch": used_offline_batch,
                "critic_loss": metrics.get("critic_loss", 0.0),
                "actor_loss": metrics.get("actor_loss", 0.0),
                "q1_mean": metrics.get("q1_mean", 0.0),
                "q2_mean": metrics.get("q2_mean", 0.0),
            }
            _write_csv_row(train_csv, ep_row)

            episode_idx += 1
            episode_return = 0.0
            episode_len = 0
            success_flag = 0

            time_step = train_env.reset()
            obs = time_step.observation
            train_base.reset_episode()
            step_in_ep = 0
            cached_base_action_dict = train_base.act(obs, step_in_ep, global_step)
        else:
            obs = next_obs
            step_in_ep += 1

    # Final checkpoint.
    import torch

    final_payload = {
        "step": args.steps,
        "td3": td3_agent.state_dict(),
        "obs_dim": obs_dim,
        "action_dim": 7,
        "residual_scale": residual_scale.tolist(),
        "low": low.tolist(),
        "high": high.tolist(),
        "include_eef_pos": bool(args.include_eef_pos),
        "bc_weight": str(Path(args.bc_weight).resolve()),
        "suite": suite_name,
        "task_name": task_name,
        "pixel_key": pixel_key,
        "offline_fraction": offline_fraction,
        "offline_demo_root": str(offline_demo_root),
        "offline_base_action_mode": args.offline_base_action_mode,
        "num_updates_per_step": num_updates_per_step,
        "update_every_n_steps": update_every_n_steps,
        "args": vars(args),
    }
    torch.save(final_payload, ckpt_dir / "final.pt")
    torch.save(final_payload, ckpt_dir / "latest.pt")


if __name__ == "__main__":
    main()
