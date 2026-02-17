#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None


def _load_demo_pkl(pkl_path: Path) -> List[Dict[str, Any]]:
    with pkl_path.open("rb") as f:
        payload = pickle.load(f)
    if not isinstance(payload, dict) or "observations" not in payload:
        raise ValueError(f"Unsupported pkl format: {pkl_path}")
    observations = payload["observations"]
    if not isinstance(observations, list):
        raise ValueError(f"'observations' must be a list in: {pkl_path}")
    return observations


def _iter_hdf5_demo_groups(h5: "h5py.File"):
    if "data" in h5 and isinstance(h5["data"], h5py.Group):
        keys = sorted(h5["data"].keys())
        for k in keys:
            obj = h5["data"][k]
            if isinstance(obj, h5py.Group):
                yield f"data/{k}", obj
        return
    for k in sorted(h5.keys()):
        obj = h5[k]
        if isinstance(obj, h5py.Group):
            yield k, obj


def _load_hdf5_actions(hdf5_path: Optional[Path]) -> Optional[List[np.ndarray]]:
    if hdf5_path is None:
        return None
    if h5py is None:
        raise ImportError("h5py is required when --hdf5-path is used")

    actions: List[np.ndarray] = []
    with h5py.File(str(hdf5_path), "r") as h5:
        # Case 1: root-level 3D actions [E, T, A]
        if "actions" in h5 and isinstance(h5["actions"], h5py.Dataset):
            arr = np.asarray(h5["actions"])
            if arr.ndim == 3:
                for e in range(arr.shape[0]):
                    actions.append(np.asarray(arr[e], dtype=np.float32))
                return actions

        # Case 2: robomimic-like groups with per-demo actions
        for _, group in _iter_hdf5_demo_groups(h5):
            if "actions" in group and isinstance(group["actions"], h5py.Dataset):
                act = np.asarray(group["actions"], dtype=np.float32)
                if act.ndim == 2:
                    actions.append(act)
        if actions:
            return actions

    raise ValueError(f"Could not find action datasets in hdf5: {hdf5_path}")


def _episode_horizon(obs_episode: Dict[str, Any]) -> int:
    lengths: List[int] = []
    for _, value in obs_episode.items():
        arr = np.asarray(value)
        if arr.ndim >= 1 and arr.shape[0] > 1:
            lengths.append(int(arr.shape[0]))
    return min(lengths) if lengths else 0


def _slice_obs_step(obs_episode: Dict[str, Any], t: int) -> Dict[str, Any]:
    step_obs: Dict[str, Any] = {}
    for key, value in obs_episode.items():
        arr = np.asarray(value)
        if arr.ndim == 0 or arr.shape[0] <= t:
            continue
        step_obs[key] = np.asarray(arr[t])
    return step_obs


def _build_episode_steps(
    obs_episode: Dict[str, Any],
    action_seq: Optional[np.ndarray],
    step_reward: float,
    terminal_reward: float,
) -> List[Dict[str, Any]]:
    horizon = _episode_horizon(obs_episode)
    if horizon < 2:
        return []

    if action_seq is not None and action_seq.ndim == 2:
        if action_seq.shape[0] == horizon:
            num_transitions = horizon
        else:
            num_transitions = min(horizon - 1, int(action_seq.shape[0]))
    else:
        num_transitions = horizon - 1

    steps: List[Dict[str, Any]] = []
    for t in range(num_transitions):
        is_first = t == 0
        is_last = t == (num_transitions - 1)
        is_terminal = is_last
        reward = float(terminal_reward if is_terminal else step_reward)
        discount = 0.0 if is_terminal else 1.0

        action = None
        if action_seq is not None and action_seq.ndim == 2 and t < action_seq.shape[0]:
            action = np.asarray(action_seq[t], dtype=np.float32)

        step = {
            "is_first": bool(is_first),
            "is_last": bool(is_last),
            "is_terminal": bool(is_terminal),
            "is_truncated": False,
            "reward": reward,
            "discount": discount,
            "observation": _slice_obs_step(obs_episode, t),
            "action": action,
        }
        steps.append(step)
    return steps


def build_rlds_dataset(
    pkl_path: Path,
    hdf5_path: Optional[Path],
    output_path: Path,
    step_reward: float,
    terminal_reward: float,
) -> Dict[str, Any]:
    observations = _load_demo_pkl(pkl_path)
    hdf5_actions = _load_hdf5_actions(hdf5_path)

    episode_count = len(observations)
    if hdf5_actions is not None:
        episode_count = min(episode_count, len(hdf5_actions))

    episodes: List[Dict[str, Any]] = []
    lengths: List[int] = []
    returns: List[float] = []
    action_norms: List[float] = []

    for ep_idx in range(episode_count):
        obs_episode = observations[ep_idx]
        if not isinstance(obs_episode, dict):
            continue

        action_seq = None
        if hdf5_actions is not None:
            action_seq = hdf5_actions[ep_idx]
        else:
            for k in ("actions", "action"):
                if k in obs_episode:
                    arr = np.asarray(obs_episode[k], dtype=np.float32)
                    if arr.ndim == 2:
                        action_seq = arr
                        break

        steps = _build_episode_steps(
            obs_episode=obs_episode,
            action_seq=action_seq,
            step_reward=step_reward,
            terminal_reward=terminal_reward,
        )
        if not steps:
            continue

        ep_return = float(np.sum([s["reward"] for s in steps]))
        lengths.append(len(steps))
        returns.append(ep_return)
        for s in steps:
            if s["action"] is not None:
                action_norms.append(float(np.linalg.norm(s["action"])))

        episodes.append(
            {
                "episode_id": int(ep_idx),
                "steps": steps,
                "metadata": {
                    "source_pkl_episode_idx": int(ep_idx),
                    "action_source": "hdf5" if hdf5_actions is not None else "pkl_or_none",
                },
            }
        )

    dataset = {
        "format": "rlds_like",
        "source": {
            "pkl_path": str(pkl_path),
            "hdf5_path": str(hdf5_path) if hdf5_path is not None else None,
        },
        "reward_scheme": {
            "step_reward": float(step_reward),
            "terminal_reward": float(terminal_reward),
            "policy_note": "done=1, fail=0, truncate=0",
        },
        "episodes": episodes,
        "stats": {
            "num_episodes": int(len(episodes)),
            "mean_episode_len": float(np.mean(lengths)) if lengths else 0.0,
            "max_episode_len": int(np.max(lengths)) if lengths else 0,
            "min_episode_len": int(np.min(lengths)) if lengths else 0,
            "mean_episode_return": float(np.mean(returns)) if returns else 0.0,
            "action_available_ratio": float(len(action_norms) / max(1, int(np.sum(lengths)))),
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
    return dataset


def make_visuals(dataset: Dict[str, Any], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = dataset["episodes"]
    lengths = np.asarray([len(ep["steps"]) for ep in episodes], dtype=np.int32)
    returns = np.asarray([np.sum([s["reward"] for s in ep["steps"]]) for ep in episodes], dtype=np.float32)
    action_norms = []
    for ep in episodes:
        for s in ep["steps"]:
            if s["action"] is not None:
                action_norms.append(float(np.linalg.norm(s["action"])))
    action_norms = np.asarray(action_norms, dtype=np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    if lengths.size > 0:
        axes[0].hist(lengths, bins=min(30, max(5, len(lengths) // 2)))
    axes[0].set_title("Episode Length")
    axes[0].set_xlabel("steps")
    axes[0].set_ylabel("count")
    axes[0].grid(alpha=0.25)

    if returns.size > 0:
        axes[1].plot(np.arange(len(returns)), returns, marker="o", linewidth=1.0)
    axes[1].set_title("Episode Return")
    axes[1].set_xlabel("episode index")
    axes[1].set_ylabel("return")
    axes[1].grid(alpha=0.25)

    if action_norms.size > 0:
        axes[2].hist(action_norms, bins=30)
    axes[2].set_title("Action Norm")
    axes[2].set_xlabel("||a||")
    axes[2].set_ylabel("count")
    axes[2].grid(alpha=0.25)

    fig.tight_layout()
    fig_path = output_dir / "rlds_summary.png"
    fig.savefig(fig_path, dpi=180)
    plt.close(fig)

    # Draw a simple flow chart so users can verify the implemented structure at a glance.
    fig2, ax = plt.subplots(figsize=(11, 3))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    boxes = [
        (0.02, 0.3, 0.22, 0.4, "PKL\n(point/features)"),
        (0.28, 0.3, 0.22, 0.4, "HDF5\n(actions)"),
        (0.54, 0.3, 0.22, 0.4, "RLDS Builder\n(done=1/fail=0)"),
        (0.80, 0.3, 0.18, 0.4, "RLDS-like\npickle"),
    ]
    for x, y, w, h, txt in boxes:
        rect = plt.Rectangle((x, y), w, h, facecolor="#e5e7eb", edgecolor="#1f2937", linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center")
    for x1, x2 in [(0.24, 0.28), (0.50, 0.54), (0.76, 0.80)]:
        ax.annotate("", xy=(x2, 0.5), xytext=(x1, 0.5), arrowprops=dict(arrowstyle="->", lw=1.5))
    flow_path = output_dir / "rlds_build_flow.png"
    fig2.tight_layout()
    fig2.savefig(flow_path, dpi=180)
    plt.close(fig2)

    report_md = output_dir / "RLDS_BUILD_REPORT.md"
    with report_md.open("w", encoding="utf-8") as f:
        f.write("# RLDS Build Report\n\n")
        f.write("## Build Flow\n")
        f.write("![flow](rlds_build_flow.png)\n\n")
        f.write("## Dataset Summary\n")
        f.write("![summary](rlds_summary.png)\n\n")
        f.write("## Stats\n")
        f.write("```json\n")
        f.write(json.dumps(dataset["stats"], indent=2))
        f.write("\n```\n")


def main():
    parser = argparse.ArgumentParser(description="Build RLDS-like dataset from hdf5 + pkl")
    parser.add_argument("--pkl-path", required=True, type=Path)
    parser.add_argument("--hdf5-path", default=None, type=Path)
    parser.add_argument("--output-path", required=True, type=Path, help="Output .pkl path for RLDS-like dataset")
    parser.add_argument("--step-reward", default=0.0, type=float)
    parser.add_argument("--terminal-reward", default=1.0, type=float)
    parser.add_argument("--viz-dir", default=None, type=Path, help="Directory for png/md visualization outputs")
    args = parser.parse_args()

    dataset = build_rlds_dataset(
        pkl_path=args.pkl_path,
        hdf5_path=args.hdf5_path,
        output_path=args.output_path,
        step_reward=args.step_reward,
        terminal_reward=args.terminal_reward,
    )

    stats_path = args.output_path.with_suffix(".stats.json")
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(dataset["stats"], f, indent=2)

    if args.viz_dir is not None:
        make_visuals(dataset, args.viz_dir)
        print(f"[ok] visuals: {args.viz_dir}")

    print(f"[ok] dataset: {args.output_path}")
    print(f"[ok] stats: {stats_path}")
    print(f"[ok] episodes: {dataset['stats']['num_episodes']}")


if __name__ == "__main__":
    main()
