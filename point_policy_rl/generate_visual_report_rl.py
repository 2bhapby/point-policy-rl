#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch


@dataclass
class RunData:
    run_dir: str
    run_id: str
    args: Dict
    train_df: Optional[pd.DataFrame]
    eval_df: Optional[pd.DataFrame]


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_run(run_dir: str) -> RunData:
    run_id = os.path.basename(run_dir.rstrip("/"))
    meta = load_json(os.path.join(run_dir, "run_meta.json"))
    args = meta.get("args", {})

    train_path = os.path.join(run_dir, "train_log.csv")
    eval_path = os.path.join(run_dir, "eval_log.csv")
    train_df = pd.read_csv(train_path) if os.path.exists(train_path) else None
    eval_df = pd.read_csv(eval_path) if os.path.exists(eval_path) else None
    if train_df is not None:
        train_df = clean_train_df(train_df)
    return RunData(run_dir=run_dir, run_id=run_id, args=args, train_df=train_df, eval_df=eval_df)


def clean_train_df(df: pd.DataFrame) -> pd.DataFrame:
    keep = df.copy()

    # Remove malformed rows occasionally mixed into csv during long runs.
    if "offline_fraction" in keep:
        keep = keep[(keep["offline_fraction"] >= 0.0) & (keep["offline_fraction"] <= 1.0)]
    if "success" in keep:
        keep = keep[(keep["success"] >= 0) & (keep["success"] <= 1)]
    if "used_online_batch" in keep:
        keep = keep[~keep["used_online_batch"].isna()]
    if "step" in keep:
        keep = keep[keep["step"] >= 0]

    keep = keep.dropna(subset=["step"]).sort_values("step").reset_index(drop=True)
    return keep


def _box(ax, x, y, w, h, text, color="#E8F0FE"):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        linewidth=1.2,
        facecolor=color,
        edgecolor="#334155",
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)


def _arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.5, color="#0f172a"))


def plot_architecture(out_path: str):
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, 0.03, 0.62, 0.2, 0.22, "LIBERO Env\n(obs: point/features)")
    _box(ax, 0.29, 0.62, 0.2, 0.22, "Frozen Point-Policy BC\n(base_policy_adapter_rl.py)")
    _box(ax, 0.55, 0.62, 0.2, 0.22, "Residual Actor/Critic\n(ResFiT TD3-style)")
    _box(ax, 0.80, 0.62, 0.17, 0.22, "Action Merge\nbase + residual")

    _box(ax, 0.29, 0.22, 0.2, 0.22, "Offline Demo Builder\n(offline_demo_builder_rl.py)", color="#ECFDF5")
    _box(ax, 0.55, 0.22, 0.2, 0.22, "Replay Mixer\noffline + online", color="#ECFDF5")
    _box(ax, 0.80, 0.22, 0.17, 0.22, "Q Update\nn-step / subset-min", color="#ECFDF5")

    _arrow(ax, 0.23, 0.73, 0.29, 0.73)
    _arrow(ax, 0.49, 0.73, 0.55, 0.73)
    _arrow(ax, 0.75, 0.73, 0.80, 0.73)
    _arrow(ax, 0.88, 0.62, 0.12, 0.62)

    _arrow(ax, 0.39, 0.62, 0.39, 0.44)
    _arrow(ax, 0.65, 0.62, 0.65, 0.44)
    _arrow(ax, 0.49, 0.33, 0.55, 0.33)
    _arrow(ax, 0.75, 0.33, 0.80, 0.33)
    _arrow(ax, 0.88, 0.44, 0.65, 0.62)

    ax.set_title("Point-Policy Residual RL Pipeline (Current _rl Implementation)", fontsize=13, pad=18)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_training_curves(runs: List[RunData], out_path: str):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=False)

    for run in runs:
        if run.train_df is not None and len(run.train_df) > 0:
            x = run.train_df["step"]
            if "actor_loss" in run.train_df:
                axes[0].plot(x, run.train_df["actor_loss"], label=run.run_id, linewidth=1.2)
            if "critic_loss" in run.train_df:
                axes[1].plot(x, run.train_df["critic_loss"], label=run.run_id, linewidth=1.2)

        if run.eval_df is not None and len(run.eval_df) > 0 and "eval/success" in run.eval_df:
            axes[2].plot(run.eval_df["step"], run.eval_df["eval/success"], marker="o", label=run.run_id, linewidth=1.2)

    axes[0].set_title("Actor Loss")
    axes[1].set_title("Critic Loss")
    axes[2].set_title("Eval Success")
    axes[2].set_ylim(-0.02, 1.02)

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")
        ax.set_xlabel("step")

    axes[1].set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_buffer_mix(runs: List[RunData], out_path: str):
    labels = [r.run_id for r in runs]
    online_sizes = []
    offline_sizes = []
    online_batch = []
    offline_batch = []

    for r in runs:
        if r.train_df is None or len(r.train_df) == 0:
            online_sizes.append(0)
            offline_sizes.append(0)
            online_batch.append(0)
            offline_batch.append(0)
            continue
        last = r.train_df.iloc[-1]
        online_sizes.append(float(last.get("online_buffer_size", 0)))
        offline_sizes.append(float(last.get("offline_buffer_size", 0)))
        online_batch.append(float(r.train_df.get("used_online_batch", pd.Series([0])).mean()))
        offline_batch.append(float(r.train_df.get("used_offline_batch", pd.Series([0])).mean()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = range(len(labels))
    width = 0.35

    axes[0].bar([i - width / 2 for i in x], online_sizes, width=width, label="online_buffer")
    axes[0].bar([i + width / 2 for i in x], offline_sizes, width=width, label="offline_buffer")
    axes[0].set_title("Buffer Size (last step)")
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar([i - width / 2 for i in x], online_batch, width=width, label="used_online_batch")
    axes[1].bar([i + width / 2 for i in x], offline_batch, width=width, label="used_offline_batch")
    axes[1].set_title("Batch Mix (mean per update)")
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def summarize_run(run: RunData) -> Dict:
    result = {
        "run_id": run.run_id,
        "run_dir": run.run_dir,
        "steps_cfg": run.args.get("steps"),
        "offline_fraction_cfg": run.args.get("offline_fraction"),
        "offline_max_demos_cfg": run.args.get("offline_max_demos"),
        "offline_base_action_mode": run.args.get("offline_base_action_mode"),
        "transition_action_mode": run.args.get("transition_action_mode"),
        "residual_action_scale": run.args.get("residual_action_scale"),
        "random_action_noise_scale": run.args.get("random_action_noise_scale"),
        "tau": run.args.get("critic_target_tau"),
    }
    if run.train_df is not None and len(run.train_df) > 0:
        last = run.train_df.iloc[-1]
        result.update(
            {
                "train_last_step": int(last.get("step", 0)),
                "online_buffer_last": int(last.get("online_buffer_size", 0)),
                "offline_buffer_last": int(last.get("offline_buffer_size", 0)),
                "train_success_mean": float(run.train_df.get("success", pd.Series([0])).mean()),
                "critic_loss_last": float(last.get("critic_loss", 0)),
                "actor_loss_last": float(last.get("actor_loss", 0)),
            }
        )
    else:
        result["train_last_step"] = None

    if run.eval_df is not None and len(run.eval_df) > 0 and "eval/success" in run.eval_df:
        eval_last = run.eval_df.iloc[-1]
        result.update(
            {
                "eval_last_step": int(eval_last.get("step", 0)),
                "eval_success_last": float(eval_last.get("eval/success", 0)),
            }
        )
    else:
        result["eval_last_step"] = None
        result["eval_success_last"] = None
    return result


def write_report_md(out_path: str, summaries: List[Dict], png_arch: str, png_curve: str, png_mix: str):
    lines = []
    lines.append("# RL Visual Verification Report")
    lines.append("")
    lines.append("아래 자료는 현재 `_rl` 구현이 의도한 구조(BC base + residual RL + replay 혼합)로 동작하는지 빠르게 점검하기 위한 시각화입니다.")
    lines.append("")
    lines.append("## 1) Pipeline 구조")
    lines.append(f"![architecture]({os.path.basename(png_arch)})")
    lines.append("")
    lines.append("## 2) 학습/평가 추세")
    lines.append(f"![curves]({os.path.basename(png_curve)})")
    lines.append("")
    lines.append("## 3) 버퍼/배치 혼합 비율")
    lines.append(f"![mix]({os.path.basename(png_mix)})")
    lines.append("")
    lines.append("## 4) Run 요약")
    lines.append("| run_id | train_last_step | eval_success_last | offline_fraction_cfg | offline_max_demos_cfg | offline_buffer_last | online_buffer_last |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for s in summaries:
        lines.append(
            f"| {s.get('run_id')} | {s.get('train_last_step')} | {s.get('eval_success_last')} | "
            f"{s.get('offline_fraction_cfg')} | {s.get('offline_max_demos_cfg')} | "
            f"{s.get('offline_buffer_last')} | {s.get('online_buffer_last')} |"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-dir", action="append", required=True, help="Run directory containing run_meta.json")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    runs = [load_run(d) for d in args.run_dir]
    summaries = [summarize_run(r) for r in runs]

    arch_path = os.path.join(args.output_dir, "architecture_overview.png")
    curve_path = os.path.join(args.output_dir, "training_curves.png")
    mix_path = os.path.join(args.output_dir, "buffer_mix.png")
    report_path = os.path.join(args.output_dir, "VISUAL_REPORT_RL.md")
    summary_json = os.path.join(args.output_dir, "visual_summary.json")

    plot_architecture(arch_path)
    plot_training_curves(runs, curve_path)
    plot_buffer_mix(runs, mix_path)
    write_report_md(report_path, summaries, arch_path, curve_path, mix_path)
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump({"runs": summaries}, f, indent=2)

    print(f"[ok] report: {report_path}")
    print(f"[ok] summary: {summary_json}")
    print(f"[ok] images: {arch_path}, {curve_path}, {mix_path}")


if __name__ == "__main__":
    main()
