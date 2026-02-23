#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from subprocess import TimeoutExpired
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep RL checkpoints and build success-rate curve (REL eval script)."
    )
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--resfit-root",
        type=str,
        default="/sjw_alinlab2/home/sanghyeok/residual-offpolicy-rl",
    )

    parser.add_argument("--suite", type=str, default="")
    parser.add_argument("--task-name", type=str, default="")
    parser.add_argument("--benchmark-name", type=str, default="")
    parser.add_argument("--task-order-index", type=int, default=-1)
    parser.add_argument("--state-coord-mode", type=str, default="")
    parser.add_argument("--image-size-override", type=int, default=84)

    parser.add_argument("--min-step", type=int, default=0)
    parser.add_argument("--max-step", type=int, default=0)
    parser.add_argument("--ckpt-stride", type=int, default=10000)
    parser.add_argument("--timeout-sec", type=int, default=2400)

    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-dir", type=str, default="")
    parser.add_argument("--video-fps", type=int, default=20)
    parser.add_argument("--video-render-size", type=int, default=256)

    parser.add_argument("--output-tag", type=str, default="")
    parser.add_argument("--output-dir", type=str, default="")
    return parser.parse_args()


def _load_run_meta(run_dir: Path) -> dict[str, Any]:
    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        return {}
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _infer_defaults(args: argparse.Namespace, meta: dict[str, Any]) -> argparse.Namespace:
    cfg = dict(meta.get("args", {})) if isinstance(meta.get("args", {}), dict) else {}

    if not args.suite and cfg.get("suite"):
        args.suite = str(cfg.get("suite"))
    if not args.task_name and cfg.get("task_name"):
        args.task_name = str(cfg.get("task_name"))
    if not args.benchmark_name and cfg.get("benchmark_name"):
        args.benchmark_name = str(cfg.get("benchmark_name"))
    if args.task_order_index < 0 and isinstance(cfg.get("task_order_index"), (int, float)):
        args.task_order_index = int(cfg.get("task_order_index"))
    if not args.state_coord_mode and cfg.get("state_coord_mode"):
        args.state_coord_mode = str(cfg.get("state_coord_mode"))
    if args.image_size_override <= 0 and isinstance(cfg.get("dummy_image_size"), (int, float)):
        args.image_size_override = int(cfg.get("dummy_image_size"))
    if not args.output_tag:
        args.output_tag = str(cfg.get("run_name") or run_dir.name)

    return args


def _find_checkpoints(snapshot_dir: Path, min_step: int, max_step: int, stride: int) -> list[Path]:
    pts: list[tuple[int, Path]] = []
    for ckpt in snapshot_dir.glob("*.pt"):
        stem = ckpt.stem
        if not stem.isdigit():
            continue
        step = int(stem)
        if step < int(min_step):
            continue
        if int(max_step) > 0 and step > int(max_step):
            continue
        if int(stride) > 0 and (step % int(stride) != 0):
            continue
        pts.append((step, ckpt))
    pts.sort(key=lambda x: x[0])
    return [p for _, p in pts]


def _extract_last_json_block(text: str) -> dict[str, Any] | None:
    lines = text.splitlines()
    if not lines:
        return None

    buf: list[str] = []
    brace_balance = 0
    started = False
    for line in reversed(lines):
        if not started and "}" not in line:
            continue
        started = True
        buf.append(line)
        brace_balance += line.count("}")
        brace_balance -= line.count("{")
        if brace_balance == 0 and "{" in line:
            break

    if not buf:
        return None
    maybe_json = "\n".join(reversed(buf)).strip()
    try:
        return json.loads(maybe_json)
    except Exception:
        return None


def _build_eval_cmd(
    repo_root: Path,
    ckpt: Path,
    args: argparse.Namespace,
    video_dir_for_ckpt: Path | None,
) -> list[str]:
    cmd = [
        sys.executable,
        str(repo_root / "point_policy_rl" / "eval_residual_td3_rl_rel.py"),
        "--ckpt",
        str(ckpt),
        "--episodes",
        str(int(args.episodes)),
        "--device",
        str(args.device),
        "--resfit-root",
        str(args.resfit_root),
    ]
    if args.suite:
        cmd += ["--suite", str(args.suite)]
    if args.task_name:
        cmd += ["--task-name", str(args.task_name)]
    if args.benchmark_name:
        cmd += ["--benchmark-name", str(args.benchmark_name)]
    if args.task_order_index >= 0:
        cmd += ["--task-order-index", str(int(args.task_order_index))]
    if args.state_coord_mode:
        cmd += ["--state-coord-mode", str(args.state_coord_mode)]
    if int(args.image_size_override) > 0:
        cmd += ["--image-size-override", str(int(args.image_size_override))]
    if args.save_video:
        cmd += ["--save-video"]
        cmd += ["--video-fps", str(int(args.video_fps))]
        cmd += ["--video-render-size", str(int(args.video_render_size))]
        if video_dir_for_ckpt is not None:
            cmd += ["--video-dir", str(video_dir_for_ckpt)]
            cmd += ["--video-tag", f"step{ckpt.stem}"]
    return cmd


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    cols = [
        "step",
        "checkpoint",
        "return_code",
        "elapsed_sec",
        "episodes",
        "mean_return",
        "mean_success",
        "suite",
        "task",
        "stdout_tail",
        "stderr_tail",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({c: row.get(c, "") for c in cols})


def _plot_curve(path: Path, rows: list[dict[str, Any]], title: str) -> None:
    pts = [
        (int(r["step"]), float(r["mean_success"]))
        for r in rows
        if r.get("return_code") == 0 and r.get("mean_success") not in ("", None)
    ]
    pts.sort(key=lambda x: x[0])

    fig, ax = plt.subplots(figsize=(10, 5), dpi=140)
    if pts:
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", linewidth=1.8)
        for x, y in pts:
            ax.text(x, y + 0.015, f"{y:.2f}", ha="center", va="bottom", fontsize=7)
    else:
        ax.text(0.5, 0.5, "No valid eval points", transform=ax.transAxes, ha="center", va="center")

    ax.set_title(title)
    ax.set_xlabel("checkpoint step")
    ax.set_ylabel("mean success (episodes)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_outputs(
    run_dir: Path,
    tag: str,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    csv_path: Path,
    png_path: Path,
    json_path: Path,
) -> dict[str, Any]:
    _write_csv(csv_path, rows)
    title = f"{tag} | eval_success curve (episodes={args.episodes}, stride={args.ckpt_stride})"
    _plot_curve(png_path, rows, title=title)

    valid = [r for r in rows if r.get("return_code") == 0 and r.get("mean_success") not in ("", None)]
    best = None
    if valid:
        best = max(valid, key=lambda r: float(r["mean_success"]))

    payload = {
        "run_dir": str(run_dir),
        "episodes": int(args.episodes),
        "ckpt_stride": int(args.ckpt_stride),
        "count_ckpt": len(rows),
        "count_valid": len(valid),
        "best": best,
        "csv": str(csv_path),
        "plot": str(png_path),
        "rows": rows,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return payload


def main() -> None:
    args = _parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"run-dir not found: {run_dir}")

    snapshot_dir = run_dir / "snapshot"
    if not snapshot_dir.exists():
        raise FileNotFoundError(f"snapshot dir not found: {snapshot_dir}")

    repo_root = Path(__file__).resolve().parents[1]
    meta = _load_run_meta(run_dir)
    args = _infer_defaults(args, meta)

    ckpts = _find_checkpoints(
        snapshot_dir=snapshot_dir,
        min_step=int(args.min_step),
        max_step=int(args.max_step),
        stride=int(args.ckpt_stride),
    )
    if not ckpts:
        raise RuntimeError(
            "No checkpoints matched filter. "
            f"snapshot_dir={snapshot_dir} min_step={args.min_step} max_step={args.max_step} stride={args.ckpt_stride}"
        )

    out_root = Path(args.output_dir).expanduser().resolve() if args.output_dir else (repo_root / "point_policy_rl" / "visual_reports")
    out_root.mkdir(parents=True, exist_ok=True)
    tag = args.output_tag or run_dir.name
    csv_path = out_root / f"{tag}_eval_curve_e{int(args.episodes)}_s{int(args.ckpt_stride)}.csv"
    png_path = out_root / f"{tag}_eval_curve_e{int(args.episodes)}_s{int(args.ckpt_stride)}.png"
    json_path = out_root / f"{tag}_eval_curve_e{int(args.episodes)}_s{int(args.ckpt_stride)}.json"

    if args.save_video:
        base_video_dir = out_root / f"{tag}_eval_videos"
        base_video_dir.mkdir(parents=True, exist_ok=True)
    else:
        base_video_dir = None

    rows: list[dict[str, Any]] = []
    print(
        f"[sweep] run_dir={run_dir} ckpts={len(ckpts)} episodes={args.episodes} "
        f"stride={args.ckpt_stride} device={args.device}",
        flush=True,
    )
    for idx, ckpt in enumerate(ckpts, start=1):
        step = int(ckpt.stem)
        video_dir_for_ckpt = (base_video_dir / f"step_{step:07d}") if base_video_dir is not None else None
        if video_dir_for_ckpt is not None:
            video_dir_for_ckpt.mkdir(parents=True, exist_ok=True)

        cmd = _build_eval_cmd(repo_root=repo_root, ckpt=ckpt, args=args, video_dir_for_ckpt=video_dir_for_ckpt)
        t0 = time.time()
        return_code = 0
        stdout_text = ""
        stderr_text = ""
        parsed = None
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(repo_root),
                text=True,
                capture_output=True,
                timeout=max(1, int(args.timeout_sec)),
                check=False,
            )
            return_code = int(proc.returncode)
            stdout_text = proc.stdout or ""
            stderr_text = proc.stderr or ""
            parsed = _extract_last_json_block(stdout_text)
        except TimeoutExpired as exc:
            return_code = 124
            stdout_text = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
            stderr_text = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        elapsed = time.time() - t0

        row: dict[str, Any] = {
            "step": step,
            "checkpoint": str(ckpt),
            "return_code": int(return_code),
            "elapsed_sec": round(float(elapsed), 3),
            "episodes": int(args.episodes),
            "mean_return": "",
            "mean_success": "",
            "suite": "",
            "task": "",
            "stdout_tail": "\n".join(stdout_text.splitlines()[-20:]),
            "stderr_tail": "\n".join(stderr_text.splitlines()[-20:]),
        }
        if parsed is not None:
            row["mean_return"] = parsed.get("mean_return", "")
            row["mean_success"] = parsed.get("mean_success", "")
            row["suite"] = parsed.get("suite", "")
            row["task"] = parsed.get("task", "")

        rows.append(row)
        print(
            f"[{idx:02d}/{len(ckpts):02d}] step={step} rc={return_code} "
            f"mean_success={row['mean_success']} elapsed={elapsed:.1f}s",
            flush=True,
        )
        _write_outputs(
            run_dir=run_dir,
            tag=tag,
            args=args,
            rows=rows,
            csv_path=csv_path,
            png_path=png_path,
            json_path=json_path,
        )

    payload = _write_outputs(
        run_dir=run_dir,
        tag=tag,
        args=args,
        rows=rows,
        csv_path=csv_path,
        png_path=png_path,
        json_path=json_path,
    )
    best = payload.get("best")

    print("[done]", flush=True)
    print(f"csv={csv_path}", flush=True)
    print(f"plot={png_path}", flush=True)
    print(f"json={json_path}", flush=True)
    if best is not None:
        print(
            f"best_step={best.get('step')} "
            f"best_success={best.get('mean_success')}",
            flush=True,
        )


if __name__ == "__main__":
    main()
