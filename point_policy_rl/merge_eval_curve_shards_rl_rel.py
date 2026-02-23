#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge per-checkpoint eval shard CSVs into one curve (RL REL)."
    )
    p.add_argument("--input-dir", type=str, required=True)
    p.add_argument(
        "--pattern",
        type=str,
        default="scene3_img_fix0220_bg_e50_step*_eval_curve_e50_s10000.csv",
    )
    p.add_argument("--out-tag", type=str, default="scene3_img_fix0220_bg_e50_merged")
    p.add_argument("--expected-min-step", type=int, default=10000)
    p.add_argument("--expected-max-step", type=int, default=400000)
    p.add_argument("--expected-stride", type=int, default=10000)
    return p.parse_args()


def _to_int(v: Any, default: int = -1) -> int:
    try:
        return int(float(v))
    except Exception:
        return int(default)


def _to_float(v: Any, default: float = float("nan")) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            step = _to_int(row.get("step"), default=-1)
            if step < 0:
                continue
            out.append(row)
    return out


def main() -> None:
    args = _parse_args()
    in_dir = Path(args.input_dir).expanduser().resolve()
    if not in_dir.exists():
        raise FileNotFoundError(f"input dir not found: {in_dir}")

    shard_paths = sorted(in_dir.glob(args.pattern))
    rows: list[dict[str, Any]] = []
    for p in shard_paths:
        rows.extend(_read_rows(p))

    # Deduplicate by step; keep the latest discovered row for that step.
    by_step: dict[int, dict[str, Any]] = {}
    for row in rows:
        step = _to_int(row.get("step"), default=-1)
        if step >= 0:
            by_step[step] = row
    steps_sorted = sorted(by_step.keys())
    merged_rows = [by_step[s] for s in steps_sorted]

    expected = list(
        range(
            int(args.expected_min_step),
            int(args.expected_max_step) + 1,
            int(args.expected_stride),
        )
    )
    missing = [s for s in expected if s not in by_step]

    out_csv = in_dir / f"{args.out_tag}.csv"
    out_json = in_dir / f"{args.out_tag}.json"
    out_png = in_dir / f"{args.out_tag}.png"

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

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for row in merged_rows:
            w.writerow({c: row.get(c, "") for c in cols})

    xs = [int(r["step"]) for r in merged_rows if _to_int(r.get("return_code"), 1) == 0]
    ys = [
        _to_float(r.get("mean_success"))
        for r in merged_rows
        if _to_int(r.get("return_code"), 1) == 0
    ]
    rs = [
        _to_float(r.get("mean_return"))
        for r in merged_rows
        if _to_int(r.get("return_code"), 1) == 0
    ]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, dpi=140)
    ax_succ, ax_ret = axes
    ax_succ.plot(xs, ys, marker="o", linewidth=1.6, label="mean_success")
    ax_succ.set_ylabel("success")
    ax_succ.set_ylim(-0.02, 1.02)
    ax_succ.grid(alpha=0.3)
    ax_succ.legend()
    ax_succ.set_title(args.out_tag)

    ax_ret.plot(xs, rs, marker="o", linewidth=1.6, color="tab:orange", label="mean_return")
    ax_ret.set_ylabel("return")
    ax_ret.set_xlabel("checkpoint step")
    ax_ret.grid(alpha=0.3)
    ax_ret.legend()
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    payload = {
        "input_dir": str(in_dir),
        "pattern": args.pattern,
        "shard_count": len(shard_paths),
        "merged_count": len(merged_rows),
        "expected_count": len(expected),
        "missing_steps": missing,
        "csv": str(out_csv),
        "png": str(out_png),
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

