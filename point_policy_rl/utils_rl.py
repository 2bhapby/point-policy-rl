from __future__ import annotations

import datetime as dt
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

_PATH_SEGMENT_PATTERN = re.compile(r"^([A-Za-z0-9_]+)((?:\[-?\d+\])*)$")
_PATH_INDEX_PATTERN = re.compile(r"\[(-?\d+)\]")


def add_point_policy_path(repo_root: Path) -> Path:
    """Ensure `point_policy/` is importable as a top-level module namespace."""
    point_policy_dir = (repo_root / "point_policy").resolve()
    if str(point_policy_dir) not in sys.path:
        sys.path.insert(0, str(point_policy_dir))
    return point_policy_dir


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def soft_update(target, source, tau: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


def _deep_get(data: dict[str, Any], path: str) -> Any:
    cur: Any = data
    for raw_segment in path.split("."):
        match = _PATH_SEGMENT_PATTERN.fullmatch(raw_segment)
        if match is None:
            raise KeyError(path)
        key = match.group(1)
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(path)
        cur = cur[key]
        index_expr = match.group(2)
        if index_expr:
            for idx_match in _PATH_INDEX_PATTERN.finditer(index_expr):
                idx = int(idx_match.group(1))
                if not isinstance(cur, (list, tuple)):
                    raise KeyError(path)
                if idx < 0 or idx >= len(cur):
                    raise KeyError(path)
                cur = cur[idx]
    return cur


def _resolve_value(value: Any, root: dict[str, Any], max_depth: int = 8) -> Any:
    if isinstance(value, dict):
        return {k: _resolve_value(v, root, max_depth=max_depth) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_value(v, root, max_depth=max_depth) for v in value]

    if not isinstance(value, str):
        return value

    if value == "???":
        return None

    pattern = re.compile(r"\$\{([^}]+)\}")
    out = value
    depth = 0
    while depth < max_depth and pattern.search(out):
        depth += 1

        def repl(match: re.Match[str]) -> str:
            expr = match.group(1)
            resolved = _deep_get(root, expr)
            if resolved is None:
                return ""
            return str(resolved)

        out = pattern.sub(repl, out)

    if out.lower() in {"true", "false"}:
        return out.lower() == "true"

    # Preserve literal strings that contain commas/slashes/etc.
    if out != "" and re.fullmatch(r"[-+]?\d+", out):
        try:
            return int(out)
        except Exception:
            return out
    if out != "" and re.fullmatch(r"[-+]?\d*\.\d+(e[-+]?\d+)?", out, flags=re.IGNORECASE):
        try:
            return float(out)
        except Exception:
            return out

    return out


def resolve_hydra_like_config(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve simple ${a.b.c} references in a hydra-exported YAML dict."""
    resolved = config
    for _ in range(5):
        resolved = _resolve_value(resolved, resolved)
    return resolved


def load_hydra_config_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Invalid config at {path}: expected dict root")
    return resolve_hydra_like_config(data)


def build_run_dir(output_root: Path, suite_name: str, run_name: str) -> Path:
    now = dt.datetime.now()
    day = now.strftime("%Y.%m.%d")
    ts = now.strftime("%H%M%S")
    suffix = run_name.strip() if run_name.strip() else "run"
    run_dir = output_root / day / f"residual_td3_{suite_name}_rl" / f"{ts}_{suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def dump_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def coerce_device(requested_device: str) -> str:
    try:
        import torch

        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            return "cpu"
    except Exception:
        return "cpu"
    return requested_device
