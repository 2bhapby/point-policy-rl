#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import struct
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


def _build_crc32c_table() -> List[int]:
    poly = 0x82F63B78  # reversed Castagnoli polynomial
    table: List[int] = []
    for i in range(256):
        crc = i
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ poly
            else:
                crc >>= 1
        table.append(crc & 0xFFFFFFFF)
    return table


_CRC32C_TABLE = _build_crc32c_table()


def _crc32c(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for b in data:
        crc = _CRC32C_TABLE[(crc ^ b) & 0xFF] ^ (crc >> 8)
    return (~crc) & 0xFFFFFFFF


def _mask_crc32c(crc: int) -> int:
    crc &= 0xFFFFFFFF
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def _write_tfrecord_record(fp, payload: bytes) -> int:
    length = len(payload)
    length_bytes = struct.pack("<Q", length)
    length_crc = struct.pack("<I", _mask_crc32c(_crc32c(length_bytes)))
    data_crc = struct.pack("<I", _mask_crc32c(_crc32c(payload)))
    fp.write(length_bytes)
    fp.write(length_crc)
    fp.write(payload)
    fp.write(data_crc)
    return 8 + 4 + length + 4


def _to_py_scalar_or_list(v: Any) -> Any:
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, dict):
        return {str(k): _to_py_scalar_or_list(vv) for k, vv in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_py_scalar_or_list(x) for x in v]
    return v


def _iter_step_records(dataset: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    episodes = dataset.get("episodes", [])
    for ep_idx, episode in enumerate(episodes):
        episode_id = int(episode.get("episode_id", ep_idx))
        metadata = episode.get("metadata", {})
        steps = episode.get("steps", [])
        for step_idx, step in enumerate(steps):
            yield {
                "record_type": "step",
                "episode_id": episode_id,
                "step_idx": int(step_idx),
                "is_first": bool(step.get("is_first", step_idx == 0)),
                "is_last": bool(step.get("is_last", False)),
                "is_terminal": bool(step.get("is_terminal", False)),
                "is_truncated": bool(step.get("is_truncated", False)),
                "reward": float(step.get("reward", 0.0)),
                "discount": float(step.get("discount", 1.0)),
                "observation": step.get("observation", {}),
                "action": step.get("action", None),
                "policy_debug": step.get("policy_debug", {}),
                "info": step.get("info", {}),
                "episode_metadata": metadata,
            }


def convert_rlds_like_pickle_to_tfrecord(
    input_path: Path,
    output_path: Path,
    index_path: Path,
) -> Dict[str, Any]:
    with input_path.open("rb") as f:
        dataset = pickle.load(f)
    if not isinstance(dataset, dict) or "episodes" not in dataset:
        raise ValueError(f"Unsupported RLDS-like payload: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    num_episodes = int(len(dataset.get("episodes", [])))
    num_steps = 0
    num_records = 0
    bytes_written = 0
    episode_lengths: List[int] = []

    header = {
        "record_type": "header",
        "source_format": str(dataset.get("format", "unknown")),
        "source_path": str(input_path),
        "num_episodes": num_episodes,
    }

    with output_path.open("wb") as fp:
        payload = pickle.dumps(header, protocol=pickle.HIGHEST_PROTOCOL)
        bytes_written += _write_tfrecord_record(fp, payload)
        num_records += 1

        for ep in dataset.get("episodes", []):
            episode_lengths.append(int(len(ep.get("steps", []))))

        for record in _iter_step_records(dataset):
            payload = pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL)
            bytes_written += _write_tfrecord_record(fp, payload)
            num_records += 1
            num_steps += 1

    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "index_path": str(index_path),
        "payload_encoding": "pickle",
        "container": "tfrecord",
        "notes": (
            "Each TFRecord payload is a pickled Python dict following RLDS-like step schema. "
            "First record is a header."
        ),
        "num_episodes": num_episodes,
        "num_steps": num_steps,
        "num_records": num_records,
        "bytes_written": int(bytes_written),
        "mean_episode_len": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "min_episode_len": int(np.min(episode_lengths)) if episode_lengths else 0,
        "max_episode_len": int(np.max(episode_lengths)) if episode_lengths else 0,
    }

    with index_path.open("w", encoding="utf-8") as f:
        json.dump(_to_py_scalar_or_list(summary), f, indent=2, ensure_ascii=False)

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert RLDS-like pickle dataset to TFRecord container. "
            "Record payloads are pickled step dicts."
        )
    )
    parser.add_argument("--input", required=True, type=Path, help="Input RLDS-like .pkl path")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .tfrecord path (default: <input_stem>.tfrecord)",
    )
    parser.add_argument(
        "--index-output",
        type=Path,
        default=None,
        help="Output JSON summary path (default: <output>.index.json)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"input not found: {input_path}")

    if args.output is None:
        output_path = input_path.with_suffix(".tfrecord")
    else:
        output_path = args.output.expanduser().resolve()

    if args.index_output is None:
        index_path = Path(str(output_path) + ".index.json")
    else:
        index_path = args.index_output.expanduser().resolve()

    summary = convert_rlds_like_pickle_to_tfrecord(
        input_path=input_path,
        output_path=output_path,
        index_path=index_path,
    )
    print(json.dumps(_to_py_scalar_or_list(summary), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
