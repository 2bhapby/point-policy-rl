#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml
from huggingface_hub import HfApi, HfFolder, hf_hub_download


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload / download RL artifacts with a YAML manifest.")
    p.add_argument(
        "--manifest",
        type=str,
        default="point_policy_rl/hf_artifacts_manifest.yaml",
        help="Manifest YAML path.",
    )
    p.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["validate", "upload", "download"],
        help="Operation mode.",
    )
    p.add_argument(
        "--repo-root",
        type=str,
        default=".",
        help="Repository root used to resolve local_path.",
    )
    p.add_argument(
        "--download-root",
        type=str,
        default=".",
        help="Where files are written in download mode (relative layout preserved).",
    )
    p.add_argument(
        "--name-regex",
        type=str,
        default=".*",
        help="Select artifacts by name regex.",
    )
    p.add_argument(
        "--hf-namespace",
        type=str,
        default="",
        help="Override ${HF_NAMESPACE} variable in manifest.",
    )
    p.add_argument(
        "--create-repos",
        action="store_true",
        help="Create target repos before upload (exist_ok=True).",
    )
    p.add_argument(
        "--private",
        action="store_true",
        help="Create repos as private when --create-repos is used.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Fail if an artifact local file is missing.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print actions without upload/download.",
    )
    p.add_argument(
        "--commit-message",
        type=str,
        default="Upload Point-Policy RL artifacts",
        help="Commit message for upload.",
    )
    return p.parse_args()


def _load_manifest(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Manifest is not a dict: {path}")
    arts = data.get("artifacts", [])
    if not isinstance(arts, list):
        raise ValueError("manifest.artifacts must be a list")
    return data


def _resolve_hf_namespace(args: argparse.Namespace, manifest: Dict[str, Any], api: HfApi, token: str | None) -> str:
    if args.hf_namespace:
        return args.hf_namespace
    vars_dict = manifest.get("variables", {})
    if isinstance(vars_dict, dict) and vars_dict.get("HF_NAMESPACE"):
        return str(vars_dict["HF_NAMESPACE"])
    if token:
        who = api.whoami(token=token)
        name = who.get("name")
        if name:
            return str(name)
    return ""


def _sub_vars(text: str, vars_dict: Dict[str, str]) -> str:
    out = str(text)
    for k, v in vars_dict.items():
        out = out.replace(f"${{{k}}}", str(v))
    return out


def _select_artifacts(manifest: Dict[str, Any], name_regex: str, sub_vars_dict: Dict[str, str]) -> List[Dict[str, Any]]:
    pat = re.compile(name_regex)
    selected: List[Dict[str, Any]] = []
    for raw in manifest.get("artifacts", []):
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name", "")).strip()
        if not name or not pat.search(name):
            continue
        art = dict(raw)
        art["name"] = name
        art["repo_id"] = _sub_vars(str(raw.get("repo_id", "")).strip(), sub_vars_dict)
        art["repo_type"] = str(raw.get("repo_type", "model")).strip() or "model"
        art["local_path"] = str(raw.get("local_path", "")).strip()
        art["path_in_repo"] = str(raw.get("path_in_repo", "")).strip()
        selected.append(art)
    return selected


def _ensure_token() -> str:
    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or HfFolder.get_token()
    )
    if not token:
        raise RuntimeError(
            "No Hugging Face token found. "
            "Set HF_TOKEN/HUGGINGFACE_HUB_TOKEN or run `huggingface-cli login`."
        )
    return token


def _validate_local(artifacts: List[Dict[str, Any]], repo_root: Path, strict: bool) -> int:
    missing = 0
    print(f"[validate] artifacts={len(artifacts)}")
    for art in artifacts:
        local = (repo_root / art["local_path"]).resolve()
        if not local.exists():
            missing += 1
            print(f"[missing] {art['name']} -> {local}")
            continue
        size_mb = local.stat().st_size / (1024 * 1024)
        print(f"[ok] {art['name']} size={size_mb:.2f}MB local={local}")
    if missing and strict:
        print(f"[validate] missing={missing} (strict mode)")
        return 1
    print(f"[validate] missing={missing}")
    return 0


def _create_repos_if_needed(api: HfApi, artifacts: List[Dict[str, Any]], token: str, private: bool, dry_run: bool) -> None:
    repos = sorted({(a["repo_id"], a["repo_type"]) for a in artifacts})
    for repo_id, repo_type in repos:
        if dry_run:
            print(f"[dry-run][create_repo] repo_id={repo_id} repo_type={repo_type} private={private}")
            continue
        api.create_repo(
            repo_id=repo_id,
            token=token,
            repo_type=repo_type,
            private=private,
            exist_ok=True,
        )
        print(f"[create_repo] ok repo_id={repo_id} repo_type={repo_type}")


def _upload(api: HfApi, artifacts: List[Dict[str, Any]], repo_root: Path, token: str, dry_run: bool, strict: bool, commit_message: str) -> int:
    failed = 0
    for art in artifacts:
        local = (repo_root / art["local_path"]).resolve()
        if not local.exists():
            print(f"[skip-missing] {art['name']} local={local}")
            failed += 1
            continue
        print(
            f"[upload] {art['name']} "
            f"{local} -> {art['repo_id']}:{art['path_in_repo']} ({art['repo_type']})"
        )
        if dry_run:
            continue
        try:
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=art["path_in_repo"],
                repo_id=art["repo_id"],
                repo_type=art["repo_type"],
                token=token,
                commit_message=commit_message,
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[upload-failed] {art['name']} reason={type(exc).__name__}: {exc}")
    if failed and strict:
        return 1
    print(f"[upload] done failed={failed}")
    return 0


def _download(artifacts: List[Dict[str, Any]], download_root: Path, token: str, dry_run: bool, strict: bool) -> int:
    failed = 0
    for art in artifacts:
        target = (download_root / art["local_path"]).resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        p = Path(art["path_in_repo"])
        filename = p.name
        subfolder = str(p.parent) if str(p.parent) != "." else None
        print(
            f"[download] {art['name']} "
            f"{art['repo_id']}:{art['path_in_repo']} -> {target}"
        )
        if dry_run:
            continue
        try:
            tmp = hf_hub_download(
                repo_id=art["repo_id"],
                repo_type=art["repo_type"],
                filename=filename,
                subfolder=subfolder,
                token=token,
                local_dir=str(target.parent),
                local_dir_use_symlinks=False,
            )
            Path(tmp).replace(target)
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[download-failed] {art['name']} reason={type(exc).__name__}: {exc}")
    if failed and strict:
        return 1
    print(f"[download] done failed={failed}")
    return 0


def main() -> int:
    args = _parse_args()
    repo_root = Path(args.repo_root).resolve()
    manifest_path = Path(args.manifest).resolve()
    manifest = _load_manifest(manifest_path)
    api = HfApi()

    token: str | None = None
    if args.mode in {"upload", "download"} or args.create_repos:
        token = _ensure_token()

    hf_namespace = _resolve_hf_namespace(args, manifest, api, token)
    sub_vars_dict = {"HF_NAMESPACE": hf_namespace}
    artifacts = _select_artifacts(manifest, args.name_regex, sub_vars_dict)
    if not artifacts:
        print("[info] no artifacts matched")
        return 0

    print(f"[info] mode={args.mode} artifacts={len(artifacts)} namespace={hf_namespace or '(unset)'}")
    print(f"[info] manifest={manifest_path}")

    if args.mode == "validate":
        return _validate_local(artifacts, repo_root, args.strict)

    assert token is not None
    if args.create_repos:
        _create_repos_if_needed(
            api=api,
            artifacts=artifacts,
            token=token,
            private=bool(args.private),
            dry_run=bool(args.dry_run),
        )

    if args.mode == "upload":
        return _upload(
            api=api,
            artifacts=artifacts,
            repo_root=repo_root,
            token=token,
            dry_run=bool(args.dry_run),
            strict=bool(args.strict),
            commit_message=str(args.commit_message),
        )

    if args.mode == "download":
        return _download(
            artifacts=artifacts,
            download_root=Path(args.download_root).resolve(),
            token=token,
            dry_run=bool(args.dry_run),
            strict=bool(args.strict),
        )

    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())

