#!/usr/bin/env python3
from __future__ import annotations

import pickle
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

DATASET_REPO = "insagur/point-policy-rl-datasets"
MODEL_REPO = "insagur/point-policy-rl-models"

IMG50_BASENAME = "libero10_3_turn_on_the_stove_and_put_the_moka_pot_on_it_img50.pkl"
BC_RUN_REL = (
    "point_policy/exp_local/2026.02.17/"
    "point_policy_libero_object_basefix_v1_lib10_scene3_obj15_sephead_nocond_bf1_0217/"
    "debug_201911_lib10_scene3_obj15_sephead_nocond_bf1_0217/004042_hidden_dim_256"
)


def _select_img50(files: list[str]) -> str:
    exact = f"offline_rlds/libero10/scene3/{IMG50_BASENAME}"
    if exact in files:
        return exact
    cands = [f for f in files if f.endswith(".pkl") and "libero10_3" in f and "img50" in f.lower()]
    if not cands:
        raise FileNotFoundError("No scene3 img50 pkl found in dataset repo")
    cands.sort()
    return cands[0]


def _select_cfg(files: list[str]) -> str | None:
    cands = [f for f in files if "bc/libero10/scene3" in f and f.endswith("config.yaml")]
    if not cands:
        return None

    def score(path: str) -> tuple[int, int, str]:
        return (
            0 if "/.hydra/config.yaml" in path else 1,
            0 if "basefix_v1" in path else 1,
            path,
        )

    cands.sort(key=score)
    return cands[0]


def _verify_img50(path: Path) -> None:
    with path.open("rb") as f:
        payload = pickle.load(f)

    if not isinstance(payload, dict) or "episodes" not in payload:
        raise ValueError(f"Unexpected img50 payload format: {path}")

    episodes = payload.get("episodes") or []
    if not episodes:
        raise ValueError(f"No episodes in img50 payload: {path}")

    obs = episodes[0]["steps"][0].get("observation", {})
    if not isinstance(obs, dict) or "pixels1" not in obs:
        raise ValueError(f"pixels1 missing in img50 payload: {path}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    api = HfApi()

    print("[hf] listing dataset repo files...")
    dataset_files = api.list_repo_files(repo_id=DATASET_REPO, repo_type="dataset")
    img50_remote = _select_img50(dataset_files)
    print(f"[hf] selected img50: {img50_remote}")

    print("[hf] listing model repo files...")
    model_files = api.list_repo_files(repo_id=MODEL_REPO, repo_type="model")
    model_scene3_files = sorted([f for f in model_files if "bc/libero10/scene3" in f])
    print("[hf] model scene3 entries:")
    for f in model_scene3_files:
        print(f)
    cfg_remote = _select_cfg(model_files)
    if cfg_remote:
        print(f"[hf] selected cfg: {cfg_remote}")
    else:
        print("[hf] WARNING: scene3 BC config.yaml not found in model repo")

    print("[hf] downloading img50...")
    local_img = Path(
        hf_hub_download(
            repo_id=DATASET_REPO,
            repo_type="dataset",
            filename=img50_remote,
            local_dir=str(repo_root),
            local_dir_use_symlinks=False,
        )
    )

    local_cfg = None
    if cfg_remote:
        print("[hf] downloading bc config...")
        local_cfg = Path(
            hf_hub_download(
                repo_id=MODEL_REPO,
                repo_type="model",
                filename=cfg_remote,
                local_dir=str(repo_root),
                local_dir_use_symlinks=False,
            )
        )

    target_img = repo_root / "point_policy" / "exp_local" / "offline_rlds" / IMG50_BASENAME
    target_img.parent.mkdir(parents=True, exist_ok=True)
    if local_img.resolve() != target_img.resolve():
        shutil.copy2(local_img, target_img)

    target_cfg = repo_root / BC_RUN_REL / ".hydra" / "config.yaml"
    if local_cfg:
        target_cfg.parent.mkdir(parents=True, exist_ok=True)
        if local_cfg.resolve() != target_cfg.resolve():
            shutil.copy2(local_cfg, target_cfg)

    _verify_img50(target_img)

    print(f"[ok] img50: {target_img}")
    if local_cfg:
        print(f"[ok] bc_cfg: {target_cfg}")


if __name__ == "__main__":
    main()
