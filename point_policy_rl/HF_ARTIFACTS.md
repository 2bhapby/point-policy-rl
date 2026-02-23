# HF Artifacts Guide

This folder supports reusable transfer of RL artifacts to Hugging Face.

## Files

- Manifest: `point_policy_rl/hf_artifacts_manifest.yaml`
- Sync script: `point_policy_rl/hf_sync_artifacts.py`

## What is included in manifest

- BC checkpoints (scene3/6/8, basefix_v1)
- Offline RLDS-like `.pkl` (+summary txt)
- Representative residual checkpoints/logs (image-only scene3/6)

## 1) Validate local files

```bash
python point_policy_rl/hf_sync_artifacts.py \
  --mode validate \
  --manifest point_policy_rl/hf_artifacts_manifest.yaml \
  --repo-root .
```

## 2) Upload to HF

Requires login (`huggingface-cli login`) or `HF_TOKEN`.

```bash
python point_policy_rl/hf_sync_artifacts.py \
  --mode upload \
  --manifest point_policy_rl/hf_artifacts_manifest.yaml \
  --repo-root . \
  --create-repos \
  --private
```

Optional filters:

```bash
python point_policy_rl/hf_sync_artifacts.py \
  --mode upload \
  --manifest point_policy_rl/hf_artifacts_manifest.yaml \
  --name-regex "scene3|scene6" \
  --repo-root .
```

## 3) Download on another server

```bash
python point_policy_rl/hf_sync_artifacts.py \
  --mode download \
  --manifest point_policy_rl/hf_artifacts_manifest.yaml \
  --download-root /path/to/workspace
```

## Notes

- Manifest uses `${HF_NAMESPACE}` variable. Default is set in the manifest.
- Override namespace at runtime:

```bash
python point_policy_rl/hf_sync_artifacts.py \
  --mode upload \
  --manifest point_policy_rl/hf_artifacts_manifest.yaml \
  --hf-namespace your_hf_id
```

