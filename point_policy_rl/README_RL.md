# Point-Policy Residual RL (`_rl` isolated path)

This directory adds residual TD3 training/evaluation without modifying existing
`point_policy/` code paths.

## Design

- Base policy: frozen Point-Policy BC checkpoint (`--bc-weight`)
- Residual learner: TD3 actor + twin critics
- Control: `env_action = clip(base_action + residual_action)`
- Input (default): point tracks + gripper feature + base action
- Off-policy update: replay-buffer learning with optional offline+online batch mixing
- Output root: `point_policy/exp_local_rl/...`

## Train

```bash
cd /sjw_alinlab2/home/sanghyeok/Point-Policy_codex_libero
python point_policy_rl/train_residual_td3_rl.py \
  --bc-weight /path/to/point_policy_snapshot.pt \
  --suite libero_spatial \
  --steps 100000 \
  --run-name smoke
```

ResFiT core backend (QAgent + TorchRL replay + MultiStepTransform):

```bash
python point_policy_rl/train_resfit_residual_td3_rl.py \
  --bc-weight /path/to/point_policy_snapshot.pt \
  --suite libero_spatial \
  --task-name pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate \
  --offline-fraction 0.3 \
  --offline-base-action-mode bc_track_delta \
  --run-name resfit_bridge_smoke
```

W&B logging example:

```bash
python point_policy_rl/train_resfit_residual_td3_rl.py \
  --bc-weight /path/to/point_policy_snapshot.pt \
  --suite libero_spatial \
  --wandb-enable \
  --wandb-project point-policy-residual-rl \
  --wandb-mode online \
  --run-name resfit_wandb_smoke
```

Optional offline demo mix:

```bash
python point_policy_rl/train_residual_td3_rl.py \
  --bc-weight /path/to/point_policy_snapshot.pt \
  --suite libero_spatial \
  --task-name pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate \
  --offline-fraction 0.3 \
  --offline-demo-root expert_demos/libero_spatial \
  --offline-max-demos 20
```

Residual-offpolicy-style schedule:

```bash
python point_policy_rl/train_residual_td3_rl.py \
  --bc-weight /path/to/point_policy_snapshot.pt \
  --suite libero_spatial \
  --offline-fraction 0.3 \
  --num-updates-per-step 4 \
  --update-every-n-steps 1
```

## Eval

```bash
cd /sjw_alinlab2/home/sanghyeok/Point-Policy_codex_libero
python point_policy_rl/eval_residual_td3_rl.py \
  --ckpt point_policy/exp_local_rl/.../snapshot/latest.pt \
  --episodes 10
```

## Notes

- Existing BC scripts remain untouched (`point_policy/train.py`, `point_policy/eval_point_track.py`).
- Offline transitions are built from Point-Policy demo pkl files (`observations`) with
  residual target fixed to zero (`action=0`) and terminal reward shaping.
- `--offline-base-action-mode`:
  - `demo_delta`: demo frame difference as base action (GT-as-base style)
  - `bc_track_delta`: frozen BC prediction delta as base action approximation
- `train_resfit_residual_td3_rl.py` keeps Point-Policy env/base adapter, but replaces
  core RL learner/replay/update loop with ResFiT components.
