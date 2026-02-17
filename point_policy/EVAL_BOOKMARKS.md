# Eval Bookmarks

Manual bookmark file for notable eval runs (safe from auto-overwrite hooks).

## Entry: Job 190608 (User-marked success)
- Date: `2026-02-11`
- Suite: `libero_spatial`
- Task: `pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`
- SLURM job id: `190608`
- Video: `point_policy/exp_local/eval_videos/job190608__env0.mp4`
- Eval workspace: `point_policy/exp_local/eval/2026.02.11_eval_point_policy_libero_spatial_189956_objgate0025_ct055_db6_hold400_openhard_long2000/eval_190608_189956_objgate0025_ct055_db6_hold400_openhard_long2000/132353_hidden_dim_256`
- Source checkpoint: `point_policy/exp_local/2026.02.10/point_policy_libero_spatial_cmd_sephead_ctx_bce5_201k_v1/debug_189680_cmd_sephead_ctx_bce5_201k_v1/232901_hidden_dim_256/snapshot/200000.pt`
- Eval summary (from log): `R=519`, `L=2000`, `S=0`

### Key eval overrides
- `agent.separate_gripper_head=true`
- `agent.condition_on_gripper_state=false`
- `agent.gripper_bce_weight=5`
- `agent.gripper_head_use_transformer_context=true`
- `agent.gripper_label_mode=command`
- `suite.gripper_label_mode=command`
- `suite.gripper_close_position_gate=false`
- `suite.gripper_min_hold_steps=400`
- `suite.gripper_cmd_slew_rate=2.0`
- `suite.gripper_reopen_tip_gate=true`
- `suite.gripper_tip_open_threshold=0.08`
- `suite.gripper_close_object_gate=true`
- `suite.gripper_close_object_threshold=0.025`
- `suite.gripper_close_object_point=tip_mid`
- `suite.gripper_close_threshold=0.55`
- `suite.gripper_close_debounce_steps=6`
- `suite.gripper_open_debounce_steps=24`
- `suite.gripper_open_threshold=-0.95`
- `EVAL_ROLLOUT_STEPS=2000`
- `EVAL_MAX_EPISODE_LEN=2000`

### Reproduce command
```bash
cd /sjw_alinlab2/home/sanghyeok/Point-Policy_codex_libero
BC_WEIGHT=/sjw_alinlab2/home/sanghyeok/Point-Policy_codex_libero/point_policy/exp_local/2026.02.10/point_policy_libero_spatial_cmd_sephead_ctx_bce5_201k_v1/debug_189680_cmd_sephead_ctx_bce5_201k_v1/232901_hidden_dim_256/snapshot/200000.pt \
RUN_SUFFIX=189956_objgate0025_ct055_db6_hold400_openhard_long2000 \
VIDEO_EXPORT_TAG=189956_objgate0025_ct055_db6_hold400_openhard_long2000 \
EVAL_ROLLOUT_STEPS=2000 EVAL_MAX_EPISODE_LEN=2000 POINT_DEBUG_MAX_STEPS=2000 \
EXTRA_ARGS='agent.separate_gripper_head=true agent.condition_on_gripper_state=false agent.gripper_bce_weight=5 agent.gripper_head_use_transformer_context=true agent.gripper_label_mode=command suite.gripper_label_mode=command suite.gripper_close_position_gate=false suite.gripper_min_hold_steps=400 suite.gripper_cmd_slew_rate=2.0 suite.gripper_reopen_tip_gate=true suite.gripper_tip_open_threshold=0.08 suite.gripper_close_object_gate=true suite.gripper_close_object_threshold=0.025 suite.gripper_close_object_point=tip_mid suite.gripper_close_threshold=0.55 suite.gripper_close_debounce_steps=6 suite.gripper_open_debounce_steps=24 suite.gripper_open_threshold=-0.95' \
sbatch -p background --job-name=pp-libsp-eval point_policy/experiments/libero_spatial/slurm/eval_debug.sbatch
```

## Update Rule
- After each train/eval run that is meaningful, append one section in this file with:
  - `job id`, `video path`, `source ckpt`, `key overrides`, and short outcome note.
