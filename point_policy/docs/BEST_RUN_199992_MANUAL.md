# LIBERO Spatial Best Run Manual (Train 189680 / Eval 199992)

> Comment tag: 유의미한 성공률

## 1) Best checkpoint
- Base BC checkpoint (train run):
  - `/sjw_alinlab2/home/sanghyeok/Point-Policy_codex_libero/point_policy/exp_local/2026.02.10/point_policy_libero_spatial_cmd_sephead_ctx_bce5_201k_v1/debug_189680_cmd_sephead_ctx_bce5_201k_v1/232901_hidden_dim_256/snapshot/200000.pt`
- Training run dir:
  - `/sjw_alinlab2/home/sanghyeok/Point-Policy_codex_libero/point_policy/exp_local/2026.02.10/point_policy_libero_spatial_cmd_sephead_ctx_bce5_201k_v1/debug_189680_cmd_sephead_ctx_bce5_201k_v1/232901_hidden_dim_256`
- Eval run dir (basefix_v1, per-episode dump):
  - `/sjw_alinlab2/home/sanghyeok/Point-Policy_codex_libero/point_policy/exp_local/eval/2026.02.15_eval_point_policy_libero_spatial_basefix_v1_ep10_step3000_per_ep_dump/eval_199992_ep10_step3000_per_ep_dump/153219_hidden_dim_256`

## 2) Training method (actual run: 189680)
Train script:
- `point_policy/experiments/libero_spatial/slurm/train_debug.sbatch`

Core Hydra overrides used in run 189680:
- `agent.separate_gripper_head=true`
- `agent.condition_on_gripper_state=false`
- `agent.gripper_label_mode=command`
- `suite.gripper_label_mode=command`
- `agent.gripper_head_use_transformer_context=true`
- `agent.gripper_bce_weight=5`
- `suite.num_train_steps=201000`

Equivalent submit example:
```bash
RUN_SUFFIX=cmd_sephead_ctx_bce5_201k_v1 \
EXTRA_ARGS='agent.separate_gripper_head=true agent.condition_on_gripper_state=false agent.gripper_label_mode=command suite.gripper_label_mode=command agent.gripper_head_use_transformer_context=true agent.gripper_bce_weight=5 suite.num_train_steps=201000' \
sbatch -p debug --job-name=pp-libsp point_policy/experiments/libero_spatial/slurm/train_debug.sbatch
```

## 3) Training hyperparameters (from train run config)
- Model:
  - `policy_head=deterministic`
  - `hidden_dim=256`
  - Transformer params: ~3.36M
- Optimizer / learning:
  - `lr=1e-4`
  - `batch_size=64`
- Temporal:
  - `history_len=10`
  - `num_queries=20`
  - `temporal_agg=true`
- Data / points:
  - `num_robot_points=9`
  - `num_object_points=10`
  - `object_point_mode=geom_fps`
  - `point_dim=3`
  - `gripper label mode=command`
- Steps:
  - `num_train_steps=201000`
- Image size in training:
  - `suite.img_size=[128,128]`

## 4) Eval method (actual run: 199992)
Eval script:
- `point_policy/experiments/libero_spatial/slurm/eval_debug_basefix_v1.sbatch`

Submitted eval settings:
- `BC_WEIGHT=<200000.pt above>`
- `NUM_EVAL_EPISODES=10`
- `EVAL_ROLLOUT_STEPS=3000`
- `EVAL_MAX_EPISODE_LEN=3000`
- `POINT_DEBUG_MAX_STEPS=3000`
- `video_save_per_episode=true` (basefix config default)
- `point_debug_save_per_episode=true` (basefix config default)

Equivalent submit command:
```bash
sbatch -p debug --job-name=pp-libsp-eval-bf1 \
  --export=ALL,BC_WEIGHT=/sjw_alinlab2/home/sanghyeok/Point-Policy_codex_libero/point_policy/exp_local/2026.02.10/point_policy_libero_spatial_cmd_sephead_ctx_bce5_201k_v1/debug_189680_cmd_sephead_ctx_bce5_201k_v1/232901_hidden_dim_256/snapshot/200000.pt,NUM_EVAL_EPISODES=10,EVAL_ROLLOUT_STEPS=3000,EVAL_MAX_EPISODE_LEN=3000,POINT_DEBUG_MAX_STEPS=3000,RUN_SUFFIX=ep10_step3000_per_ep_dump \
  point_policy/experiments/libero_spatial/slurm/eval_debug_basefix_v1.sbatch
```

## 5) Eval-time hyperparameters (basefix_v1 config used)
- Visual:
  - `suite.img_size=[256,256]`
  - `video_render_size=960`
  - `point_overlay=true`
- Point2Action / control:
  - `pose_solve_mode=rigid`
  - `max_delta_pos=0.05`
  - `max_delta_rot=0.25`
  - `normalize_delta_action_to_osc=true`
  - `controller_output_max_pos=0.05`
  - `controller_output_max_rot=0.5`
- Gripper postprocess:
  - `gripper_label_mode=command`
  - `gripper_close_threshold=0.2`
  - `gripper_open_threshold=-0.2`
  - `gripper_cmd_slew_rate=2.0`
  - `gripper_close_position_gate=false`
  - `gripper_close_object_gate=false`
  - `real_deploy_mode=true`
  - `gripper_command_fallback_to_tip_distance=false`
  - `gripper_reopen_tip_gate=false`
  - `gripper_min_hold_steps=12`

## 6) Action generation method (point2action)
Code reference:
- `point_policy/suite/libero_spatial_basefix_v1.py`

Runtime flow:
1. 정책 출력에서 `future_tracks_<pixel_key>`를 가져옴 (주로 robot 9점 사용).
2. 현재 로봇 포인트(`current_robot_points`)와 목표 로봇 포인트(`target_robot_points`)를 구성.
3. `rigid_transform_3D`로 현재->목표 rigid 변환을 추정 (tip 점은 rigid solve에서 제외 가능).
4. EEF 기준 `delta_pos`, `delta_rot(rotvec)`를 계산.
5. `max_delta_pos/max_delta_rot`로 clip.
6. `normalize_delta_action_to_osc=true`면 OSC 입력 스케일로 정규화 후 `[-1,1]` clip.
7. Gripper는 command 스칼라 출력(`gripper` 또는 `future_gripper_states`)에 대해
   hysteresis(close/open threshold), debounce/hold, optional gate, slew를 적용.
8. 최종 action:
   - `[dx, dy, dz, drot_x, drot_y, drot_z, gripper_cmd]`.

## 7) Per-episode output locations
- Exported videos:
  - `/sjw_alinlab2/home/sanghyeok/Point-Policy_codex_libero/point_policy/exp_local/eval_videos_basefix_v1`
  - Filename pattern: `job<jobid>__env<e>__ep<k>__chunk<c>__<success|fail>.mp4`
- Eval run dir artifacts (per episode):
  - `point_debug_env<env>_ep<ep>.npz`
  - `point_debug_env<env>_ep<ep>.txt`
  - `gripper_debug_env<env>_ep<ep>.txt`
  - `temporal_debug_env<env>_ep<ep>.txt` (if enabled)
  - `contact_debug_env<env>_ep<ep>.txt` (if enabled)
  - `point2action_stats_env<env>_ep<ep>.txt`

## 8) Notes
- This recipe is policy rollout eval (not replay / not offline GT-points mode).
- If `/tmp/robosuite.log` permission issue appears on worker nodes, disable robosuite file logging via `macros_private.py` (`FILE_LOGGING_LEVEL=None`).
