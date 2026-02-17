# Point2Action Reference

This repo now includes a standalone reference implementation of the rule-based
`point -> action` conversion used in `libero_spatial`.

## Files

- Core reference logic:
  - `point_policy/suite/point2action_reference.py`
- Offline trace utility from `point_debug_offline_env*.npz`:
  - `point_policy/scripts/trace_point2action_from_npz.py`

## Why this exists

The policy predicts points, but execution quality also depends on the hard-coded
conversion stage (rigid solve, clipping, gripper thresholds/gates/slew).

This reference module makes that stage testable outside the simulator.

## Quick usage

```bash
cd /sjw_alinlab2/home/sanghyeok/Point-Policy_codex_libero
source /sjw_alinlab2/home/sanghyeok/miniconda3/etc/profile.d/conda.sh
conda activate libero_codex

python point_policy/scripts/trace_point2action_from_npz.py \
  --npz point_policy/exp_local/eval/.../point_debug_offline_env0.npz \
  --pose-solve-mode rigid \
  --gripper-label-mode command \
  --gripper-close-threshold 0.2 \
  --gripper-open-threshold -0.2 \
  --gripper-slew-rate 0.15 \
  --gripper-min-hold-steps 12
```

Output:

- `<npz_stem>_point2action_trace.csv`

Main columns:

- `raw_gripper`: model gripper output used as input to post-processing
- `desired_cmd`: thresholded target command before final smoothing/gates
- `cmd_final`: final command after debounce/hold/gates/slew
- `dx,dy,dz,drx,dry,drz`: resulting delta action components

## Notes

- Offline trace uses `kp0` as EEF position anchor and identity EEF rotation.
- So translation/rotation magnitudes are approximate offline diagnostics.
- Gripper command state transitions are the primary target for rule verification.

