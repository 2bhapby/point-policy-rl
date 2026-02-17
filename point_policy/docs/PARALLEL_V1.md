# Parallel V1 Line (No Overwrite)

This line runs in parallel to existing code paths and does not require modifying
any existing files.

## New entry points

- Train: `point_policy/train_parallel_v1.py`
- Eval: `point_policy/eval_point_track_parallel_v1.py`

## New Hydra configs

- Train config: `point_policy/cfgs/config_parallel_v1.yaml`
- Eval config: `point_policy/cfgs/config_eval_parallel_v1.yaml`
- Agent config: `point_policy/cfgs/agent/point_policy_parallel_v1.yaml`
- Suite configs:
  - `point_policy/cfgs/suite/libero_spatial_parallel_v1.yaml`
  - `point_policy/cfgs/suite/libero_object_parallel_v1.yaml`
- Dataloader configs:
  - `point_policy/cfgs/dataloader/libero_spatial_parallel_v1.yaml`
  - `point_policy/cfgs/dataloader/libero_object_parallel_v1.yaml`

## Temporal alignment controls (agent)

- `query_stride`: defaults to `${dataloader.bc_dataset.subsample}`
- `query_offset`: defaults to `0`
- `temporal_weight_mode`: `recent|legacy|uniform` (default `recent`)
- `temporal_weight_k`: exponential weight temperature (default `0.01`)
- `emit_temporal_debug`: if true, eval writes `temporal_debug_env*.txt`

## Temporal write rule

For each query index `q` at env step `t`:

- slot = `t + query_offset + q * query_stride`

Candidates targeting current step are aggregated with configurable weights.

## New SLURM scripts

Spatial:

- `point_policy/experiments/libero_spatial/slurm/train_debug_parallel_v1.sbatch`
- `point_policy/experiments/libero_spatial/slurm/eval_debug_parallel_v1.sbatch`

Object:

- `point_policy/experiments/libero_object/slurm/train_debug_parallel_v1.sbatch`
- `point_policy/experiments/libero_object/slurm/eval_debug_parallel_v1.sbatch`

## Output isolation

- CKPT summary: `point_policy/CKPT_SUMMARY_parallel_v1.md`
- Eval videos (default): `point_policy/exp_local/eval_videos_parallel_v1`

## Quick examples

Spatial train:

```bash
sbatch -p debug point_policy/experiments/libero_spatial/slurm/train_debug_parallel_v1.sbatch
```

Spatial eval:

```bash
sbatch -p debug --export=ALL,BC_WEIGHT=/path/to/snapshot/200000.pt \
  point_policy/experiments/libero_spatial/slurm/eval_debug_parallel_v1.sbatch
```

Enable temporal debug on eval:

```bash
EXTRA_ARGS='agent.emit_temporal_debug=true' \
sbatch -p debug --export=ALL,BC_WEIGHT=/path/to/snapshot/200000.pt,EXTRA_ARGS \
  point_policy/experiments/libero_spatial/slurm/eval_debug_parallel_v1.sbatch
```
