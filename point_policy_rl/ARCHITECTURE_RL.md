# Point-Policy RL Architecture (`_rl`)

This document visualizes the current residual RL structure in
`Point-Policy_codex_libero/point_policy_rl`.

## 1) Runtime Dataflow (ResFiT Core Backend)

```mermaid
flowchart TD
  subgraph EnvSide["Point-Policy / LIBERO Side"]
    ENV["LIBERO Env\n(build_single_env_from_bc_config)"]
    BASE["Frozen BC Base Policy\n(FrozenPointPolicyBaseRL)"]
    OBS["observation_to_state\n(point + gripper + base_action)"]
  end

  subgraph RLCore["ResFiT Core (Imported)"]
    Q["QAgent\n(resfit.rl_finetuning.off_policy.rl.q_agent)"]
    RBON["Online Replay\nTensorDictPrioritizedReplayBuffer"]
    RBOFF["Offline Replay\nTensorDictPrioritizedReplayBuffer"]
    MIX["Mixed Sampler\n(online/offline fraction)"]
    UPD["QAgent.update(...)"]
  end

  subgraph Offline["Offline Demo Builder"]
    DEMO["expert_demos/*.pkl"]
    BUILDER["build_offline_transitions_from_expert_demos_rl"]
  end

  ENV --> BASE
  ENV --> OBS
  BASE --> OBS
  OBS -->|"state -> agent obs\n(dummy image + state + base_action_norm)"| Q
  Q -->|"residual action (norm)"| ENV
  BASE -->|"base action (raw)"| ENV

  ENV -->|"executed combined action + transition"| RBON
  DEMO --> BUILDER --> RBOFF
  RBON --> MIX
  RBOFF --> MIX
  MIX --> UPD --> Q
```

## 2) Training Pipeline

```mermaid
flowchart LR
  A["Reset env + base policy history"] --> B["Rollout 1 step"]
  B --> C["Build transition\n(obs, action, reward, next_obs, done)"]
  C --> D["Push to online replay"]
  D --> E{"Update step?"}
  E -->|No| B
  E -->|Yes| F["Sample online/offline mixed batch"]
  F --> G["QAgent.update(...)"]
  G --> H["Soft target update + priority update(if PER)"]
  H --> I["Log / Eval / Checkpoint"]
  I --> B
```

## 3) Offline Transition Modes

`point_policy_rl/offline_demo_builder_rl.py` supports:

- `base_action_mode=demo_delta`
  Uses demo frame deltas as base action approximation.
- `base_action_mode=bc_track_delta`
  Uses frozen BC prediction-derived track delta as base action approximation.
- `transition_action_mode=residual_zero`
  Stores zero residual action in replay (residual target = 0).
- `transition_action_mode=combined_base`
  Stores combined/base action in replay (useful for ResFiT-style combined-action storage).

## 4) Main Files

- `point_policy_rl/train_resfit_residual_td3_rl.py`
  Main trainer using ResFiT `QAgent`, TorchRL replay, MultiStep transform.
- `point_policy_rl/offline_demo_builder_rl.py`
  Demo pkl to replay transition conversion.
- `point_policy_rl/base_policy_adapter_rl.py`
  Frozen Point-Policy BC checkpoint loader and inference adapter.
- `point_policy_rl/env_bridge_rl.py`
  LIBERO environment construction and state feature extraction bridge.
- `point_policy_rl/experiments_rl/libero_spatial/slurm/train_resfit_residual_td3_rl.sbatch`
  Spatial suite launcher.
- `point_policy_rl/experiments_rl/libero_object/slurm/train_resfit_residual_td3_rl.sbatch`
  Object suite launcher.

## 5) Current Design Note

`resfit` RL core is reused directly, while env/base-policy integration remains
Point-Policy-specific through `_rl` adapters.
