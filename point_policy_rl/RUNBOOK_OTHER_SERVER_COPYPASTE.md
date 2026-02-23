# Point-Policy Residual RL Other-Server Runbook (Copy-Paste)

이 문서는 다른 서버에서 **그대로 복붙**해서 실행하는 용도입니다.

기준 코드:
- repo: `git@github.com:2bhapby/point-policy-rl.git`
- branch: `feat/parallel-v1-delta-osc-scale`
- commit: `16544f7`

---

## 0) 준비

필수:
1. NVIDIA GPU + CUDA 드라이버
2. MuJoCo / robosuite / LIBERO 실행 환경
3. Conda (예: miniconda)
4. Hugging Face 토큰 (`HF_TOKEN`)

권장 파티션:
- 빠른 테스트: `debug`
- 대량 sweep: `background`

---

## 1) 코드 클론

```bash
cd $HOME
git clone git@github.com:2bhapby/point-policy-rl.git
cd point-policy-rl
git checkout feat/parallel-v1-delta-osc-scale
git rev-parse --short HEAD
```

`residual-offpolicy-rl`도 함께 필요:

```bash
cd $HOME
git clone git@github.com:amazon-science/residual-offpolicy-rl.git
```

---

## 2) 파이썬 환경

```bash
conda create -n libero_codex python=3.8 -y
conda activate libero_codex
pip install --upgrade pip
pip install torch torchvision
pip install numpy opencv-python imageio matplotlib pyyaml tqdm
pip install huggingface_hub wandb
```

`LIBERO/robosuite/MuJoCo`는 서버 환경에 맞게 별도 설치 필요.

---

## 3) Hugging Face artifact 다운로드

manifest 기반으로 BC ckpt / offline pkl / residual run artifact를 내려받습니다.

```bash
cd $HOME/point-policy-rl
export HF_TOKEN=hf_xxx_your_token
export REPO_ROOT=$PWD

python point_policy_rl/hf_sync_artifacts.py \
  --mode download \
  --manifest point_policy_rl/hf_artifacts_manifest.yaml \
  --repo-root "$REPO_ROOT" \
  --download-root "$REPO_ROOT"
```

검증:

```bash
cd $HOME/point-policy-rl
python point_policy_rl/hf_sync_artifacts.py \
  --mode validate \
  --manifest point_policy_rl/hf_artifacts_manifest.yaml \
  --repo-root "$PWD" \
  --strict
```

---

## 4) 공통 환경변수 (복붙)

```bash
cd $HOME/point-policy-rl
conda activate libero_codex

export REPO_ROOT=$PWD
export RESFIT_ROOT=$HOME/residual-offpolicy-rl
export CONDA_BASE=$HOME/miniconda3
export CONDA_ENV=libero_codex
```

---

## 5) 단일 eval (scene3, image-only residual ckpt)

```bash
cd $REPO_ROOT

export CKPT=$REPO_ROOT/point_policy/exp_local_rl_rel/2026.02.20/residual_td3_libero_object_basefix_v1_rl/123809_resfit_img_scene3_pkl50_fix0220/snapshot/400000.pt

python point_policy_rl/eval_residual_td3_rl_rel.py \
  --ckpt "$CKPT" \
  --episodes 10 \
  --device cuda \
  --resfit-root "$RESFIT_ROOT" \
  --suite libero_object_basefix_v1 \
  --task-name KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it \
  --benchmark-name LIBERO_10 \
  --task-order-index -1 \
  --state-coord-mode both_eef \
  --image-size-override 84
```

---

## 6) SLURM eval (sbatch)

```bash
cd $REPO_ROOT

export CKPT=$REPO_ROOT/point_policy/exp_local_rl_rel/2026.02.20/residual_td3_libero_object_basefix_v1_rl/123809_resfit_img_scene3_pkl50_fix0220/snapshot/400000.pt

sbatch -p background \
  --job-name=pp-eval-scene3 \
  --export=ALL,REPO_ROOT,CONDA_BASE,CONDA_ENV,CKPT,EPISODES=10,SUITE=libero_object_basefix_v1,BENCHMARK_NAME=LIBERO_10,TASK_ORDER_INDEX=-1,STATE_COORD_MODE=both_eef,TASK_NAME=KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it,IMAGE_SIZE_OVERRIDE=84,SAVE_VIDEO=0 \
  point_policy_rl/experiments_rl/libero_object/slurm/eval_residual_td3_rl_rel.sbatch
```

---

## 7) SLURM 학습 (scene3 예시)

```bash
cd $REPO_ROOT

export BC_WEIGHT=$REPO_ROOT/point_policy/exp_local/2026.02.17/point_policy_libero_object_basefix_v1_lib10_scene3_obj15_sephead_nocond_bf1_0217/debug_201911_lib10_scene3_obj15_sephead_nocond_bf1_0217/004042_hidden_dim_256/snapshot/100000.pt
export TASK_NAME=KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
export RUN_NAME=resfit_rel_scene3_serverB
export EXTRA_ARGS='--suite libero_object_basefix_v1 --benchmark-name LIBERO_10 --task-order-index -1 --steps 100000 --offline-source expert_demo --offline-demo-root point_policy/exp_local/offline_rlds --offline-max-demos 50 --residual-obs-mode image_only --image-obs-key pixels1 --state-coord-mode both_eef --offline-fraction 0.5 --critic-target-tau 0.005 --residual-action-scale 0.2 --random-action-noise-scale 0.2'

sbatch -p background \
  --job-name=pp-train-scene3 \
  --export=ALL,REPO_ROOT,RESFIT_ROOT,CONDA_BASE,CONDA_ENV,BC_WEIGHT,TASK_NAME,RUN_NAME,EXTRA_ARGS \
  point_policy_rl/experiments_rl/libero_object/slurm/train_resfit_residual_td3_rl_rel.sbatch
```

---

## 8) checkpoint sweep eval curve (10k 간격, 50 episodes)

`RUN_DIR`는 학습 런 디렉토리로 바꿔서 사용.

```bash
cd $REPO_ROOT

export RUN_DIR=$REPO_ROOT/point_policy/exp_local_rl_rel/2026.02.20/residual_td3_libero_object_basefix_v1_rl/123809_resfit_img_scene3_pkl50_fix0220
export OUTPUT_DIR=$REPO_ROOT/point_policy_rl/visual_reports/2026-02-23_eval_curve_bg_scene3_e50

mkdir -p "$OUTPUT_DIR"

for step in $(seq 10000 10000 400000); do
  step6=$(printf "%06d" "$step")
  k=$((step/1000))
  sbatch -p background \
    --job-name="pp-e50-s3-${k}k-bg" \
    --gres=gpu:1 --cpus-per-task=8 --mem=32G \
    --output="$REPO_ROOT/point_policy_rl/experiments_rl/libero_object/slurm_logs/pp-e50-s3-${k}k-bg_%j.out" \
    --error="$REPO_ROOT/point_policy_rl/experiments_rl/libero_object/slurm_logs/pp-e50-s3-${k}k-bg_%j.err" \
    --export=ALL,REPO_ROOT,CONDA_BASE,CONDA_ENV,RESFIT_ROOT,RUN_DIR,EPISODES=50,CKPT_STRIDE=10000,MIN_STEP=$step,MAX_STEP=$step,DEVICE=cuda,SUITE=libero_object_basefix_v1,TASK_NAME=KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it,BENCHMARK_NAME=LIBERO_10,TASK_ORDER_INDEX=-1,STATE_COORD_MODE=both_eef,IMAGE_SIZE_OVERRIDE=84,OUTPUT_TAG=scene3_img_fix0220_bg_e50_step${step6},OUTPUT_DIR=$OUTPUT_DIR \
    point_policy_rl/experiments_rl/libero_object/slurm/sweep_eval_success_curve_rl_rel.sbatch
done
```

모니터링:

```bash
squeue -u $USER -o "%.10i %.10P %.24j %.2t %.10M %.6D %R"
```

---

## 9) 결과 병합 + 플롯 생성

```bash
cd $REPO_ROOT
python point_policy_rl/merge_eval_curve_shards_rl_rel.py \
  --input-dir point_policy_rl/visual_reports/2026-02-23_eval_curve_bg_scene3_e50 \
  --pattern 'scene3_img_fix0220_bg_e50_step*_eval_curve_e50_s10000.csv' \
  --out-tag scene3_img_fix0220_bg_e50_merged
```

생성물:
- `.../scene3_img_fix0220_bg_e50_merged.csv`
- `.../scene3_img_fix0220_bg_e50_merged.json`
- `.../scene3_img_fix0220_bg_e50_merged.png`

---

## 10) 트러블슈팅

1. `ImportError: cannot import name 'PointEncoder'`
- 환경/패키지 불일치 가능성이 큼. 동일 conda env, 동일 branch/commit 재확인.

2. `sbatch/squeue ... slurm stream socket ...`
- 클러스터 컨트롤러 일시 장애. 잠시 후 재시도.

3. EGL teardown warning
- `EGLError: EGL_NOT_INITIALIZED`는 종료 시 warning으로 자주 발생.
- `return_code=0`이면 eval 결과는 보통 정상 저장됨.

4. 결과 파일이 섞여 보일 때
- `OUTPUT_TAG`를 실험별로 다르게 지정.
- 병합은 반드시 해당 tag pattern으로만 수행.

---

## 11) 최소 재현 체크리스트

1. `git rev-parse --short HEAD`가 `16544f7`인지 확인
2. `hf_sync_artifacts.py --mode validate --strict` 통과
3. 단일 eval 1회 성공 (`return_code=0`, mean_success 출력)
4. train 후 `run_meta.json`, `train_log.csv`, `eval_log.csv`, `snapshot/latest.pt` 생성 확인

