# Other Server Full Process (Train + Eval)

이 문서는 **다른 서버에서 재학습까지** 바로 수행하기 위한 전체 절차입니다.
기준 실험은 `scene3 image_only (123809 계열)`입니다.

---

## 0) 무엇이 필수인가

필수 자산:
1. 코드 2개 레포
- `Point-Policy_codex_libero` (RL wrapper + env bridge)
- `residual-offpolicy-rl` (ResFiT/QAgent 백엔드)

2. Base BC checkpoint (frozen)
- 예: `.../snapshot/100000.pt`

3. Offline dataset (image 포함)
- **image-only 재학습이면 `*_img50.pkl` 필수**
- 일반 `*.pkl`(point만)로는 image-only 학습 재현 불가

4. 실행 환경
- CUDA GPU, MuJoCo, robosuite, LIBERO, conda env

---

## 1) 소스 서버에서 자산 준비 (현재 서버)

아래 2개 파일이 있는지 먼저 확인:

```bash
ls -lh /sjw_alinlab2/home/sanghyeok/Point-Policy_codex_libero/point_policy/exp_local/offline_rlds/libero10_3_turn_on_the_stove_and_put_the_moka_pot_on_it_img50.pkl
ls -lh /sjw_alinlab2/home/sanghyeok/Point-Policy_codex_libero/point_policy/exp_local/2026.02.17/point_policy_libero_object_basefix_v1_lib10_scene3_obj15_sephead_nocond_bf1_0217/debug_201911_lib10_scene3_obj15_sephead_nocond_bf1_0217/004042_hidden_dim_256/snapshot/100000.pt
```

체크섬 저장(전송 검증용):

```bash
cd /sjw_alinlab2/home/sanghyeok
sha256sum \
  Point-Policy_codex_libero/point_policy/exp_local/offline_rlds/libero10_3_turn_on_the_stove_and_put_the_moka_pot_on_it_img50.pkl \
  Point-Policy_codex_libero/point_policy/exp_local/2026.02.17/point_policy_libero_object_basefix_v1_lib10_scene3_obj15_sephead_nocond_bf1_0217/debug_201911_lib10_scene3_obj15_sephead_nocond_bf1_0217/004042_hidden_dim_256/snapshot/100000.pt \
  > /tmp/scene3_img50_assets.sha256
```

---

## 2) 대상 서버로 전송

`<DST_USER>@<DST_HOST>`만 바꿔서 실행.

### 2-A) 코드 클론(권장)

```bash
# 대상 서버
cd $HOME
git clone git@github.com:2bhapby/point-policy-rl.git Point-Policy_codex_libero
cd Point-Policy_codex_libero
git checkout feat/parallel-v1-delta-osc-scale

git clone https://github.com/amazon-far/residual-offpolicy-rl.git $HOME/residual-offpolicy-rl
```

### 2-B) 데이터/체크포인트 전송

```bash
# 현재 서버에서 실행
scp \
  /sjw_alinlab2/home/sanghyeok/Point-Policy_codex_libero/point_policy/exp_local/offline_rlds/libero10_3_turn_on_the_stove_and_put_the_moka_pot_on_it_img50.pkl \
  <DST_USER>@<DST_HOST>:/home/<DST_USER>/Point-Policy_codex_libero/point_policy/exp_local/offline_rlds/

scp \
  /sjw_alinlab2/home/sanghyeok/Point-Policy_codex_libero/point_policy/exp_local/2026.02.17/point_policy_libero_object_basefix_v1_lib10_scene3_obj15_sephead_nocond_bf1_0217/debug_201911_lib10_scene3_obj15_sephead_nocond_bf1_0217/004042_hidden_dim_256/snapshot/100000.pt \
  <DST_USER>@<DST_HOST>:/home/<DST_USER>/Point-Policy_codex_libero/point_policy/exp_local/2026.02.17/point_policy_libero_object_basefix_v1_lib10_scene3_obj15_sephead_nocond_bf1_0217/debug_201911_lib10_scene3_obj15_sephead_nocond_bf1_0217/004042_hidden_dim_256/snapshot/

scp /tmp/scene3_img50_assets.sha256 <DST_USER>@<DST_HOST>:/home/<DST_USER>/scene3_img50_assets.sha256
```

검증(대상 서버):

```bash
cd /home/<DST_USER>
sha256sum -c scene3_img50_assets.sha256
```

---

## 3) 대상 서버 환경 세팅

```bash
cd /home/<DST_USER>/Point-Policy_codex_libero
conda create -n libero_codex python=3.8 -y
conda activate libero_codex
pip install --upgrade pip
pip install torch torchvision
pip install numpy opencv-python imageio matplotlib pyyaml tqdm wandb hydra-core omegaconf
```

주의:
- MuJoCo/robosuite/LIBERO는 서버 환경에 맞게 별도 설치 필요
- GPU 확인: `nvidia-smi`

---

## 4) 최소 동작 점검 (데이터/CKPT)

```bash
cd /home/<DST_USER>/Point-Policy_codex_libero
python - <<'PY'
import pickle
p='point_policy/exp_local/offline_rlds/libero10_3_turn_on_the_stove_and_put_the_moka_pot_on_it_img50.pkl'
with open(p,'rb') as f:
    d=pickle.load(f)
obs=d['episodes'][0]['steps'][0]['observation']
print('pixels1' in obs, obs['pixels1'].shape if 'pixels1' in obs else None)
PY

ls -lh point_policy/exp_local/2026.02.17/point_policy_libero_object_basefix_v1_lib10_scene3_obj15_sephead_nocond_bf1_0217/debug_201911_lib10_scene3_obj15_sephead_nocond_bf1_0217/004042_hidden_dim_256/snapshot/100000.pt
```

정상 기준:
- `pixels1` 존재하고 shape 출력됨
- BC ckpt 파일 존재

---

## 5) 학습 실행 (scene3 image-only, 123809 하이퍼파라미터)

### 5-A) 인터랙티브 실행(디버그)

```bash
cd /home/<DST_USER>/Point-Policy_codex_libero
conda activate libero_codex

export REPO_ROOT=$PWD
export RESFIT_ROOT=/home/<DST_USER>/residual-offpolicy-rl

BC_WEIGHT=$REPO_ROOT/point_policy/exp_local/2026.02.17/point_policy_libero_object_basefix_v1_lib10_scene3_obj15_sephead_nocond_bf1_0217/debug_201911_lib10_scene3_obj15_sephead_nocond_bf1_0217/004042_hidden_dim_256/snapshot/100000.pt
OFFLINE_PKL=$REPO_ROOT/point_policy/exp_local/offline_rlds/libero10_3_turn_on_the_stove_and_put_the_moka_pot_on_it_img50.pkl

python point_policy_rl/train_resfit_residual_td3_rl_rel.py \
  --resfit-root "$RESFIT_ROOT" \
  --suite libero_object_basefix_v1 \
  --benchmark-name LIBERO_10 \
  --task-name KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it \
  --task-order-index -1 \
  --bc-weight "$BC_WEIGHT" \
  --offline-source expert_demo \
  --offline-demo-root "$OFFLINE_PKL" \
  --offline-max-demos 50 \
  --residual-obs-mode image_only \
  --image-obs-key pixels1 \
  --state-coord-mode both_eef \
  --steps 400000 \
  --batch-size 256 \
  --buffer-size 300000 \
  --gamma 0.99 \
  --n-step 3 \
  --offline-fraction 0.5 \
  --critic-target-tau 0.005 \
  --residual-action-scale 0.2 \
  --random-action-noise-scale 0.2 \
  --stddev-min 0.025 \
  --stddev-max 0.025 \
  --warmup-steps 5000 \
  --save-every 10000 \
  --eval-every 5000 \
  --eval-episodes 5 \
  --device cuda \
  --strict-resfit \
  --run-name resfit_img_scene3_retrain_server
```

### 5-B) Slurm 제출

```bash
cd /home/<DST_USER>/Point-Policy_codex_libero

export REPO_ROOT=$PWD
export RESFIT_ROOT=/home/<DST_USER>/residual-offpolicy-rl
export CONDA_BASE=/home/<DST_USER>/miniconda3
export CONDA_ENV=libero_codex

export BC_WEIGHT=$REPO_ROOT/point_policy/exp_local/2026.02.17/point_policy_libero_object_basefix_v1_lib10_scene3_obj15_sephead_nocond_bf1_0217/debug_201911_lib10_scene3_obj15_sephead_nocond_bf1_0217/004042_hidden_dim_256/snapshot/100000.pt
export TASK_NAME=KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it
export RUN_NAME=resfit_img_scene3_retrain_server
export EXTRA_ARGS="--suite libero_object_basefix_v1 --benchmark-name LIBERO_10 --task-order-index -1 --steps 400000 --offline-source expert_demo --offline-demo-root $REPO_ROOT/point_policy/exp_local/offline_rlds/libero10_3_turn_on_the_stove_and_put_the_moka_pot_on_it_img50.pkl --offline-max-demos 50 --residual-obs-mode image_only --image-obs-key pixels1 --state-coord-mode both_eef --offline-fraction 0.5 --critic-target-tau 0.005 --residual-action-scale 0.2 --random-action-noise-scale 0.2 --stddev-min 0.025 --stddev-max 0.025 --strict-resfit"

sbatch -p background --gres=gpu:1 \
  --job-name=pp-rl-s3-imgonly \
  --export=ALL,REPO_ROOT,RESFIT_ROOT,CONDA_BASE,CONDA_ENV,BC_WEIGHT,TASK_NAME,RUN_NAME,EXTRA_ARGS \
  point_policy_rl/experiments_rl/libero_object/slurm/train_resfit_residual_td3_rl_rel.sbatch
```

---

## 6) 학습 결과 확인

```bash
# Slurm
squeue -u $USER -o "%.10i %.10P %.24j %.2t %.10M %.6D %R"

# 런 디렉토리
ls -lah /home/<DST_USER>/Point-Policy_codex_libero/point_policy/exp_local_rl_rel/*/residual_td3_libero_object_basefix_v1_rl/*resfit_img_scene3_retrain_server*

# 핵심 산출물
# run_meta.json / train_log.csv / eval_log.csv / snapshot/latest.pt
```

---

## 7) 평가 (체크포인트 1개)

```bash
cd /home/<DST_USER>/Point-Policy_codex_libero
conda activate libero_codex

CKPT=/home/<DST_USER>/Point-Policy_codex_libero/point_policy/exp_local_rl_rel/<DATE>/residual_td3_libero_object_basefix_v1_rl/resfit_img_scene3_retrain_server/snapshot/400000.pt

python point_policy_rl/eval_residual_td3_rl_rel.py \
  --ckpt "$CKPT" \
  --episodes 10 \
  --device cuda \
  --resfit-root /home/<DST_USER>/residual-offpolicy-rl \
  --suite libero_object_basefix_v1 \
  --benchmark-name LIBERO_10 \
  --task-name KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it \
  --task-order-index -1 \
  --state-coord-mode both_eef \
  --image-size-override 84
```

---

## 8) 흔한 실패 원인

1. image-only인데 offline pkl이 `*_img50.pkl`이 아님
- 증상: image key 관련 에러 또는 성능 붕괴

2. base ckpt 경로 오타
- 증상: 로딩 실패 또는 base-only 성능 0 근처

3. MuJoCo/robosuite/LIBERO 버전 불일치
- 증상: env 생성 실패, 렌더 키 불일치

4. 절대경로 하드코딩
- 해결: 위 명령처럼 환경변수(`REPO_ROOT`, `RESFIT_ROOT`)로 치환

---

## 9) 재현성 체크 (권장)

학습 seed와 eval seed를 분리:
- train seed: `0,1,2` (최소 3개)
- eval seed: 공통 세트 `100~109`

로그 비교:
- `eval_success` mean/std
- `critic_qt`, `actor_loss`, `critic_loss`
- `diag_clip_rate_dim`, `diag_res_over_base`

