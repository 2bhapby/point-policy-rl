# Point-Policy Residual RL 재구성 가이드 (서술형)

이 문서는 구현 파일 경로나 라인 번호 인용 없이, 시스템 동작을 처음부터 다시 만들 수 있도록 연속 서술 형태로 정리한 문서다.

---

## 1. 목표

현재 시스템의 목표는 다음 한 줄로 요약된다.

Frozen BC 정책을 기본 행동 생성기로 유지하고, 오프폴리시 RL이 residual만 학습해서 최종 행동을 `base + residual`로 합성한다.

핵심 제약은 두 가지다.

1. 기존 BC 체크포인트는 절대 덮어쓰거나 재학습하지 않는다.
2. RL은 base 정책을 대체하지 않고 보정기로만 동작한다.

---

## 2. 전체 구조

전체는 네 개의 계층으로 구성된다.

1. 환경 계층: LIBERO 단일 task 환경 생성, step/reset, reward/종료 신호 관리
2. Base 정책 계층: Point-Policy BC를 frozen 상태로 로드해 action dict 생성
3. Residual RL 계층: state를 입력받아 residual action 출력, critic/actor 업데이트 수행
4. 데이터 계층: online replay와 offline demo replay를 동시에 관리하고 배치 단위로 혼합 샘플링

실행 시점에는 매 step마다 base action을 먼저 만들고, residual을 더해 최종 action을 만든다.

---

## 3. 관측, 상태, 행동 정의

### 3.1 관측

환경 관측은 point tracks와 low-dimensional feature를 포함한다.
feature의 마지막 값은 gripper 상태로 사용한다.
feature의 앞쪽 일부는 end-effector 위치 정보로 사용할 수 있다.

### 3.2 RL 상태 벡터

RL 상태 벡터는 다음을 순서대로 이어붙여 만든다.

1. point tracks 평탄화 값
2. gripper scalar
3. 선택적 eef position (설정으로 on/off)
4. base action 7차원

이 설계의 목적은 residual policy가 현재 관측뿐 아니라 base가 어떤 행동을 하려는지도 직접 보게 하는 데 있다.

### 3.3 행동

행동은 7차원으로 구성한다.

1. 위치 변화 3축
2. 회전 변화 3축
3. gripper 1축

base action과 residual action은 정규화 공간에서 합성한다.
즉 base와 residual을 같은 범위로 맞춘 뒤 더하고 clip한 다음, 환경 범위로 되돌려 실행한다.

---

## 4. Base 정책(BC) 동작 원칙

Base 정책은 훈련 완료된 BC 체크포인트를 로드한 뒤 inference-only로 사용한다.
매 episode 시작 시 내부 버퍼를 reset한다.
매 step에서 현재 관측을 넣어 action dict를 얻는다.
그 action dict는 환경 쪽 point-to-action 디코더를 거쳐 7차원 로봇 action으로 변환된다.

중요한 점은 base 정책 내부가 history/chunk를 이미 처리한다는 것이다.
따라서 residual 쪽은 step-wise 보정기여도 전체 구조가 모순되지 않는다.

---

## 5. Residual 정책 동작 원칙

Residual 정책은 오프폴리시 Q 기반 에이전트로 구성한다.
actor는 residual만 출력하고, critic은 최종 결합 행동의 가치를 학습한다.

실행 시 행동 생성 순서는 다음과 같다.

1. base action 계산
2. 상태 벡터 구성
3. residual action 샘플링
4. `combined = clip(base + residual)`
5. 결합 행동을 환경 액션 범위로 변환해서 step

warmup 구간에서는 residual을 무작위로 샘플링한다.
warmup 이후부터는 actor 출력을 사용한다.

---

## 6. 보상과 종료 처리

온라인 수집 보상은 sparse terminal 정책을 사용한다.

1. 성공 terminal이면 1
2. 실패 terminal이면 0
3. truncate/non-terminal이면 0

환경이 종료 직후 한 step 늦게 예외를 던지는 경우가 있어, 해당 케이스는 강제 terminal transition으로 흡수한다.
그 후 reset으로 이어서 학습을 진행한다.

---

## 7. Replay 설계

### 7.1 Online replay

환경 rollout에서 생성한 transition을 online replay에 계속 적재한다.
온라인 버퍼는 학습 진행과 함께 증가한다.

### 7.2 Offline replay

오프라인 데모는 시작 시 한 번 로드하여 transition으로 변환하고 offline replay에 넣는다.
offline 버퍼는 고정 크기이며, online처럼 step마다 증가하지 않는다.

### 7.3 혼합 샘플링

학습 배치는 online과 offline에서 각각 샘플링한 미니배치를 합쳐 만든다.
혼합 비율은 `offline_fraction`으로 제어한다.

예를 들어 배치 256, offline_fraction 0.5면 offline 128 + online 128이 필요하다.
offline transition 개수가 128보다 작으면 실제 혼합이 의도대로 동작하지 않을 수 있다.

---

## 8. Offline demo 변환 규칙

데모 소스는 pkl이며, 각 demo episode를 순회하면서 `(s, a, r, s', done)`를 만든다.

base action 생성 모드는 두 가지를 지원한다.

1. demo 기반 delta 추정
2. frozen BC를 실제로 돌려 추정

학습에 넣는 action은 delta-only가 아니라 full action 경로를 유지한다.
데모에 full action이 있으면 그것을 사용하고, 없으면 base action을 사용한다.

offline reward는 non-terminal과 terminal 값을 각각 별도 설정값으로 준다.

---

## 9. Loss와 업데이트

critic 업데이트는 TD target 기반이다.
target 행동은 다음 상태의 base action과 target actor residual을 결합해 만든다.
target Q는 ensemble 기반의 conservative 계산(min 계열)을 사용한다.

actor 업데이트는 결합 행동의 Q를 최대화하는 방향이다.
실구현에서는 `-Q`를 최소화하는 형태로 계산한다.

업데이트 스케줄은 보통 다음 형태를 사용한다.

1. env step 1번당 update loop 실행
2. critic 여러 번(예: 4회)
3. actor는 지연 업데이트(예: critic 4회 중 1회)

---

## 10. n-step, prioritized, fallback 동작

n-step return은 핵심 설정이다.
백엔드가 달라져도 n-step이 빠지지 않게 유지해야 한다.

prioritized replay를 요청해도 런타임 의존성 이슈가 있으면 uniform으로 fallback될 수 있다.
이때 중요한 것은 요청값과 실제 적용값을 메타에 모두 기록하는 것이다.

---

## 11. 하이퍼파라미터 운영 가이드

실제 운영에서 크게 영향이 큰 항목은 다음이다.

1. offline_fraction
2. offline_max_demos
3. critic target tau
4. residual_action_scale
5. warmup random residual scale
6. exploration stddev

현재 태스크에서는 데모가 50개이므로 `offline_max_demos=50`을 쓰는 것이 합리적이다.
그래야 `offline_fraction=0.5` 사용 시 배치 절반을 안정적으로 채울 수 있다.

---

## 12. 로그와 산출물

학습이 정상 동작하면 다음 파일이 항상 생성된다.

1. run_meta.json
2. train_log.csv
3. eval_log.csv
4. snapshot/latest.pt

W&B를 켜면 run이 생성되고 동일 step 축으로 지표가 업로드된다.

평가 스크립트에서는 옵션으로 영상(mp4) 저장이 가능하다.

---

## 13. 실행 절차(재현용)

1. BC 체크포인트 경로를 고정한다.
2. ResFiT 코드 루트를 설정한다.
3. 단일 task/suite를 확정한다.
4. offline demos 개수와 혼합 비율을 정한다.
5. tau, residual scale, stddev를 설정한다.
6. 학습 잡을 제출한다.
7. run_meta에서 요청값/실효값을 즉시 검증한다.
8. train/eval 로그가 주기적으로 쌓이는지 확인한다.
9. latest checkpoint가 갱신되는지 확인한다.
10. 종료 상태를 COMPLETED/TIMEOUT으로 분기해 후속 조치한다.

---

## 14. 실패 대응 체크리스트

### success가 계속 0일 때

1. offline buffer 크기가 충분한지 확인
2. offline_fraction이 실제로 적용됐는지 확인
3. sampling strategy가 fallback됐는지 확인
4. residual scale이 base를 압도하는지 확인
5. timeout으로 early stop되는지 확인

### 배치 혼합이 이상할 때

1. offline_batch_size와 offline transitions 크기 비교
2. offline_max_demos 증가
3. run_meta의 applied fraction 재확인

### 예외가 뜰 때

terminated episode 관련 예외는 강제 terminal 처리 경로가 작동하는지 로그로 확인한다.

---

## 15. 재구성 10단계 요약

1. Frozen BC 로딩 경로를 고정한다.
2. env 생성과 action bound를 확정한다.
3. state 벡터 형식을 고정한다.
4. residual actor 입력을 state/base-action 포함 형태로 맞춘다.
5. action 합성 규칙을 base+residual로 고정한다.
6. online/offline replay를 동시에 구축한다.
7. n-step과 delayed actor update를 반영한다.
8. sparse terminal reward를 적용한다.
9. run_meta/log/checkpoint 산출 규약을 맞춘다.
10. fallback/timeout/terminal 예외 대응을 포함해 운영한다.
