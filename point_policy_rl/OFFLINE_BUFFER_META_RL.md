# Offline Buffer Meta (Base-Generated, Success-Only)

이 문서는 `point_policy_rl/OFFLINE_BUFFER_META_RL_TEMPLATE.json`을 채울 때 사용하는 기준이다.

## 목적
- 사람 비디오 기반 파이프라인처럼 `hdf5 action`이 없는 경우에도,
  `base policy`로 생성한 action으로 offline RL transition을 구성한다.
- residual RL 업데이트에 필요한 최소 계약(`obs, action, reward, next_obs, done`)을 고정한다.

## 필수 저장 항목
1. `obs`: `observation_to_state` 결과 벡터
2. `action`: 해당 step의 robot action(offline에서는 base-generated action)
3. `reward`: sparse reward (`success done -> 1`, else `0`)
4. `next_obs`: 다음 step state
5. `done`: terminal 여부

## 강력 권장 항목
1. `base_action`
2. `next_base_action`
3. `episode_id`
4. `step_idx`
5. `success_episode` (episode-level flag)

## 현재 구현 정합 기준
1. `offline_base_action_mode="bc_track_delta"` 사용
2. `offline_transition_action_mode="combined_base"` 사용
3. `offline_success_only=true`
4. `online_store_all_transitions=true`
5. 학습 중 base action 재계산 금지(transition에 인코딩된 값 사용)

## 왜 action이 필수인가
- off-policy critic 업데이트는 `Q(s,a)`를 학습하므로 action이 없으면 TD target을 만들 수 없다.
- 따라서 `hdf5 action`이 없을 때는 `base-generated action`을 저장해서 대체해야 한다.

## 빠른 체크리스트
1. transition마다 `obs/action/reward/next_obs/done` 키가 존재하는가
2. `done=true`인 transition의 `reward==1.0`인가
3. `done=false`인 transition의 `reward==0.0`인가
4. `action` shape가 항상 `(7,)`인가
5. `offline buffer`에 성공 episode transition만 들어갔는가
