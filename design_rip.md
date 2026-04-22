# Quanser Qube-Servo 3 RL Control System Design (Hardware-first)

## 1) 전체 시스템 아키텍처

```text
[Quanser HIL Device Layer]
    - read: encoder(0,1), other(14000,14001)
    - write: pwm(0)
    - fixed sample time Ts (e.g., 0.005~0.01s)

[Safety Supervisor Layer]
    - PWM saturation/clipping
    - rate limit (duty slew limit)
    - watchdog/heartbeat
    - emergency stop condition
    - finally/atexit/signal -> pwm=0

[Environment Layer (Gym-like)]
    - state transform/normalization
    - reward/done computation
    - reset state machine
    - episode logger

[Controller Layer]
    - option A: rule-based swing-up + stabilizer baseline
    - option B: RL policy (discrete or continuous)

[Training + Evaluation Layer]
    - replay buffer / rollout storage
    - policy update schedule
    - curriculum
    - model checkpoint + safe evaluation mode
```

권장 시작점:
1. **하드웨어 I/O + 안전계층 + rule-based reset**를 먼저 완성
2. 그 다음 **baseline controller(스윙업 + 밸런싱)**로 동작 검증
3. 마지막으로 RL을 붙여 온라인/오프라인 학습 분리

---

## 2) 추천 강화학습 알고리즘 후보 3개와 장단점

## 후보 A: **SAC (Soft Actor-Critic, Continuous)**
- Action: 연속 
- 장점
  - 샘플 효율이 비교적 좋고(오프폴리시), 실제 하드웨어 실험 횟수를 줄이기 유리
  - 엔트로피 기반이라 탐색이 비교적 안정적
  - PWM 연속 제어에 자연스럽게 맞음
- 단점
  - 구현/튜닝이 DQN보다 복잡
  - 하이퍼파라미터 민감(temperature, target update, reward scaling)
- 하드웨어 적합성: **높음 (1순위 권장)**

## 후보 B: **PPO (Continuous Gaussian policy)**
- Action: 연속
- 장점
  - 구현 안정성/재현성이 좋아 입문에 유리
  - on-policy라 코드 흐름이 단순
- 단점
  - 샘플 효율이 낮아 실제 장비 학습엔 시간이 오래 걸릴 수 있음
  - 너무 잦은 파라미터 업데이트로 장비 피로 증가 가능
- 하드웨어 적합성: **중간 (시뮬레이터 사전학습 후 파인튜닝용)**

## 후보 C: **Double DQN / Dueling DQN (Discrete 5 actions)**
- Action: 이산(질문에서 제시한 5개 PWM)
- 장점
  - 구현이 가장 단순하고 디버깅 쉬움
  - 안전한 action set을 미리 고정하기 좋음
- 단점
  - 토크 해상도가 낮아 upright 근처 미세 제어가 불리
  - 채터링(좌우 진동) 발생 가능
- 하드웨어 적합성: **중간~낮음 (초기 프로토타입용)**

### Discrete vs Continuous 결론
- **최종 목표가 upright 유지 품질**이면 continuous(SAC/PPO)가 유리.
- **초기 안전 검증/파이프라인 점검**은 discrete(DQN)로 빠르게 시작 가능.
- 권장 전략: **DQN으로 장비 I/O 검증 -> SAC로 이행**.

---

## 3) State / Action / Reward / Done 조건 상세 설계

## 3-1. State 정의 (권장)
사용자 후보를 거의 그대로 채택:
- \(\theta_m\): motor angle [rad]
- \(\sin(\theta_p)\), \(\cos(\theta_p)\): pendulum angle 임베딩
- \(\dot{\theta}_m\): motor angular velocity [rad/s]
- \(\dot{\theta}_p\): pendulum angular velocity [rad/s]

추가 권장:
- 직전 action \(u_{t-1}\) 1개를 state에 넣으면 진동 감소에 도움.

## 3-2. 센서 변환/정규화
1. **엔코더 -> 라디안 변환**
   - 드라이버가 rad로 제공하면 그대로 사용.
   - 카운트 값이면 \(\theta = 2\pi \cdot \frac{count}{N_{enc}}\).
2. **각도 wrap**
   - \(\theta\)는 \([-\pi, \pi]\)로 wrap.
3. **속도 저역통과 필터**
   - \(\dot{\theta}_{f,t}=\alpha \dot{\theta}_{f,t-1} + (1-\alpha)\dot{\theta}_{raw,t}\), \(\alpha\approx0.8\sim0.95\).
4. **정규화 예시**
   - \(\theta_m\): \(\theta_m / \pi\) 후 clip [-1,1]
   - \(\dot{\theta}_m\): \(\dot{\theta}_m / 20\) clip
   - \(\dot{\theta}_p\): \(\dot{\theta}_p / 30\) clip
   - \(\sin,\cos\): 이미 [-1,1]

## 3-3. Action 설계
### Discrete
- 인덱스 {0,1,2,3,4} -> PWM {-0.35,-0.02,0,+0.02,+0.35}
- 장점: 안전한 제한이 명확
- 보완: 고주파 채터링 방지를 위해 **action hold 2~3 step** 권장

### Continuous
- policy output \(a\in[-1,1]\)
- 변환: \(u = 0.35a\)
- 안전 계층 적용:
  - clip: \([-0.35,0.35]\)
  - slew rate limit: \(|u_t-u_{t-1}|\le\Delta_u\) (예: 0.03/step)

## 3-4. Reward 설계 (swing-up + balance 공용)
권장 보상:
\[
r_t =
w_p\cos(\theta_p)
- w_{pm}\theta_m^2
- w_v\dot{\theta}_p^2
- w_m\dot{\theta}_m^2
- w_u u_t^2
- w_{du}(u_t-u_{t-1})^2
\]

- upright에서 \(\cos(\theta_p)\to 1\) 최대
- motor 과도 회전, 과속, 과제어 벌점 포함
- 예시 가중치 초기값:
  - \(w_p=2.0, w_{pm}=0.2, w_v=0.02, w_m=0.01, w_u=0.01, w_{du}=0.02\)

보너스(선택):
- \(|\theta_p| < 12^\circ\) & \(|\dot{\theta}_p|<1.5\)이면 +0.5

## 3-5. Done 조건
아래 중 하나면 종료:
1. 시간 제한 (예: 8~12초)
2. motor angle 한계 초과 (예: \(|\theta_m| > 100^\circ\))
3. 연속 N step 동안 위험 상태(과속/과전류 추정)
4. 사용자 E-stop

주의: done 여부와 무관하게 **종료 시 PWM=0 필수**.

---

## 4) Reset 전략 설계 (rule-based 우선)

에피소드 종료 후 reset은 RL 이전에 반드시 독립 검증:

1. **출력 0화**: 즉시 pwm=0, 100~300ms 유지
2. **진동 감쇠**: 작은 반대 방향 PWM으로 속도 감쇠 (PD-lite)
3. **모터 원점 복귀**:
   - 목표 \(\theta_m^{ref}=0\)
   - 제어 \(u = k_p(\theta_m^{ref}-\theta_m)-k_d\dot{\theta}_m\)
   - clip +-0.15 내에서 천천히 복귀
4. **복귀 완료 조건**:
   - \(|\theta_m|<5^\circ\), \(|\dot{\theta}_m|<0.5\,rad/s\), 0.5초 유지
5. **pendulum settling 대기**: 0.5~1.0초 pwm=0
6. reset timeout (예: 5초) 시 fault 로그 + 강제 종료

핵심: reset 중에는 RL action 무시하고 **rule-based FSM이 우선권**.

---

## 5) 학습 루프 설계

## 5-1. 단계 분리 vs 통합
### 대안 1) 단일 정책 (swing-up+balance 통합)
- 장점: 최종적으로 한 policy
- 단점: reward shaping 난이도 높고 초반 실패율 큼

### 대안 2) 단계 분리 (권장)
- Phase A: swing-up 정책/규칙
- Phase B: balance 정책
- 전이 조건: \(|\theta_p|<20^\circ\) 진입 시 balance 제어로 스위칭
- 장점: 학습 안정성↑, 장비 보호↑, 디버깅 용이

**권장:** 초기에는 `rule-based swing-up + RL balance`.
성능 확보 후 `end-to-end RL` 도전.

## 5-2. Practical training protocol
1. 시뮬레이터(또는 제한된 하드웨어) 사전학습
2. 하드웨어 파인튜닝 시
   - 짧은 에피소드
   - 보수적 action limit
   - 에피소드 간 냉각 시간
3. 주기적 평가 에피소드(no exploration noise)
4. 실패 로그(각도, 속도, PWM, done reason) 저장

---

## 6) Quanser API와 연결되는 Python 모듈 구조 제안

```text
project/
  config/
    hw.yaml                 # 채널, 샘플타임, 제한값
    rl_sac.yaml
  qube/
    io_quanser.py           # HIL open/read/write/close + watchdog
    safety.py               # clip, slew limit, estop, fault state
    signals.py              # wrap, filtering, normalization
    reset_fsm.py            # rule-based reset state machine
    env_qube.py             # gym.Env interface
  controllers/
    swingup_rule.py         # energy-based or heuristic swing-up
    balance_pid.py          # baseline stabilizer
  rl/
    train_sac.py
    train_ppo.py
    train_dqn.py
    replay.py
  scripts/
    run_eval.py
    run_safe_io_test.py
```

`io_quanser.py` 필수 메서드 예시:
- `open()`
- `read_raw()` / `read_state()`
- `write_pwm(u)`
- `zero_output()`
- `close()`

종료 안전장치:
- `try/finally`에서 `zero_output()` + `close()`
- `atexit.register(zero_output)`
- `signal(SIGINT/SIGTERM)` 핸들러에서 `zero_output()`

---

## 7) 안전장치와 디버깅 포인트

## 필수 안전장치
1. PWM hard clip: 절대 \(|u|\le0.35\)
2. PWM slew rate limit
3. watchdog 설정 + heartbeat 실패 시 출력 차단
4. fault state 진입 시 RL 정지, reset FSM로 이관
5. 예외 발생 시 `finally: pwm=0`
6. 긴급정지 키보드 인터럽트 (`q` 등)

## 필수 로깅/디버깅
- 샘플타임 실제 주기(지터)
- 각도 wrap 오류 여부
- velocity spike
- action saturation 비율
- done reason 통계
- reset 성공률/시간

하드웨어 보호 관점에서,
- **과속/과진동 탐지 임계값**을 먼저 넣고
- RL 성능보다 **장비 보호 규칙**이 우선되도록 설계.

---

## 8) 실제 코드 구현 순서

1. **I/O Smoke Test**
   - read/write 최소 코드
   - 종료 시 pwm=0 검증
2. **Safety Layer 구현**
   - clip/slew/watchdog/fault
3. **Reset FSM 구현**
   - episode 끝날 때 원점 복귀 검증
4. **신호처리 모듈 구현**
   - wrap/filter/normalize
5. **Gym Env 구현**
   - step/reset/done/reward
6. **Baseline 제어기 구현**
   - swing-up rule + balance PID
7. **RL 도입 1차**
   - DQN(이산) 또는 SAC(연속) 중 하나로 작은 실험
8. **RL 도입 2차 (권장 SAC)**
   - 하이퍼파라미터 튜닝 + 커리큘럼
9. **통합 평가**
   - 100 episode 성공률, 평균 유지시간, fault율

---

## Quanser 문서 기반 체크포인트
- Python HIL `write_pwm`은 즉시 PWM 출력 채널에 값을 씀.
- Qube-Servo 3 문서에서 watchdog 기반 출력 리셋 안전 메커니즘 사용 가능.
- SDK 최신 릴리스는 2026 (v26.0.5144)로 표시됨.

위 3가지는 환경/버전에 따라 세부 동작이 달라질 수 있으므로, 실제 장비에선 설치된 SDK 버전을 고정하고 동일 버전에서 재현 실험을 수행하는 것을 권장.
