# Quanser Qube-Servo 3 회전형 역진자 — TD3 강화학습

실하드웨어(**Quanser Qube-Servo 3**) 위에서 **TD3**(Twin Delayed DDPG)로 회전형 역진자(Furuta pendulum)를
아래에서 흔들어 올리고(**swing-up**) 정점에서 **무한히 균형**을 유지하는 프로젝트입니다.

핵심은 **보상 설계를 실패→개선하며 바꿔 보상을 최대화**한 과정입니다.
(스피닝 함정 → "잠깐 버티기" 보상 해킹 → 정점 안정 유지 보너스로 무한 균형 달성)

---

## 결과

- **2000 step(약 20초) 무한 균형**, 에피소드 보상 ≈ **5,600**
- 테스트 5/5 에피소드 모두 완주 (편차 < 4%)
- 정점 균형 유지 + 외란 복구(손으로 툭 건드려도 다시 정점으로 복귀)

---

## 디렉토리 구조

```
.
├── a_actor_and_twin_q_critic.py   # TD3 Actor / Twin Q-Critic / ReplayBuffer
├── b_td3_train.py                 # 학습 루프 (warmup, 에피소드 사이 일괄 업데이트, validation, best 저장)
├── c_td3_test.py                  # 테스트/시연 (결정론적 정책으로 평가)
├── envs/
│   ├── __init__.py
│   └── quanser_env.py             # Gymnasium 환경 (관측 · 보상 · 종료 조건)
└── hardware/
    ├── __init__.py
    └── quanser_reset.py           # HW 인터페이스: reset · 모터 자동 중앙보정 · 펜듈럼 영점 재보정 · 안전 종료 · LED
```

> **중요**: `quanser_env.py`는 `envs/`, `quanser_reset.py`는 `hardware/` 안에 두어야 import가 동작합니다
> (`from envs.quanser_env import QuanserEnv`, `from hardware.quanser_reset import ...`).
> 각 하위 폴더에는 빈 `__init__.py`가 필요합니다.

---

## 환경 정의 (MDP)

| 구분 | 정의 |
|------|------|
| **State (5차원)** | `motor angle`, `sin(pendulum angle)`, `cos(pendulum angle)`, `motor angular velocity`, `pendulum angular velocity` |
| **Action (연속)** | policy `[-1, 1]` → PWM `[-0.35, 0.35]` |
| **관측 정규화** | `motor/1.8`, `sin`, `cos`, `m_vel/20`, `p_vel/40` |
| **제어 주기** | 100 Hz (`CONTROL_TS = 0.01`) |

- 각도를 `sin`·`cos`로 분해해 `−π↔π` 불연속을 제거 → 학습 안정화
- 연속 행동으로 정점에서의 미세 균형 제어에 유리

### 보상 (`envs/quanser_env.py` 의 `_compute_reward`)

PDF 명세(각도 기반)를 따르되, 실패→개선을 거쳐 다음 형태로 정착했습니다.

```
핵심        r = (θ/π)²            # 정점 1, 바닥 0  (제곱 → 스피닝 평균 0.5→0.33로 완화)
            정점(|θ|>170°)에서 r ×= 2
            r += 0.1              # 생존항
안정화      − 정점 속도 페널티     # 정점에서 흔들리면 손해
            − 모터 이탈 페널티     # 모터가 한계로 가면 감점
무한 균형   + 정점 안정 유지 보너스 # 수직 AND 저속일 때만 가산 → "잠깐 버티기"는 못 받음
```

### 안전 종료 조건

- 모터 각도 한계 초과 (`MOTOR_SAFE_LIMIT`)
- 펜듈럼 과회전 (5바퀴 초과)
- 펜듈럼 과속

---

## 알고리즘: TD3

DDPG에 3가지 개선을 더한 연속 제어 알고리즘.

1. **Twin Q-Critic** — 두 Q 중 최솟값을 타깃으로 → Q값 과대평가 방지
2. **Delayed Policy Update** — Critic 2회당 Actor 1회 갱신 → 학습 안정화
3. **Target Policy Smoothing** — 타깃 액션에 클리핑된 노이즈 → 과적합 방지

### 주요 하이퍼파라미터 (`b_td3_train.py`)

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `max_steps` | 800 | 에피소드당 최대 스텝 |
| `learning_rate` | 3e-4 | Adam |
| `gamma` | 0.99 | 감가율 |
| `batch_size` | 256 | |
| `replay_buffer_size` | 1,000,000 | off-policy 버퍼 |
| `soft_update_tau` | 0.995 | target soft update |
| `learning_starts` | 5,000 | warmup(랜덤 행동) 스텝 |
| `exploration_noise` | 0.3 | 행동 수집 가우시안 노이즈 |
| `policy_update_delay` | 2 | delayed policy update |
| `target_policy_noise` / `clip` | 0.2 / 0.5 | target smoothing |
| `validation_num_episodes` | 5 | 검증 에피소드 수 |

> 제어 루프(100 Hz) 중에는 학습하지 않고 **에피소드가 끝난 뒤 일괄 업데이트**하여 실시간 제어를 보호합니다.

---

## 실행 방법

### 요구 사항

- Python 3.10+
- [Quanser SDK](https://github.com/quanser/quanser_sdk_win64/releases) 및 `quanser` Python 패키지 (Quanser Qube-Servo 3 하드웨어 필요)
- `torch`, `numpy`, `gymnasium`, `wandb`

```bash
pip install torch numpy gymnasium wandb
# Quanser SDK 설치 후:
pip install --find-links "C:\Program Files\Quanser\Quanser SDK\python" \
    quanser_api quanser_common quanser_communications quanser_devices quanser_hardware
```

### 학습

```bash
python b_td3_train.py
```

- 장치 전원 연결 + USB 연결 후, **펜듈럼을 아래로 매달아 완전히 정지**시킨 상태에서 실행
- 시작 시 모터가 양극단을 짚어 중앙을 자동 보정합니다 (`[INIT] 모터 극단: ...`)
- best 모델은 `models/td3_QuanserQube_<reward>_<time>_best.pth` 로 저장되고 `..._latest.pth` 가 갱신됩니다

### 테스트 / 시연

```bash
python c_td3_test.py
```

- 결정론적 정책(`exploration=False`)으로 평가
- `models/td3_QuanserQube_latest.pth` 를 로드 (특정 best 파일로 바꿔 지정 가능)

> ⚠️ **하드웨어 필수**: 이 프로젝트는 실제 Quanser Qube-Servo 3 장치 위에서 동작합니다. 시뮬레이터는 포함되어 있지 않습니다.

---

## 보상 설계 — 실패 → 개선 (이 프로젝트의 핵심)

| 단계 | 보상 | 에이전트의 행동 | 문제 / 개선 |
|------|------|----------------|-------------|
| ① | `(|θ|/π) + 정점보너스 + 생존항` | **스피닝** (빙빙 돌기) | 도는 동안 평균 ~0.5 + 생존항 → 안 세우고 도는 게 이득 → `(θ/π)²` 제곱으로 완화 |
| ② | `(θ/π)²` (제곱) | swing-up 성공, 그러나 **정점에서 1초만 버티고 반복** | 정점 ×2 보상 때문에 "잠깐 버티기"와 "오래 버티기" 차이가 작음 (보상 해킹) |
| ③ | ② + **속도 페널티 강화 + 정점 안정 유지 보너스** | **무한 균형** | "수직 AND 저속"일 때만 보너스 → 오래 안정적으로 서 있는 것만 보상 |

**교훈**: 보상 설계가 정책을 결정한다 — 같은 목표라도 보상 형태에 따라 스피닝 / 잠깐 버티기 / 무한 균형으로 갈린다.

---

## 참고

- 학습 곡선(validation 평균 보상)은 0 근처에서 머물다 swing-up을 발견하는 시점에 계단식으로 도약하여 ~2160에서 수렴합니다.
- 실하드웨어 특성상 매 reset에서 펜듈럼 영점을 재보정하고, 모터 중앙을 자동 보정하여 좌표계 일관성을 유지합니다.
