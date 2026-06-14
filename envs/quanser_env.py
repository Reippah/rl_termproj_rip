import math
import time

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from hardware.quanser_reset import (
    CONTROL_TS,
    MOTOR_SAFE_LIMIT,
    init_hardware,
    close_hardware,
    reset_motor,
    read_state,
    write_pwm,
    check_fault,
    set_led,
    LED_SWINGUP,
    LED_BALANCE,
    pendulum_spin_num,
)

# 보상 계수 
VEL_PENALTY_COEF       = 0.15   # 정점 근처 속도 페널티, 정점에서 빠르게 흔들리면 손해
MOTOR_POS_PENALTY_COEF = 0.05   # 모터 중앙 이탈 페널티 ((motor/limit)^2)
NEAR_TOP_THRESHOLD     = 0.7    # upright > 이 값이면 "정점 근처"로 간주

# 정점 안정 유지 보너스
UPRIGHT_BONUS          = 1.0    # 안정 유지 시 매 스텝 보너스 (정점 보상을 사실상 2배 이상으로)
UPRIGHT_BONUS_VEL      = 3.0    # 이 속도(rad/s) 미만일 때만 보너스 (흔들리면 못 받음)
UPRIGHT_BONUS_UPRIGHT  = 0.9    # -cos(angle) > 이 값 

# 모터가 안전 한계(MOTOR_SAFE_LIMIT)에 닿아 종료될 때의 감점
TERMINATION_PENALTY    = 30.0

# action -> PWM 변환 계수
ACTION_TO_PWM = 0.15

# 채터링/강타 억제: 빠른 좌우 반전, 급격한 큰 입력에 페널티
ACTION_RATE_COEF = 0.08

# 펜듈럼 안전 종료
PEND_SPIN_LIMIT = 5.0     # 5바퀴 제한
PEND_VEL_LIMIT  = 50.0    # rad/s


class QuanserEnv(gym.Env):
    """
    Observation (5): [motor_angle, sin(pend), cos(pend), motor_vel, pend_vel]
    Action (1):      [-1, 1] -> PWM = action * ACTION_TO_PWM

    Reward 구조:
      upright = -cos(pendulum_angle)   # 정점 +1, 바닥 -1
      r = upright
          - (정점 근처일 때) VEL_PENALTY_COEF * |pend_vel|
          - MOTOR_POS_PENALTY_COEF * |motor_angle|
          - ACTION_RATE_COEF * |Δaction|
          - (모터 한계 종료 시) TERMINATION_PENALTY
    """

    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 500, verbose_reset: bool = False):
        super().__init__()

        self.max_steps     = max_steps
        self.verbose_reset = verbose_reset
        self._step_count   = 0
        self._prev_action  = 0.0
        self._led_state    = None   # LED 단계 추적 (변할 때만 갱신 → 100Hz 스팸 방지)

        # 관측 정규화 스케일 (네트워크 입력용; 보상 계산은 raw 값 사용)
        self.norm_motor_angle = 1.8
        self.norm_motor_vel   = 20.0
        self.norm_pend_vel    = 40.0

        # 정규화 후 범위 ~[-1, 1] 유지 (네트워크 출력도 이 범위로 설계)
        obs_low  = np.array([-2.0, -1.0, -1.0, -2.0, -2.0], dtype=np.float32)
        obs_high = np.array([ 2.0,  1.0,  1.0,  2.0,  2.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([ 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self.card, self.channels = init_hardware()

    def _get_obs(self):
        motor_angle, pendulum_angle, motor_vel, pendulum_vel = read_state(
            self.card, self.channels
        )
        #정규화된 관측 (네트워크 입력) — reward 는 아래 raw 값으로 계산
        obs = np.array(
            [
                motor_angle / self.norm_motor_angle,
                math.sin(pendulum_angle),
                math.cos(pendulum_angle),
                motor_vel / self.norm_motor_vel,
                pendulum_vel / self.norm_pend_vel,
            ],
            dtype=np.float32,
        )
        return obs, pendulum_angle, motor_angle, pendulum_vel

    def _compute_reward(self, pendulum_angle: float, pendulum_vel: float,
                        motor_angle: float) -> float:
        """
        |angle|/π 대신 (angle/π)^2 사용:
          - 정점(±π) = 1, 바닥(0) = 0   (PDF와 동일 범위/형식 유지)
          - 스피닝 평균 ≈ 0.33 < PDF 0.5 -> 스피닝을 덜 보상(완화)
        (모터 한계 종료 감점은 step() 에서 별도 처리)
        """
        # PDF 형식 핵심 보상 (제곱 버전)
        r = (pendulum_angle / math.pi) ** 2       # [0, 1]: 정점 1, 바닥 0
        if abs(pendulum_angle) > 2.96706:         # 170° 초과 = 정점 근처
            r *= 2                                 # PDF r_a 정점 보너스
        r += 0.1                                   # PDF r_b 생존항
        # 여기까지: max = 1*2+0.1 = 2.1, min = 0+0.1 = 0.1 (PDF와 동일)

        # 실하드웨어 안정화 보조항 
        # 정점 근처에서만 속도 페널티 -> 균형 안정화 (스윙up 구간은 방해 안 함)
        if -math.cos(pendulum_angle) > NEAR_TOP_THRESHOLD:
            r -= VEL_PENALTY_COEF * abs(pendulum_vel)

        # 모터 중앙 이탈 페널티 (한계 근처 급증 -> 중간 펌핑은 관대, 강타 억제)
        motor_ratio = min(abs(motor_angle) / MOTOR_SAFE_LIMIT, 1.0)
        r -= MOTOR_POS_PENALTY_COEF * (motor_ratio ** 2)

        # 정점 안정 유지 보너스 (무한 균형 유도)
        # "진짜 위쪽(정점 ±26°) AND 속도 낮음"일 때만 추가 보너스
        if (-math.cos(pendulum_angle) > UPRIGHT_BONUS_UPRIGHT
                and abs(pendulum_vel) < UPRIGHT_BONUS_VEL):
            r += UPRIGHT_BONUS

        return r

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count  = 0
        self._prev_action = 0.0
        self._led_state   = None   # reset_motor 가 파랑 켜둠; 첫 step에서 재판단

        success = reset_motor(self.card, self.channels, verbose=self.verbose_reset)
        if not success:
            print("[ENV] reset_motor failed -- forcing continue.")

        obs, _, _, _ = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        loop_start = time.perf_counter()

        action_scalar = float(np.squeeze(action))
        action_scalar = float(np.clip(action_scalar, -1.0, 1.0))
        pwm = action_scalar * ACTION_TO_PWM

        write_pwm(self.card, self.channels, pwm)

        elapsed = time.perf_counter() - loop_start
        if elapsed < CONTROL_TS:
            time.sleep(CONTROL_TS - elapsed)

        next_obs, pendulum_angle, motor_angle, pendulum_vel = self._get_obs()
        self._step_count += 1

        # LED 단계 표시: Balancing(정점근처)=초록, Swing-up=노랑 (변할 때만 갱신)
        balancing = (-math.cos(pendulum_angle)) > NEAR_TOP_THRESHOLD
        desired_led = "balance" if balancing else "swingup"
        if desired_led != self._led_state:
            set_led(self.card, *(LED_BALANCE if balancing else LED_SWINGUP))
            self._led_state = desired_led

        # 보상 계산 
        reward = self._compute_reward(pendulum_angle, pendulum_vel, motor_angle)

        # 채터링 페널티
        rate_penalty = ACTION_RATE_COEF * abs(action_scalar - self._prev_action)
        self._prev_action = action_scalar
        reward -= rate_penalty

        # 종료 조건
        hit_motor_limit = bool(abs(motor_angle) > MOTOR_SAFE_LIMIT)   # 모터 각도 한계
        spin_num  = pendulum_spin_num(self.channels)                  # 펜듈럼 회전수
        over_spin = bool(abs(spin_num) > PEND_SPIN_LIMIT)             # 과회전(스피닝)
        over_vel  = bool(abs(pendulum_vel) > PEND_VEL_LIMIT)          # 과속

        terminated = hit_motor_limit or over_spin or over_vel

        has_fault, fault_msg = check_fault(self.card, self.channels)
        if has_fault:
            terminated = True

        # 감점: 행동성 위반(모터 한계 돌진 + 스피닝)에만. 과속/stall 은 안전 컷오프라 감점 없음
        # (과속에 감점하면 swing-up 의 빠른 펌핑까지 벌해 학습을 방해할 수 있음)
        if hit_motor_limit or over_spin:
            reward -= TERMINATION_PENALTY

        truncated = bool(self._step_count >= self.max_steps)

        if terminated:
            write_pwm(self.card, self.channels, 0.0)

        info = {
            "motor_angle"    : motor_angle,
            "pendulum_angle" : pendulum_angle,
            "pendulum_vel"   : pendulum_vel,
            "spin_num"       : spin_num,
            "pwm"            : pwm,
            "rate_penalty"   : rate_penalty,
            "step"           : self._step_count,
            "terminated_by"  : ("motor" if hit_motor_limit else
                                "spin" if over_spin else
                                "vel" if over_vel else
                                "fault" if has_fault else ""),
            "fault"          : fault_msg if has_fault else "",
        }

        return next_obs, reward, terminated, truncated, info

    def close(self):
        close_hardware(self.card, self.channels)
