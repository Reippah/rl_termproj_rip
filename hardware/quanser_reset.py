from quanser.hardware import HIL, MAX_STRING_LENGTH
from array import array
import math
import time

COUNTS_PER_REV = 2048
COUNTS_PER_RAD = COUNTS_PER_REV / (2 * math.pi)
TACH_COUNTS_TO_RAD_PER_SEC = (2 * math.pi) / COUNTS_PER_REV

DEADBAND_PWM      = 0.04
RESET_THRESHOLD   = 0.10
RESET_MAX_PWM     = 0.20   
RESET_KP          = 0.4    
RESET_TIMEOUT     = 15.0   
CONTROL_TS        = 0.01
MOTOR_SAFE_LIMIT  = 2.2    
MOTOR_RESET_LIMIT = 2.8

SETTLE_VEL_THRESHOLD = 0.3
SETTLE_TIMEOUT       = 12.0   # 펜듈럼이 실제로 멈출 때까지 대기.

STALL_MAX_RETRY = 3   # reset 중 stall 발생 시 재시도 횟수

# 모터 중앙 자동 보정 (참고코드 _reset_init_count 방식) 
# 모터를 양극단까지 밀어 그 중점을 "물리적 중앙(0)"으로 잡음.
# 손으로 중앙 맞출 필요 없이 매번 동일한 진짜 중앙이 나옴 → reset/좌표계 일관성.
CALIB_PWM        = 0.06    # 극단으로 미는 약한 PWM
CALIB_PUSH_STEPS = 300     # 각 방향 미는 횟수
CALIB_TS         = 0.01

# reset 한계 탈출 (참고코드 reset_helper 방식)
# 모터가 이 각도(rad) 넘어가면 약한 P제어 대신 강하게 밀어 빼냄
RESET_STRONG_PUSH_RAD = 1.4
RESET_STRONG_PUSH_PWM = 0.35

# PHASE 0: 펜듈럼 에너지 블리딩
# 센터링 전에 휘도는 펜듈럼을 진정시키는 단계.
BLEED_TIMEOUT       = 5.0    # 펜듈럼 진정 최대 대기(초). 못 멈춰도 이후 센터링은 진행
BLEED_VEL_THRESHOLD = 1.0    # 이 속도 미만이면 충분히 진정된 것으로 간주(rad/s)


def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def calibrate_motor_center(card, channels) -> int:
    """
    모터를 양극단까지 밀어 그 중점 카운트를 반환 (= 물리적 중앙).
    참고코드 _reset_init_count 방식. 사람이 손으로 중앙 맞출 필요 없음.
    """
    pwm_ch  = channels["pwm_ch"]
    enc_ch  = channels["motor_enc_ch"]
    enc_val = channels["motor_enc_val"]

    extremes = {}
    for duty in (-CALIB_PWM, CALIB_PWM):
        for _ in range(CALIB_PUSH_STEPS):
            card.write_pwm(pwm_ch, 1, array('d', [duty]))
            time.sleep(CALIB_TS)
        card.read_encoder(enc_ch, 1, enc_val)
        extremes[duty] = enc_val[0]
    card.write_pwm(pwm_ch, 1, array('d', [0.0]))   # 정지

    center = (extremes[-CALIB_PWM] + extremes[CALIB_PWM]) // 2
    print(f"[INIT] 모터 극단: -={extremes[-CALIB_PWM]}, +={extremes[CALIB_PWM]} "
          f"→ 중앙={center}")
    return center


def init_hardware(settle_time: float = 1.5, calibrate: bool = True):
    """
    하드웨어 초기화 + 영점 보정.

    calibrate=True (권장):
      - 모터: 양극단을 밀어 중점을 물리적 중앙(0)으로 자동 보정 → 손 보정 불필요
      - 펜듈럼: 자유 매달림 상태에서 여러 번 읽어 평균을 0(아래)으로
    calibrate=False:
      - 현재 위치를 그대로 0으로 (수동 보정; 시작 자세를 직접 맞춰야 함)

    calibrate=True 라도 펜듈럼은 자유롭게 매달려 정지한 상태에서 시작할 것
    """
    card = HIL("qube_servo3_usb", "0")
    card.set_card_specific_options("pwm_en=1", MAX_STRING_LENGTH)

    input_channels  = array('I', [1])
    output_channels = array('I', [0])
    card.set_digital_directions(
        input_channels, len(input_channels),
        output_channels, len(output_channels),
    )
    card.write_digital(array('I', [0]), 1, array('I', [1]))

    channels = {
        "pwm_ch"          : array('I', [0]),
        "motor_enc_ch"    : array('I', [0]),
        "pendulum_enc_ch" : array('I', [1]),
        "tach_motor_ch"   : array('I', [14000]),
        "tach_pend_ch"    : array('I', [14001]),
        "enable_ch"       : array('I', [0]),
        "motor_enc_val"   : array('i', [0]),
        "pendulum_enc_val": array('i', [0]),
        "tach_motor_val"  : array('d', [0.0]),
        "tach_pend_val"   : array('d', [0.0]),
        "motor_bias"      : 0,
        "pendulum_bias"   : 0,
        "pendulum_spin_ref": 0,   # 회전수 종료용 기준(매 reset 바닥에서 갱신) — spin_num 이 에피소드마다 0부터
    }

    if calibrate:
        # 모터 물리적 중앙 자동 탐색 (양극단 midpoint)
        channels["motor_bias"] = calibrate_motor_center(card, channels)
    else:
        card.read_encoder(channels["motor_enc_ch"], 1, channels["motor_enc_val"])
        channels["motor_bias"] = channels["motor_enc_val"][0]

    # 모터 보정으로 흔들렸을 수 있으니 펜듈럼 정착 대기
    time.sleep(settle_time)

    # 펜듈럼 영점: 여러 번 읽어 평균 (노이즈/미세진동 완화)
    pend_samples = []
    for _ in range(50):
        card.read_encoder(channels["pendulum_enc_ch"], 1, channels["pendulum_enc_val"])
        pend_samples.append(channels["pendulum_enc_val"][0])
        time.sleep(0.005)
    channels["pendulum_bias"] = sum(pend_samples) // len(pend_samples)
    channels["pendulum_spin_ref"] = channels["pendulum_bias"]   # 초기 기준 = 바닥

    print(f"[INIT] 영점 보정 완료 — motor_bias={channels['motor_bias']}, "
          f"pendulum_bias={channels['pendulum_bias']}")
    print("[INIT] (모터=자동중앙보정, 펜듈럼=아래 매달림 평균)")

    return card, channels


def read_state(card, channels):
    """bias 보정된 (motor_angle, pendulum_angle, motor_vel, pendulum_vel)"""
    card.read_encoder(channels["motor_enc_ch"],     1, channels["motor_enc_val"])
    card.read_encoder(channels["pendulum_enc_ch"],  1, channels["pendulum_enc_val"])
    card.read_other(channels["tach_motor_ch"],      1, channels["tach_motor_val"])
    card.read_other(channels["tach_pend_ch"],       1, channels["tach_pend_val"])

    # bias 보정 후 각도 변환
    motor_count = channels["motor_enc_val"][0]    - channels["motor_bias"]
    pend_count  = channels["pendulum_enc_val"][0] - channels["pendulum_bias"]

    motor_angle    = motor_count / COUNTS_PER_RAD
    pendulum_angle = wrap_angle(pend_count / COUNTS_PER_RAD)
    motor_vel      = channels["tach_motor_val"][0] * TACH_COUNTS_TO_RAD_PER_SEC
    pendulum_vel   = channels["tach_pend_val"][0]  * TACH_COUNTS_TO_RAD_PER_SEC

    return motor_angle, pendulum_angle, motor_vel, pendulum_vel


def pendulum_spin_num(channels) -> float:
    """
    이번 에피소드에서 펜듈럼이 몇 바퀴 돌았는지(부호 포함).
    기준 = pendulum_spin_ref (매 reset 바닥에서 갱신) → 에피소드마다 0부터 시작.
    """
    pend_count = channels["pendulum_enc_val"][0] - channels["pendulum_spin_ref"]
    return pend_count / COUNTS_PER_REV


def check_fault(card, channels) -> tuple[bool, str]:
    stall_ch  = array('I', [1, 2])
    stall_val = array('i', [0, 0])
    card.read_digital(stall_ch, 2, stall_val)

    if stall_val[1]:
        return True, "Motor stall error"
    if stall_val[0]:
        return True, "Motor stall detected"
    return False, ""


def write_pwm(card, channels, pwm: float):
    pwm = max(-0.625, min(0.625, pwm))
    card.write_pwm(channels["pwm_ch"], 1, array('d', [pwm]))


# LED (베이스 RGB) 
# Qube 베이스의 RGB LED. write_other 채널 11000/11001/11002 = R/G/B (0.0~1.0)
# 제어 단계 시각화용: Reset=파랑, Swing-up=노랑, Balancing=초록
LED_CHANNELS  = array('I', [11000, 11001, 11002])
LED_RESET     = (0.0, 0.0, 1.0)   # 파랑
LED_SWINGUP   = (1.0, 1.0, 0.0)   # 노랑
LED_BALANCE   = (0.0, 1.0, 0.0)   # 초록


def set_led(card, r: float, g: float, b: float) -> None:
    """RGB LED 설정. 하드웨어가 지원 안 하면 조용히 무시(제어엔 영향 없음)."""
    try:
        card.write_other(LED_CHANNELS, 3, array('d', [float(r), float(g), float(b)]))
    except Exception:
        pass


def apply_deadband(pwm: float) -> float:
    if pwm == 0.0:
        return 0.0
    sign = 1.0 if pwm > 0 else -1.0
    return sign * max(abs(pwm), DEADBAND_PWM)


def reenable_amp(card):
    card.write_digital(array('I', [0]), 1, array('I', [0]))
    time.sleep(0.1)
    card.write_digital(array('I', [0]), 1, array('I', [1]))
    time.sleep(0.1)


def bleed_pendulum_energy(card, channels, verbose: bool = False) -> None:
    """
    PHASE 0: 센터링 전에 펜듈럼 회전 에너지 빼기.
    모터를 PWM=0 으로 풀어두고 펜듈럼이 충분히 느려질 때까지(또는 타임아웃까지) 대기.
    - 절대 실패를 반환하지 않음: 못 멈춰도 그냥 진행 (이후 센터링이 처리)
    - 모터가 reset 한계를 넘어가면 조기 종료 (의미 없는 대기 방지)
    """
    write_pwm(card, channels, 0.0)
    bleed_start = time.time()

    while time.time() - bleed_start < BLEED_TIMEOUT:
        motor_angle, _, _, pend_vel = read_state(card, channels)

        if abs(pend_vel) < BLEED_VEL_THRESHOLD:
            if verbose:
                print(f"[BLEED] 펜듈럼 진정됨 (pend_vel={pend_vel:+.2f}), "
                      f"{time.time() - bleed_start:.1f}s 소요")
            return

        # 모터가 이미 하드 한계 너머면 더 기다려도 소용 없음 → 바로 센터링 단계로
        if abs(motor_angle) > MOTOR_RESET_LIMIT:
            if verbose:
                print(f"[BLEED] motor({motor_angle:+.2f}) 한계 초과 — 블리딩 중단")
            return

        time.sleep(CONTROL_TS)

    if verbose:
        _, _, _, pv = read_state(card, channels)
        print(f"[BLEED] 타임아웃({BLEED_TIMEOUT}s) — pend_vel={pv:+.2f} 인 채로 센터링 진행")


def reset_motor(card, channels, verbose: bool = False) -> bool:
    """motor angle을 0(bias 기준 중앙)으로 복귀."""
    set_led(card, *LED_RESET)   # reset 중 = 파랑
    reenable_amp(card)

    # ── PHASE 0: 펜듈럼 에너지 빼기 (센터링 성공률 ↑) ──
    bleed_pendulum_energy(card, channels, verbose=verbose)

    # ── PHASE 1: 모터 센터링 (P 제어) ──
    start_time = time.time()
    stuck_counter = 0
    stall_retry = 0
    last_angle = None

    while True:
        elapsed = time.time() - start_time

        if elapsed > RESET_TIMEOUT:
            write_pwm(card, channels, 0.0)
            if verbose:
                print(f"[RESET] Timeout ({RESET_TIMEOUT}s)")
            return False

        has_fault, msg = check_fault(card, channels)
        if has_fault:
            write_pwm(card, channels, 0.0)
            reenable_amp(card)
            time.sleep(0.3)
            stall_retry += 1
            if stall_retry > STALL_MAX_RETRY:
                if verbose:
                    print(f"[RESET] stall {STALL_MAX_RETRY}회 초과 — 포기")
                return False
            # 재시도 (계속 루프)
            continue

        motor_angle, _, _, _ = read_state(card, channels)

        if abs(motor_angle) > MOTOR_RESET_LIMIT:
            write_pwm(card, channels, 0.0)
            if verbose:
                print(f"[RESET] motor({motor_angle:.3f}) exceeds hw limit")
            return False

        error = 0.0 - motor_angle

        if abs(error) < RESET_THRESHOLD:
            write_pwm(card, channels, 0.0)
            break

        # ── 한계 근처면 강하게 밀어 빼냄, 아니면 부드러운 P제어 ──
        if motor_angle > RESET_STRONG_PUSH_RAD:
            pwm = -RESET_STRONG_PUSH_PWM
        elif motor_angle < -RESET_STRONG_PUSH_RAD:
            pwm = RESET_STRONG_PUSH_PWM
        else:
            pwm = RESET_KP * error
            pwm = max(-RESET_MAX_PWM, min(RESET_MAX_PWM, pwm))
            pwm = apply_deadband(pwm)
        write_pwm(card, channels, pwm)

        if last_angle is not None and abs(motor_angle - last_angle) < 0.001:
            stuck_counter += 1
        else:
            stuck_counter = 0
        last_angle = motor_angle

        if stuck_counter > 150:
            if verbose:
                print(f"[RESET] stuck at {motor_angle:+.4f} rad")
            write_pwm(card, channels, 0.0)
            return False

        if verbose:
            print(f"[RESET] motor={motor_angle:+.4f}  error={error:+.4f}  pwm={pwm:+.4f}")

        time.sleep(CONTROL_TS)

    # PHASE 2: settling (펜듈럼이 연속으로 정지할 때까지 대기) 
    # 한 순간 느림(회전 중/turning point 일시적)이 아니라 '연속 정지'를 요구.
    # 펜듈럼 속도는 swing turning point 마다 0을 지나므로, 연속 N회로 진짜 정지만 인정.
    settle_start = time.time()
    still_count = 0
    settled = False
    while time.time() - settle_start < SETTLE_TIMEOUT:
        _, _, _, pend_vel = read_state(card, channels)
        if abs(pend_vel) < SETTLE_VEL_THRESHOLD:
            still_count += 1
            if still_count >= 30:        # 연속 30회(~0.3s) 정지 확인 → 진짜 멈춤
                settled = True
                break
        else:
            still_count = 0
        time.sleep(CONTROL_TS)

    if verbose:
        waited = time.time() - settle_start
        if settled:
            print(f"[SETTLE] 펜듈럼 정지 확인 ({waited:.1f}s 대기)")
        else:
            print(f"[SETTLE] 타임아웃({SETTLE_TIMEOUT}s) — 아직 흔들림. 재보정 건너뛸 수 있음")

    # PHASE 3: 펜듈럼 영점 재보정 (드리프트 누적 차단)
    # 펜듈럼이 아래(6시)에 매달려 멈춘 지금이 재보정 타이밍.
    # (회전 중/바닥 아님이면 함수 내부 안전장치가 갱신을 건너뜀)
    recalibrate_pendulum_zero(card, channels, verbose=verbose)

    return True


def recalibrate_pendulum_zero(card, channels, samples: int = 100,
                              max_drift_deg: float = 15.0, verbose: bool = False) -> None:
    """
    펜듈럼이 아래(6시)에 정지한 상태에서 영점을 '안전하게' 갱신.

    핵심 안전장치:
      - 회전수(N바퀴) 제거: 현재 bias 대비 위상차만 [-1024,1024] counts 로 환산.
        (펜듈럼이 9바퀴 돌아 raw 가 +18000 이어도, 바닥이면 위상차는 ~0)
      - 그 위상차(=실제 드리프트)가 max_drift_deg 이내일 때만 갱신.
        위상차가 크면 = 펜듈럼이 바닥에 정지하지 않음(회전 중/엉뚱한 위치) → 건너뜀.
    → bias 는 항상 초기값 근처에 머물며 '천천히 쌓이는 드리프트'만 따라감.
      (이전 버그: raw 를 그대로 bias 로 써서 회전 시 18000+ 로 튀어 좌표계 붕괴)
    """
    # ── 회전수 종료 기준(spin_ref)을 항상 먼저 갱신 → spin_num 이 다음 에피소드에 0부터 ──
    #   (bias 갱신을 건너뛰더라도 spin_ref 는 갱신해야 누적 raw 가 초기화됨)
    card.read_encoder(channels["pendulum_enc_ch"], 1, channels["pendulum_enc_val"])
    channels["pendulum_spin_ref"] = channels["pendulum_enc_val"][0]

    # ── 펜듈럼이 아직 움직이면 bias(영점) 갱신은 보류 (정지 상태에서만 신뢰) ──
    _, _, _, pend_vel_now = read_state(card, channels)
    if abs(pend_vel_now) > SETTLE_VEL_THRESHOLD:
        if verbose:
            print(f"[RECAL] 영점 갱신 보류 — 펜듈럼 아직 움직임 (pend_vel={pend_vel_now:+.2f})")
        return

    sample_list = []
    for _ in range(samples):
        card.read_encoder(channels["pendulum_enc_ch"], 1, channels["pendulum_enc_val"])
        sample_list.append(channels["pendulum_enc_val"][0])
        time.sleep(0.003)
    raw_mean = sum(sample_list) // len(sample_list)
    old_bias = channels["pendulum_bias"]

    # 회전수 제거: bias 대비 위상차를 [-1024, 1024] 로
    increment = (raw_mean - old_bias) % COUNTS_PER_REV
    if increment > COUNTS_PER_REV // 2:
        increment -= COUNTS_PER_REV
    drift_deg = abs(increment) * 360.0 / COUNTS_PER_REV

    if drift_deg > max_drift_deg:
        # 바닥에 정지하지 않음 → 갱신하지 않고 기존 bias 유지 (좌표계 보호)
        if verbose:
            print(f"[RECAL] 건너뜀 — 바닥 아님 (위상차 {drift_deg:.1f}° > {max_drift_deg}°)")
        return

    channels["pendulum_bias"] = old_bias + increment
    if verbose:
        print(f"[RECAL] 펜듈럼 영점 갱신: {old_bias} → {channels['pendulum_bias']} "
              f"(drift {increment:+d} counts, {drift_deg:.2f}°)")


def close_hardware(card, channels):
    write_pwm(card, channels, 0.0)
    card.write_digital(array('I', [0]), 1, array('I', [0]))
    card.close()
    print("[HW] Hardware connection closed.")


if __name__ == "__main__":
    print("Reset test...")
    print("→ 모터를 중앙에, 펜듈럼을 아래로 한 상태에서 시작하세요!\n")
    card, channels = init_hardware()
    try:
        m_ang, p_ang, m_vel, p_vel = read_state(card, channels)
        print(f"[State] motor={m_ang:+.4f} rad ({math.degrees(m_ang):+.1f}°), "
              f"pend={p_ang:+.4f} rad ({math.degrees(p_ang):+.1f}°)")
        print("→ 둘 다 0 근처여야 정상입니다.\n")
        result = reset_motor(card, channels, verbose=True)
        print("Result:", "OK" if result else "FAIL")
    finally:
        close_hardware(card, channels)
