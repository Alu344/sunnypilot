import numpy as np

# --- Hysteresis & thresholds ---
# 下坡判定：sin(pitch) < SLOPE，加入遲滯避免抖動
SLOPE = -0.04         # 約 -2.3°
SLOPE_HYST = 0.01     # 約 ±0.6° 的 sin 範圍

# 超過巡航判定：ON/OFF 分開，降低邊界抖動
RATIO_ON = 0.90
RATIO_OFF = 0.88

# 啟動速度門檻（避免超低速啟用 ACM）
MIN_SPEED_ON_MPS = 1.4   # ≈ 5 km/h
MIN_SPEED_OFF_MPS = 1.2  # ≈ 4.3 km/h（遲滯）

# 允許煞車量與 TTC 映射（x 軸需遞增）
TTC = 3.5
TTC_BP = [2.5, TTC]               # 2.5s → 3.5s
MIN_BRAKE_ALLOW_VALS = [-0.5, 0.] # 小 TTC 允許更多（更負）的煞車；大 TTC 允許 0

# TTC 去抖（需要連續幾幀才算有/沒有前車威脅）
TTC_ON_FRAMES = 3   # 連續 < TTC 幀數 → 觸發有威脅
TTC_OFF_FRAMES = 6  # 連續 ≥ TTC 幀數 → 解除威脅

class ACM:
  def __init__(self):
    self.enabled = False
    self.downhill_only = False
    self.suppress_accel = True  # True: 抑制油門（全滑行）

    self._is_downhill = False
    self._is_speed_over_cruise = False
    self._has_lead = False
    self._active_prev = False

    # TTC 去抖用
    self._ttc_on_ctr = 0
    self._ttc_off_ctr = 0

    self.active = False
    self.just_enabled = False
    self.just_disabled = False
    self.allowed_brake_val = 0.0
    self.lead_ttc = float('inf')

  def _reset_edge_flags(self):
    self.just_enabled = False
    self.just_disabled = False

  def update_states(self, cs, rs, user_ctrl_lon, v_ego, v_cruise):
    self.lead_ttc = float('inf')
    self._reset_edge_flags()

    # 未啟用 → 全關
    if not self.enabled:
      self.active = False
      self._active_prev = False
      return

    # 缺 orientation → 全關（無法判斷坡度）
    if not hasattr(cs, "orientationNED") or len(cs.orientationNED) != 3:
      self.active = False
      self._active_prev = False
      return

    # --- Downhill hysteresis ---
    pitch_rad = cs.orientationNED[1]
    s = float(np.sin(pitch_rad))
    if s < (SLOPE - SLOPE_HYST):
      self._is_downhill = True
    elif s > (SLOPE + SLOPE_HYST):
      self._is_downhill = False

    # --- Over-cruise hysteresis ---
    if v_ego > (v_cruise * RATIO_ON):
      self._is_speed_over_cruise = True
    elif v_ego < (v_cruise * RATIO_OFF):
      self._is_speed_over_cruise = False

    # --- Lead TTC with debouncing ---
    lead = getattr(rs, "leadOne", None) if rs is not None else None
    has_lead_raw = False
    if lead is not None and getattr(lead, "status", False):
      v_rel = getattr(lead, "vRel", None)
      d_rel = max(0.0, float(getattr(lead, "dRel", 0.0)))
      if v_rel is not None and v_rel < 0.0:  # 正在接近
        eps = 1e-3
        self.lead_ttc = (d_rel / (-v_rel)) if abs(v_rel) > eps else float('inf')
      else:
        self.lead_ttc = float('inf')
      has_lead_raw = (self.lead_ttc < TTC)
    else:
      self.lead_ttc = float('inf')
      has_lead_raw = False

    # 去抖：連續幀判斷
    if has_lead_raw:
      self._ttc_on_ctr = min(self._ttc_on_ctr + 1, TTC_ON_FRAMES)
      self._ttc_off_ctr = 0
      if self._ttc_on_ctr >= TTC_ON_FRAMES:
        self._has_lead = True
    else:
      self._ttc_off_ctr = min(self._ttc_off_ctr + 1, TTC_OFF_FRAMES)
      self._ttc_on_ctr = 0
      if self._ttc_off_ctr >= TTC_OFF_FRAMES:
        self._has_lead = False

    # 允許的（不被抑制的）最大煞車（非正值）
    self.allowed_brake_val = float(np.interp(self.lead_ttc, TTC_BP, MIN_BRAKE_ALLOW_VALS))
    self.allowed_brake_val = min(0.0, self.allowed_brake_val)

    # --- Min speed hysteresis for ACM activation ---
    above_min_speed = (v_ego > MIN_SPEED_ON_MPS) if not self.active else (v_ego > MIN_SPEED_OFF_MPS)

    # 判斷是否進入 ACM
    self.active = (
      not user_ctrl_lon
      and above_min_speed
      and not self._has_lead
      and self._is_speed_over_cruise
      and (self._is_downhill if self.downhill_only else True)
    )

    # Edge detection
    if not self._active_prev and self.active:
      self.just_enabled = True
    elif self._active_prev and not self.active:
      self.just_disabled = True
    self._active_prev = self.active

  def update_a_desired_trajectory(self, a_desired_trajectory):
    if not self.active:
      return a_desired_trajectory

    arr = np.array(a_desired_trajectory, dtype=float, copy=True)
    # 抑制輕微煞車（介於 allowed_brake_val ~ 0）
    mask_soft_brake = (arr < 0.0) & (arr > self.allowed_brake_val)
    arr[mask_soft_brake] = 0.0
    # 抑制加速（若開啟）
    if self.suppress_accel:
      arr[arr > 0.0] = 0.0
    return arr.tolist()

  def update_output_a_target(self, output_a_target):
    if not self.active:
      return output_a_target
    # 抑制輕微煞車
    if output_a_target < 0.0 and output_a_target > self.allowed_brake_val:
      output_a_target = 0.0
    # 抑制加速（若開啟）
    if self.suppress_accel and output_a_target > 0.0:
      output_a_target = 0.0
    return output_a_target
