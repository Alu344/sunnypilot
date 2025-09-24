"""
Copyright (c) 2025, Rick Lan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, and/or sublicense, 
for non-commercial purposes only, subject to the following conditions:

- The above copyright notice and this permission notice shall be included in 
  all copies or substantial portions of the Software.
- Commercial use (e.g. use in a product, service, or activity intended to 
  generate revenue) is prohibited without explicit written permission from 
  the copyright holder.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import numpy as np

SLOPE = -0.04
RATIO = 0.9

TTC = 3.5
TTC_BP = [TTC, 2.5]                # Time-to-collision breakpoints
MIN_BRAKE_ALLOW_VALS = [0., -0.5]  # Allowed braking at each TTC point

class ACM:
  def __init__(self):
    self.enabled = False
    self.downhill_only = False
    self.suppress_accel = True  # toggle: suppress throttle (full coasting)

    self._is_downhill = False
    self._is_speed_over_cruise = False
    self._has_lead = False
    self._active_prev = False

    self.active = False
    self.just_enabled = False    # NEW
    self.just_disabled = False
    self.allowed_brake_val = 0.
    self.lead_ttc = float('inf')

  def update_states(self, cs, rs, user_ctrl_lon, v_ego, v_cruise):
    self.lead_ttc = float('inf')

    if not self.enabled:
      self.active = False
      self.just_enabled = False
      self.just_disabled = False
      return

    if len(cs.orientationNED) != 3:
      self.active = False
      self.just_enabled = False
      self.just_disabled = False
      return

    pitch_rad = cs.orientationNED[1]
    self._is_downhill = np.sin(pitch_rad) < SLOPE
    self._is_speed_over_cruise = v_ego > (v_cruise * RATIO)

    lead = rs.leadOne
    if lead and lead.status:
      # More accurate TTC if relative velocity available
      if hasattr(lead, "vRel") and lead.vRel < 0:  # approaching lead
        self.lead_ttc = lead.dRel / (-lead.vRel) if lead.vRel != 0 else float('inf')
      else:
        self.lead_ttc = float('inf')

      self._has_lead = self.lead_ttc < TTC
    else:
      self._has_lead = False

    # Determine active state
    self.active = (
      not user_ctrl_lon
      and not self._has_lead
      and self._is_speed_over_cruise
      and (self._is_downhill if self.downhill_only else True)
    )

    # Edge detection
    self.just_enabled = not self._active_prev and self.active
    self.just_disabled = self._active_prev and not self.active
    self._active_prev = self.active

    # Interpolate allowed braking based on TTC
    self.allowed_brake_val = float(
      np.interp(self.lead_ttc, TTC_BP, MIN_BRAKE_ALLOW_VALS)
    )

  def update_a_desired_trajectory(self, a_desired_trajectory):
    if not self.active:
      return a_desired_trajectory

    for i in range(len(a_desired_trajectory)):
      # Suppress mild braking
      if a_desired_trajectory[i] < 0 and a_desired_trajectory[i] > self.allowed_brake_val:
        a_desired_trajectory[i] = 0.0

      # Suppress acceleration if enabled
      if self.suppress_accel and a_desired_trajectory[i] > 0:
        a_desired_trajectory[i] = 0.0

    return a_desired_trajectory

  def update_output_a_target(self, output_a_target):
    if not self.active:
      return output_a_target

    # Suppress mild braking
    if output_a_target < 0 and output_a_target > self.allowed_brake_val:
      output_a_target = 0.0

    # Suppress acceleration if enabled
    if self.suppress_accel and output_a_target > 0:
      output_a_target = 0.0

    return output_a_target
