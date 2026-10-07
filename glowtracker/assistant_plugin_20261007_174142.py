"""Sinusoidal voltage on DAC0, one full cycle per minute.

The output follows  v(t) = 2.475 + 2.475 * sin(2*pi*t/60)  volts,
i.e. it swings smoothly between 0 V and 4.95 V with a 60 s period.
DAC1 is left at 0 V. The clock is the acquisition time (state.time_s),
so the phase is tied to when the live view / recording started.
"""

import math


class Controller:

    def setup(self, scope):
        self.period_s = 60.0        # one full sine cycle per minute
        self.v_min = 0.0            # volts
        self.v_max = 4.95           # volts
        self.v_mid = (self.v_min + self.v_max) / 2.0
        self.v_amp = (self.v_max - self.v_min) / 2.0
        scope.print('sine plugin started: DAC0 = %.2f + %.2f*sin(2*pi*t/%.0f s)'
                    % (self.v_mid, self.v_amp, self.period_s))

    def update(self, state, scope):
        phase = 2.0 * math.pi * state.time_s / self.period_s
        v = self.v_mid + self.v_amp * math.sin(phase)
        scope.set_voltage(v, channel=0)   # DAC0 only; DAC1 stays at 0 V

    def teardown(self, scope):
        scope.print('sine plugin stopped')
