"""Random-pulse plugin for the 590 nm optogenetic LED (DAC1).

Every pulse has a random duration and a random gap, drawn fresh each time.
The buzzer on DAC0 is held at 4.5 V (quiet) the whole time.

Parameters at the top of the file. Runs while Record is on; everything is
off (and the buzzer quiet) before that and after Stop.
"""

import random

# --- parameters -------------------------------------------------------------
LIGHT_V = 4.5          # LED voltage during a pulse
QUIET_V = 4.5          # buzzer voltage that keeps it silent
PULSE_MIN_S = 0.5      # shortest light pulse
PULSE_MAX_S = 2.0      # longest light pulse
GAP_MIN_S = 2.0        # shortest dark gap between pulses
GAP_MAX_S = 10.0       # longest dark gap between pulses

# On stop/failure the app holds these: buzzer quiet (4.5 V), LED off (0 V).
idle_voltage = (QUIET_V, 0.0)


class Controller:

    def setup(self, scope):
        self.next_event_t = None   # wall_time of the next pulse start/end
        self.light_on = False
        scope.set_voltage(QUIET_V, channel=0)   # buzzer quiet
        scope.light_off(channel=1)               # LED off
        scope.print('random pulser started, waiting for Record')

    def update(self, state, scope):
        # Buzzer stays quiet at all times.
        scope.set_voltage(QUIET_V, channel=0)

        if not state.is_recording:
            self.next_event_t = None
            self.light_on = False
            scope.light_off(channel=1)
            return

        now = state.wall_time
        if self.next_event_t is None:
            # Schedule the first pulse a random gap after Record starts.
            self.next_event_t = now + random.uniform(GAP_MIN_S, GAP_MAX_S)

        if now >= self.next_event_t:
            if not self.light_on:
                # Start a pulse of random length.
                self.light_on = True
                duration = random.uniform(PULSE_MIN_S, PULSE_MAX_S)
                self.next_event_t = now + duration
                scope.set_voltage(LIGHT_V, channel=1)
                scope.log(event='pulse_on', duration_s=round(duration, 3))
                scope.print(f'light ON for {duration:.2f} s')
            else:
                # End the pulse, then wait a random gap.
                self.light_on = False
                gap = random.uniform(GAP_MIN_S, GAP_MAX_S)
                self.next_event_t = now + gap
                scope.light_off(channel=1)
                scope.log(event='pulse_off', gap_s=round(gap, 3))
                scope.print(f'light OFF, next pulse in {gap:.2f} s')

    def teardown(self, scope):
        scope.print('random pulser stopped')
