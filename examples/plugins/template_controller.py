"""Minimal GlowTracker plugin. Copy this file, edit `update`, load it in DAQ > Plugin > Start.

The app calls `update` once per camera frame from its own plugin thread. Keep it fast (well under
one frame period) and never block. Raise an exception and the app switches the light off, shows
the traceback in the Plugin tab and stops the plugin; fix the file and press Reload.

`state` (read-only snapshot, all positions in stage millimetres):
    state.frame          int    frame counter of the current live view / recording
    state.time_s         float  seconds since the acquisition started
    state.wall_time      float  time.time() when the snapshot was made
    state.stage_xy       (x, y) stage position
    state.worm_xy        (x, y) animal position = stage + centroid offset (== stage_xy when not tracking)
    state.cms_offset_px  (x, y) centroid offset from image centre in pixels (y up)
    state.trail          N x 2 numpy array of recent stage positions, oldest first
    state.velocity       (dx, dy) last trail step in mm (per tracking step)
    state.speed          |velocity|
    state.is_reversing   bool   the app's reversal detector verdict (tune it in DAQ > Reversal)
    state.is_tracking    bool
    state.is_recording   bool
    state.voltage        float  DAQ voltage currently applied
    state.image_shape    tuple  shape of the latest frame

`scope` (control handle):
    scope.set_voltage(v)          drive the LED, 0 .. 4.95 V, returns the value applied
    scope.light_off()
    scope.voltage / scope.daq_connected
    scope.get_frame()             latest frame as a numpy array (copy)
    scope.get_position()          (x, y, z) stage position in mm, or None
    scope.move_rel(dx, dy, dz=0)  mm; refused (returns False) while tracking / plate run / Go To
    scope.move_abs(x, y, z=None)  mm; same rules
    scope.start_recording() / scope.stop_recording()
    scope.log(**fields)           one JSON line into plugin_log_<time>.jsonl in the recording folder
    scope.print(*args)            message in the Plugin tab status line
    scope.is_stopping             True once Stop was pressed
"""


class Controller:

    def setup(self, scope):
        """Called once on Start. Put your parameters here."""
        self.pulse_voltage = 3.0
        scope.print('template plugin started')

    def update(self, state, scope):
        """Called once per frame."""
        if not state.is_tracking:
            scope.light_off()
            return

        # Example rule: light on while the animal moves faster than 0.02 mm per step.
        if state.speed > 0.02:
            scope.set_voltage(self.pulse_voltage)
        else:
            scope.light_off()

    def teardown(self, scope):
        """Called once on Stop (the app also switches the light off for you)."""
        scope.print('template plugin stopped')
