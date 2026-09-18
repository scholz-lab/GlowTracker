# GlowTracker plugins

A plugin is one Python file that GlowTracker loads and calls once per camera frame. It sees where
the tracked animal is and can drive the optogenetic LED through the DAQ. You never edit GlowTracker
itself.

## Quick start

1. Copy `template_controller.py` somewhere and edit `update`.
2. In GlowTracker: connect the camera, stage and DAQ. Start **Live view**, then **Tracking** on the
   animal.
3. Open **DAQ** (right column) and pick the **Plugin** tab. **Browse** to your file, press **Start**.
   The DAQ mode switches to `Plugin`; the built-in Sequencer / Stage program / Reversal modes are
   off while a plugin runs.
4. Edit your file, press **Reload**. Press **Stop** to end. Stopping, errors and closing the app all
   switch the light off.

The plugin keeps running after the DAQ popup is closed. Without a DAQ connected everything still
runs as a dry run and `state.voltage` shows what you requested.

## Plugin file

Either a class

```python
class Controller:
    def setup(self, scope): ...           # optional, once on Start
    def update(self, state, scope): ...   # required, once per frame
    def teardown(self, scope): ...        # optional, once on Stop
```

or a plain function `def update(state, scope): ...`.

`update` runs on its own thread. If it takes longer than a frame, frames are skipped, not queued
(you always get the latest state). An exception stops the plugin and shows the traceback in the
Plugin tab.

## `state` fields

All positions are stage coordinates in millimetres. X and Y follow the stage axes.

| field | type | meaning |
|---|---|---|
| `frame` | int | frame counter of the current live view / recording |
| `time_s` | float | seconds since the acquisition started |
| `wall_time` | float | `time.time()` when the snapshot was made |
| `stage_xy` | (x, y) | stage position |
| `worm_xy` | (x, y) | animal position = stage + centroid offset. Equals `stage_xy` when not tracking |
| `cms_offset_px` | (x, y) | centroid offset from image centre in pixels, y up; (0, 0) when not tracking |
| `trail` | N x 2 array | recent stage positions, oldest first. Length is the DAQ > Reversal "trail limit" |
| `velocity` | (dx, dy) | last trail step in mm per tracking step |
| `speed` | float | length of `velocity` |
| `is_reversing` | bool | verdict of the built-in reversal detector, tuned in DAQ > Reversal |
| `is_tracking` | bool | |
| `is_recording` | bool | |
| `voltage` | float | DAQ voltage currently applied |
| `image_shape` | tuple | shape of the latest frame |

Notes: while tracking, the stage position is estimated from the commanded moves, so `trail` is
sampled once per tracking step, not once per frame. The reversal detector needs an animal length
and trail limit set in the DAQ > Reversal tab (defaults are loaded from the config).

## `scope` methods

| call | effect |
|---|---|
| `set_voltage(v)` | drive both DAC outputs, clamped to 0 .. 4.95 V; returns the applied value. Unchanged values are not re-sent |
| `light_off()` | same as `set_voltage(0)` |
| `voltage`, `daq_connected` | properties |
| `get_frame(copy=True)` | latest camera frame as a numpy array (`uint8`, 2-D; main side in dual-colour mode) |
| `get_position()` | `(x, y, z)` stage position in mm from the position poller, or `None` |
| `move_rel(dx, dy, dz=0)` | relative move in mm, waits until idle. Returns `False` and is refused while tracking, a plate run or a Go To move is active |
| `move_abs(x, y, z=None)` | absolute move in mm, same rules; `z` defaults to the current Z |
| `start_recording()`, `stop_recording()` | toggle the Record button (asynchronous) |
| `log(**fields)` | append a JSON line to `plugin_log_<time>.jsonl` in the recording folder. numpy values are converted |
| `print(*args)` | show a message in the Plugin tab status line and the console |
| `is_stopping` | `True` once Stop was pressed; long loops should check it |

The recording's coordinate file also logs the DAQ voltage for every frame, so the light state is
always in the data.

## Rules of thumb

- Keep `update` short. Anything heavier than a few array operations on the frame should be done
  every N-th frame.
- Let GlowTracker track. Your job is to decide the light; the app keeps the animal centred.
- Use `scope.log` generously, it is what you will analyse afterwards.
- Test with the DAQ disconnected first (dry run), then connect it.

`blink_and_log_controller.py` is the smallest complete example: it blinks the light on a fixed
schedule and logs the animal position every frame.
