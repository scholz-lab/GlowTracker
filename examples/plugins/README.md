# GlowTracker plugins

A plugin is one Python file that GlowTracker loads and calls once per camera frame. 

## Quick start

1. Copy `template_controller.py` somewhere and edit `update`.
2. In GlowTracker: connect the camera, stage and DAQ. Start **Live view**, then **Tracking** on the animal.
3. Open **DAQ**  and pick the **Plugin** tab. **Browse** to your file, press **Start**.
4. Edit your file, press **Reload**. Press **Stop** to end. Stopping, errors and closing the app all switch the light off.

The plugin keeps running after the DAQ popup is closed. 

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
| `fps` | float | measured camera frame rate over the last ~30 frames, 0 until known |
| `frame_period_s` | float | `1 / fps` |
| `frames_for(seconds)` | int | how many frames span a duration at the current rate, e.g. `state.frames_for(0.5)` for a half-second pulse |
| `analysis` | `BrightnessStats` or `None` | the app's live-analysis image statistics for this frame: `min`, `max`, `mean`, `median`, `skewness`, `percentile_5`, `percentile_95`. `None` unless the app is computing them (see below) |

`analysis` is filled only when GlowTracker computes the live analysis. Turn on **Show live analysis**
in the settings for live view, and **Save analysis to recording** if you also want it while
tracking or recording. The region (whole image or the tracking window) follows the Live analysis
"region mode" setting. Check for `None`:

```python
if state.analysis is not None and state.analysis.mean > 40:
    ...
```

While tracking, the stage position is estimated from the commanded moves, so `trail` is
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
| `is_recording` | property, current Record button state |
| `wait_for_recording(recording=True, timeout=None)` | block until recording is on (or off with `recording=False`). Returns `False` on timeout or Stop. Use it in `setup` to hold the light logic until you press Record |
| `log(**fields)` | append a JSON line to `plugin_log_<time>.jsonl` in the recording folder. numpy values are converted |
| `print(*args)` | show a message in the Plugin tab status line and the console |
| `is_stopping` | `True` once Stop was pressed; long loops should check it |
| `fps` | property, same measured frame rate as `state.fps` |

The recording's coordinate file also logs the DAQ voltage for every frame.


