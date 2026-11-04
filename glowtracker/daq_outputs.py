"""The outputs under Tools, while a DAQ is connected: a small tile with the last minute of DAC0 and
DAC1, which opens a panel to set them by hand. The panel floats next to the tile without blocking
the window, so the image stays usable. The look is in layout.kv.

The plot shows what the outputs were set to, by anyone (sequencer, plugin, these sliders). The
DAC outputs cannot be read back, so it is not a measurement.
"""
from __future__ import annotations

import time

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, Line, RoundedRectangle
from kivy.metrics import dp
from kivy.properties import BooleanProperty, NumericProperty, StringProperty
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.widget import Widget

from DAQ_control import MAX_VOLTAGE
from script_preview_ui import dac_labels_from

WINDOW_S = 60.0
REFRESH_S = 0.1
SEND_S = 0.05               # a dragged slider updates the output at most this often
SCALE_V = 5.0
COLOURS = ((0.36, 0.62, 1.0, 1), (0.87, 0.38, 0.58, 1))        # as the timeline preview: DAC0, DAC1


def step_trace(history: list, now: float, seconds: float, channel: int) -> list:
    """One output's changes as a stepped line: [(fraction of the window, volts)], 1.0 being now."""
    start = now - seconds
    index = channel + 1
    earlier = [event for event in history if event[0] <= start]
    level = earlier[-1][index] if earlier else None
    points = [(0.0, level)] if earlier else []
    for event in history[len(earlier):]:
        x = (event[0] - start) / seconds
        if level is not None:
            points.append((x, level))
        level = event[index]
        points.append((x, level))
    if level is not None:
        points.append((1.0, level))
    return points


class VoltagePlot(Widget):
    """DAC0 and DAC1 over the last minute. DAC0 is drawn wider, so it shows behind DAC1 when both
    outputs are at the same voltage. With axes, the scale and time are labelled."""
    axes = BooleanProperty(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        with self.canvas:
            Color(0.075, 0.078, 0.086, 1)
            self._panel = RoundedRectangle(radius=[dp(8)])
            Color(1, 1, 1, 0.07)
            self._grid = [Line(width=1) for _ in range(3)]
            Color(*COLOURS[0])
            self._dac0 = Line(width=dp(1.6))
            Color(*COLOURS[1])
            self._dac1 = Line(width=dp(1))
        self._labels = []
        self._history: list = []
        self._now = 0.0
        self.bind(pos=self.redraw, size=self.redraw, axes=self._makeLabels)
        # And once more after the layout settles: see PadArrow (Windows display-scaling resizes).
        self._redrawSoon = Clock.create_trigger(self.redraw, -1)
        self.bind(pos=self._redrawSoon, size=self._redrawSoon)
        Window.bind(size=self._redrawSoon)

    def _makeLabels(self, *args) -> None:
        for label in self._labels:
            self.remove_widget(label)
        self._labels = []
        if self.axes:
            for text in ('5 V', '2.5', '0', f'-{WINDOW_S:.0f} s', 'now'):
                label = Label(text=text, font_size='10sp', color=(0.5, 0.5, 0.55, 1), size_hint=(None, None))
                label.bind(texture_size=label.setter('size'))
                label.texture_update()
                self.add_widget(label)
                self._labels.append(label)
        self.redraw()

    def show(self, history: list, now: float) -> None:
        self._history, self._now = history, now
        self.redraw()

    def redraw(self, *args) -> None:
        self._panel.pos, self._panel.size = self.pos, self.size
        pad = dp(6)
        left = self.x + (dp(30) if self.axes else pad)
        right = self.right - (dp(10) if self.axes else pad)
        bottom = self.y + (dp(20) if self.axes else pad)
        top = self.top - (dp(10) if self.axes else pad)
        if top <= bottom or right <= left:
            return
        for i, line in enumerate(self._grid):
            y = bottom + (top - bottom) * i / 2
            line.points = [left, y, right, y]
        for channel, line in enumerate((self._dac0, self._dac1)):
            points = []
            for x, volts in step_trace(self._history, self._now, WINDOW_S, channel):
                points += [left + x * (right - left), bottom + min(volts / SCALE_V, 1.0) * (top - bottom)]
            line.points = points
        if self._labels:
            for label, y in zip(self._labels, (top, (top + bottom) / 2, bottom)):
                label.right, label.center_y = left - dp(4), y
            self._labels[3].pos = (left, self.y + dp(3))
            self._labels[4].pos = (right - self._labels[4].width, self.y + dp(3))


class OutputRow(BoxLayout):
    """One output: its name, a slider and the exact voltage. The output changes while the slider
    is dragged, or when a typed value is confirmed with Enter."""
    channel = NumericProperty(0)
    name = StringProperty('')
    maximum = NumericProperty(MAX_VOLTAGE)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dragging = False
        self._sendSoon = Clock.create_trigger(lambda dt: self._send(), SEND_S)

    def on_touch_down(self, touch):
        self.dragging = self.ids.slider.collide_point(*touch.pos) and not self.disabled
        return super().on_touch_down(touch)

    def on_touch_up(self, touch):
        handled = super().on_touch_up(touch)
        if self.dragging:
            self.dragging = False
            self._send(final=True)
        return handled

    def sliderMoved(self) -> None:
        if self.dragging:
            self.ids.field.text = f'{self.ids.slider.value:.2f}'
            self._sendSoon()

    def _send(self, final: bool = False) -> None:
        self.parent_panel.setVoltage(self.channel, self.ids.slider.value, final)

    def typed(self) -> None:
        try:
            volts = float(self.ids.field.text.replace(',', '.'))
        except ValueError:
            return                  # the next refresh shows the real value again
        self.parent_panel.setVoltage(self.channel, volts, True)

    def show(self, volts: float, blocked: bool) -> None:
        self.disabled = blocked
        if not self.dragging and not self.ids.field.focus:
            self.ids.slider.value = volts
            self.ids.field.text = f'{volts:.2f}'


class DaqOutputsPanel(BoxLayout):
    """Set the outputs by hand. Added to the Window next to the tile, not as a popup: everything
    around it keeps working."""
    keepsTooltips = True
    hint = StringProperty('')

    def __init__(self, tile: 'DaqOutputsTile', **kwargs):
        super().__init__(**kwargs)
        self.tile = tile
        self.app = App.get_running_app()
        labels = dac_labels_from(self.app.config)
        self.rows = (self.ids.dac0, self.ids.dac1)
        self.bind(size=self.place)      # its height is known only after the first layout
        for channel, row in enumerate(self.rows):
            row.parent_panel = self
            row.name = f'DAC{channel}' + (f'  [color=999999]{labels[channel]}[/color]' if labels[channel] else '')

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            super().on_touch_down(touch)
            return True             # not through to the image underneath
        return False

    def place(self, *args) -> None:
        tile = self.tile
        self.right = tile.x - dp(8)
        self.top = max(min(tile.top, Window.height - dp(8)), self.height + dp(8))

    def show(self, history: list, now: float, voltages: list, blocked: str) -> None:
        self.ids.plot.show(history, now)
        self.hint = blocked
        for row, volts in zip(self.rows, voltages):
            row.show(volts, bool(blocked))

    def setVoltage(self, channel: int | None, volts: float, final: bool) -> None:
        if self.tile.blockedReason():
            return
        try:
            applied = self.app.daqControl.set_voltage(volts, channel)
        except Exception as e:
            self.app.log('daq', f'could not set the output: {e}', 'error')
            return
        if final:
            which = 'both outputs' if channel is None else f'DAC{channel}'
            self.app.log('daq', f'{which} set by hand to {applied:.2f} V')


class DaqOutputsTile(ButtonBehavior, BoxLayout):
    """Under Tools, only while a DAQ is connected. Click it to set the outputs by hand."""
    connected = BooleanProperty(False)
    values = StringProperty('')
    tooltip = 'What the DAQ outputs were set to in the last minute. Click to set them by hand.'

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.panel: DaqOutputsPanel | None = None
        Clock.schedule_interval(self._refresh, REFRESH_S)
        Window.bind(size=self._place)
        self.bind(pos=self._place, size=self._place)

    def _place(self, *args) -> None:
        if self.panel is not None:
            self.panel.place()

    def blockedReason(self) -> str:
        app = App.get_running_app()
        try:
            recording = app.root.ids.middlecolumn.runtimecontrols.imageacquisitionmanager.recordbutton.state == 'down'
        except (AttributeError, KeyError):
            recording = False
        host = getattr(app, 'pluginHost', None)
        return app.daqControl.manualBlockedReason(recording, host is not None and host.is_running())

    def _refresh(self, dt) -> None:
        daq = getattr(App.get_running_app(), 'daqControl', None)
        self.connected = daq is not None and daq.isConnected()
        if not self.connected:
            self.closePanel()
            return
        history, now, voltages = daq.voltageHistory(), time.monotonic(), daq.channelVoltages
        self.values = '   '.join(f'[color={"5c9eff" if i == 0 else "de6194"}]{v:.2f} V[/color]'
                                 for i, v in enumerate(voltages))
        self.ids.plot.show(history, now)
        if self.panel is not None:
            self.panel.show(history, now, voltages, self.blockedReason())

    def on_release(self) -> None:
        if self.panel is not None:
            self.closePanel()
            return
        self.panel = DaqOutputsPanel(self)
        Window.add_widget(self.panel)
        self.panel.place()

    def closePanel(self) -> None:
        if self.panel is not None:
            Window.remove_widget(self.panel)
            self.panel = None
