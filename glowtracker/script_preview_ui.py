"""The script preview: what a plugin or sequencer script does, in two swappable views.

Timeline: the DAC voltages over a minute of simulated recording (a plugin) or the script's steps
    (a sequencer). For a plugin that reacts to the worm the simulated reversals are shaded and
    the run with a second simulated worm is drawn dashed.
Code: which inputs feed which variables and outputs (tap a box to follow it), and the steps of
    update() and the helpers it calls.

Self-contained (its own colours, no layout.kv classes), so both the main window and the AI
assistant window can use it. The analysis is in script_preview.py.
"""
from __future__ import annotations

import math
from threading import Thread

from kivy.clock import Clock
from kivy.core.text import Label as CoreLabel
from kivy.graphics import Color, Line, Rectangle, RoundedRectangle
from kivy.lang import Builder
from kivy.metrics import dp, sp
from kivy.properties import BooleanProperty, DictProperty, ListProperty, ObjectProperty, StringProperty
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.scrollview import ScrollView
from kivy.uix.widget import Widget
from kivy.utils import escape_markup

import script_preview

TEXT = (0.92, 0.92, 0.94, 1)
MUTED = (0.6, 0.6, 0.65, 1)
FAINT = (1, 1, 1, 0.35)
PANEL = (0.095, 0.1, 0.115, 1)
BLUE = (0.36, 0.64, 1, 1)
PINK = (0.87, 0.38, 0.58, 1)
GREEN = (0.24, 0.86, 0.59, 1)
ORANGE = (1, 0.62, 0.2, 1)
VIOLET = (0.78, 0.57, 0.92, 1)
GREY = (0.55, 0.56, 0.62, 1)
KIND_COLOUR = {'input': BLUE, 'setting': GREY, 'memory': PINK, 'local': (0.75, 0.76, 0.8, 1), 'output': GREEN}
LANE_COLOUR = (BLUE, PINK)
SHOW_CODE_VIEW = False      # the Code view is not finished yet: the preview shows the timeline only

_TEXTURES: dict = {}


def _text(text: str, size: float, colour=TEXT, bold=False, italic=False, mono=False):
    key = (text, size, tuple(colour), bold, italic, mono)
    if key not in _TEXTURES:
        if len(_TEXTURES) > 800:
            _TEXTURES.clear()
        options = dict(text=text, font_size=size, bold=bold, italic=italic, color=colour)
        if mono:
            options['font_name'] = 'RobotoMono-Regular'
        label = CoreLabel(**options)
        label.refresh()
        _TEXTURES[key] = label.texture
    return _TEXTURES[key]


def _blit(texture, x, y):
    Color(1, 1, 1, 1)
    Rectangle(texture=texture, size=texture.size, pos=(x, y))


def _nice_step(span: float, ticks: int = 6) -> float:
    raw = max(span, 1e-9) / ticks
    magnitude = 10 ** math.floor(math.log10(raw))
    return next(m * magnitude for m in (1, 2, 5, 10) if m * magnitude >= raw)


Builder.load_string('''
<PreviewTab>:
    size_hint_x: None
    width: self.texture_size[0] + dp(24)
    font_size: '12.5sp'
    bold: self.selected
    color: (0.92, 0.92, 0.94, 1) if self.selected else (0.6, 0.6, 0.65, 1)
    canvas.before:
        Color:
            rgba: (0.3, 0.31, 0.35, 1) if self.selected else (0, 0, 0, 0)
        RoundedRectangle:
            pos: self.x + dp(2), self.y + dp(2)
            size: self.width - dp(4), self.height - dp(4)
            radius: [dp(6)]

<ScriptPreview>:
    orientation: 'vertical'
    spacing: dp(8) if root.show_tabs else 0
    BoxLayout:
        size_hint_y: None
        height: dp(30) if root.show_tabs else 0
        opacity: 1 if root.show_tabs else 0
        disabled: not root.show_tabs
        spacing: dp(10)
        BoxLayout:
            id: tabs
            size_hint_x: None
            width: self.minimum_width
            padding: dp(2)
            canvas.before:
                Color:
                    rgba: 0.14, 0.15, 0.17, 1
                RoundedRectangle:
                    pos: self.pos
                    size: self.size
                    radius: [dp(8)]
            PreviewTab:
                text: 'Timeline'
                selected: root.view == 'timeline'
                on_release: root.view = 'timeline'
            PreviewTab:
                text: 'Code'
                selected: root.view == 'code'
                on_release: root.view = 'code'
        Widget:
    BoxLayout:
        id: body
''')


class PreviewTab(ButtonBehavior, Label):
    selected = BooleanProperty(False)


# --- timeline -------------------------------------------------------------------------------

class TimelineView(Widget):
    data = DictProperty({})
    dac_labels = ListProperty(['', ''])
    message = StringProperty('')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._trigger = Clock.create_trigger(self.redraw, 0)
        self.bind(pos=self._trigger, size=self._trigger, data=self._trigger, message=self._trigger,
                  dac_labels=self._trigger)

    def redraw(self, *args):
        self.canvas.clear()
        with self.canvas:
            Color(*PANEL)
            RoundedRectangle(pos=self.pos, size=self.size, radius=[dp(10)])
            data = self.data
            if not data or not data.get('ok') or not data.get('runs'):
                texture = _text(self.message or (data.get('error', '') if data else '') or ' ', sp(13), MUTED)
                _blit(texture, self.center_x - texture.width / 2, self.center_y - texture.height / 2)
                return
            self._plot(data)

    def _plot(self, data):
        runs, times, unit = data['runs'], data['t'], data['unit']
        end = float(data.get('seconds') or (times[-1] if times else 1))
        same = all(r['v0'] == r['v1'] for r in runs)
        lanes = [('DAC0 + DAC1', 'v0', BLUE)] if same else [('DAC0', 'v0', BLUE), ('DAC1', 'v1', PINK)]
        left, right, top, bottom = dp(64), dp(16), dp(26), dp(58)
        gap = dp(30)
        x0, x1 = self.x + left, self.right - right
        plot_h = self.height - top - bottom - gap * (len(lanes) - 1)
        lane_h = max(dp(30), plot_h / len(lanes))
        width = max(1.0, x1 - x0)

        def px(t):
            return x0 + width * min(max(t, 0.0), end) / end

        step = _nice_step(end)
        ticks = [i * step for i in range(int(end / step) + 1)]
        for index, (name, key, colour) in enumerate(lanes):
            ly = self.top - top - (index + 1) * lane_h - index * gap
            # reversal spans of the simulated worm, behind
            for a, b in (data.get('spans', []) if data.get('reactive') else []):
                if a < end:
                    Color(*ORANGE[:3], 0.1)
                    Rectangle(pos=(px(a), ly), size=(px(b) - px(a), lane_h))
            # frame, ticks, labels
            Color(*FAINT)
            Line(rectangle=(x0, ly, width, lane_h), width=1)
            for volts in (0, 2.5, 5):
                y = ly + lane_h * volts / 5.0
                Color(*FAINT)
                Line(points=[x0, y, x0 + dp(5), y], width=1)
                Line(points=[x1, y, x1 - dp(5), y], width=1)
                texture = _text(f'{volts:g}', sp(10.5), MUTED)
                _blit(texture, x0 - texture.width - dp(5), y - texture.height / 2)
            for t in ticks:
                Color(*FAINT)
                Line(points=[px(t), ly, px(t), ly + dp(4)], width=1)
                Line(points=[px(t), ly + lane_h, px(t), ly + lane_h - dp(4)], width=1)
            label = f'{name} · {self.dac_labels[index]}' if not same and self.dac_labels[index] else name
            texture = _text(label, sp(11.5), colour, bold=True)
            _blit(texture, x0, ly + lane_h + dp(4))
            texture = _text('V', sp(10.5), MUTED, italic=True)
            _blit(texture, self.x + dp(10), ly + lane_h / 2 - texture.height / 2)
            # traces: the second simulated worm dashed, under the first
            for number, run in reversed(list(enumerate(runs))):
                points = self._steps(times, run[key], px, ly, lane_h, data.get('steps', False))
                if len(points) < 4:
                    continue
                if number == 0:
                    Color(*colour)
                    Line(points=points, width=dp(1.4))
                else:
                    Color(*colour[:3], 0.55)
                    Line(points=points, width=1, dash_length=5, dash_offset=4)
            # events of the first run
            for t, kind, text in runs[0].get('events', []):
                x = px(t)
                if kind == 'record':
                    Color(1, 0.4, 0.4, 0.8)
                    Line(points=[x, ly, x, ly + lane_h], width=1, dash_length=4, dash_offset=3)
                elif kind == 'stage':
                    Color(*ORANGE)
                    Line(points=[x - dp(4), ly + lane_h + dp(1), x, ly + lane_h - dp(6), x + dp(4), ly + lane_h + dp(1)],
                         width=dp(1.1))
        # x axis labels under the last lane
        base = self.top - top - len(lanes) * lane_h - (len(lanes) - 1) * gap
        for t in ticks:
            texture = _text(f'{t:g}', sp(10.5), MUTED)
            _blit(texture, px(t) - texture.width / 2, base - texture.height - dp(4))
        texture = _text('time after Record (s)' if unit == 's' else 'frames after Record', sp(11), TEXT, italic=True)
        _blit(texture, x1 - texture.width, base - texture.height * 2 - dp(8))
        self._legend(data, lanes, base - dp(50))

    def _legend(self, data, lanes, y):
        entries = []
        if data.get('spans') and data.get('reactive'):
            entries.append(('simulated reversal', 'span'))
        if len(data['runs']) > 1 and data.get('reactive'):
            entries.append(('second simulated worm', 'dash'))
        kinds = {e[1] for e in data['runs'][0].get('events', [])}
        if 'stage' in kinds:
            entries.append(('stage move', 'stage'))
        if 'record' in kinds:
            entries.append(('recording start/stop', 'record'))
        x = self.x + dp(64)
        for text, kind in entries:
            if kind == 'span':
                Color(*ORANGE[:3], 0.25)
                Rectangle(pos=(x, y + dp(2)), size=(dp(14), dp(10)))
            elif kind == 'dash':
                Color(*BLUE[:3], 0.7)
                Line(points=[x, y + dp(7), x + dp(14), y + dp(7)], width=1, dash_length=4, dash_offset=2)
            elif kind == 'stage':
                Color(*ORANGE)
                Line(points=[x + dp(3), y + dp(12), x + dp(7), y + dp(4), x + dp(11), y + dp(12)], width=dp(1.1))
            else:
                Color(1, 0.4, 0.4, 0.8)
                Line(points=[x + dp(7), y, x + dp(7), y + dp(14)], width=1, dash_length=3, dash_offset=2)
            texture = _text(text, sp(11), MUTED)
            _blit(texture, x + dp(20), y + dp(7) - texture.height / 2)
            x += dp(20) + texture.width + dp(18)

    @staticmethod
    def _steps(times, values, px, ly, lane_h, already_steps):
        def py(v):
            return ly + lane_h * min(max(v, 0.0), 5.0) / 5.0
        if already_steps:
            points = []
            for t, v in zip(times, values):
                points += [px(t), py(v)]
            return points
        if not values:
            return []
        points = [px(times[0]), py(values[0])]
        for i in range(1, len(values)):
            if values[i] != values[i - 1]:
                points += [px(times[i]), py(values[i - 1]), px(times[i]), py(values[i])]
        end = times[-1] + (times[1] - times[0] if len(times) > 1 else 0)
        points += [px(end), py(values[-1])]
        return points


# --- data-flow graph ------------------------------------------------------------------------

class DataFlowGraph(Widget):
    """Inputs on the left, outputs on the right, the variables between them in layers."""
    graph = DictProperty({})
    selected = StringProperty('')

    def __init__(self, **kwargs):
        super().__init__(size_hint_y=None, **kwargs)
        self._boxes: dict = {}
        self._trigger = Clock.create_trigger(self.redraw, 0)
        self.bind(pos=self._trigger, size=self._trigger, graph=self._layout, selected=self._trigger)

    def _layout(self, *args):
        nodes = self.graph.get('nodes', {})
        edges = [tuple(e) for e in self.graph.get('edges', [])]
        self.selected = ''
        preds: dict = {n: [] for n in nodes}
        succs: dict = {n: [] for n in nodes}
        for a, b in edges:
            if a in nodes and b in nodes:
                preds[b].append(a)
                succs[a].append(b)
        middle = [n for n, k in nodes.items() if k in ('memory', 'local')]
        # topological order of the variables, ignoring the edges that close a loop
        state, order = {}, []

        def visit(node):
            state[node] = 1
            for nxt in succs[node]:
                if nxt in middle and nxt not in state:
                    visit(nxt)
            state[node] = 2
            order.append(node)
        for node in sorted(middle, key=lambda n: (-len([p for p in preds[n] if p not in middle]), n)):
            if node not in state:
                visit(node)
        order.reverse()
        rank = {n: i for i, n in enumerate(order)}
        layer = {}
        for node in order:
            earlier = [layer[p] for p in preds[node] if p in layer and rank[p] < rank[node]]
            layer[node] = 1 + max(earlier, default=0)
        deepest = max(layer.values(), default=0)
        if deepest > 3:
            layer = {n: 1 + (depth - 1) * 3 // deepest for n, depth in layer.items()}
            deepest = max(layer.values())
        def input_order(n):
            kind = nodes[n]
            return (0 if n.startswith('state.') else 1 if n.startswith('scope.') else 2 if kind == 'input' else 3, n)
        columns = [sorted([n for n, k in nodes.items() if k in ('input', 'setting')], key=input_order)]
        for index in range(1, deepest + 1):
            columns.append(sorted(n for n, depth in layer.items() if depth == index))
        columns.append(sorted(n for n, k in nodes.items() if k == 'output'))
        columns = [c for c in columns if c] or [[]]
        # order each column by where its sources are
        for _ in range(2):
            position = {n: i for column in columns for i, n in enumerate(column)}
            for column in columns[1:]:
                column.sort(key=lambda n: sum(position.get(p, 0) for p in preds[n]) / max(1, len(preds[n])))
        self._columns, self._edges, self._preds, self._succs = columns, edges, preds, succs
        self.height = dp(30) + max(len(c) for c in columns) * dp(32) + dp(10)
        self._trigger()

    def _connected(self, node):
        up, down, stack = {node}, {node}, [node]
        while stack:
            for p in self._preds.get(stack.pop(), []):
                if p not in up:
                    up.add(p)
                    stack.append(p)
        stack = [node]
        while stack:
            for s in self._succs.get(stack.pop(), []):
                if s not in down:
                    down.add(s)
                    stack.append(s)
        return up | down

    def redraw(self, *args):
        self.canvas.clear()
        self._boxes = {}
        if not getattr(self, '_columns', None) or self.width < 50:
            return
        nodes = self.graph.get('nodes', {})
        columns = self._columns
        n = len(columns)
        pad = dp(8)
        col_w = (self.width - 2 * pad) / n
        max_w = col_w - dp(14)
        lit = self._connected(self.selected) if self.selected else None
        with self.canvas:
            # headings
            headings = [(0, 'Inputs')] + ([(1, 'Variables')] if n > 2 else []) + [(n - 1, 'Outputs')]
            for index, text in headings:
                texture = _text(text, sp(11.5), PINK, bold=True)
                _blit(texture, self.x + pad + index * col_w + dp(4), self.top - texture.height - dp(4))
            for index, column in enumerate(columns):
                top = self.top - dp(30)
                for row, node in enumerate(column):
                    texture = _text(self._fit(node, max_w), sp(11.5), TEXT, mono=True)
                    w, h = min(max_w, texture.width + dp(18)), dp(24)
                    x = self.x + pad + index * col_w + (col_w - w) / 2
                    y = top - row * dp(32) - h
                    self._boxes[node] = (x, y, w, h, texture, index)
            for a, b in self._edges:
                if a not in self._boxes or b not in self._boxes:
                    continue
                ax, ay, aw, ah, _, acol = self._boxes[a]
                bx, by, bw, bh, _, bcol = self._boxes[b]
                on = lit is None or (a in lit and b in lit)
                colour = KIND_COLOUR.get(nodes.get(a), GREY)
                alpha = (0.55 if nodes.get(b) == 'output' else 0.3) if lit is None else (0.9 if on else 0.05)
                Color(*colour[:3], alpha)
                if bcol > acol:
                    sx, sy, ex, ey = ax + aw, ay + ah / 2, bx, by + bh / 2
                    bend = (ex - sx) * 0.45
                    Line(bezier=[sx, sy, sx + bend, sy, ex - bend, ey, ex, ey], width=1.1 if on and lit else 1)
                else:       # back to the same or an earlier column: a dashed loop under the boxes
                    sx, sy, ex, ey = ax + aw / 2, ay, bx + bw / 2, by
                    drop = dp(16) + abs(sx - ex) * 0.1
                    Line(bezier=[sx, sy, sx, sy - drop, ex, ey - drop, ex, ey], width=1, dash_length=4, dash_offset=3)
            for node, (x, y, w, h, texture, _) in self._boxes.items():
                kind = nodes.get(node, 'local')
                colour = KIND_COLOUR.get(kind, GREY)
                dim = lit is not None and node not in lit
                Color(*colour[:3], 0.06 if dim else (0.28 if node == self.selected else 0.14))
                RoundedRectangle(pos=(x, y), size=(w, h), radius=[dp(6)])
                Color(*colour[:3], 0.25 if dim else 1)
                Line(rounded_rectangle=(x, y, w, h, dp(6)), width=dp(1.3) if node == self.selected else 1)
                Color(1, 1, 1, 0.3 if dim else 1)
                Rectangle(texture=texture, size=texture.size,
                          pos=(x + (w - texture.width) / 2, y + (h - texture.height) / 2))

    @staticmethod
    def _fit(text, max_w):
        limit = max(6, int(max_w / dp(7.2)))
        return text if len(text) <= limit else text[:limit - 2] + '..'

    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos) or touch.is_mouse_scrolling:
            return super().on_touch_down(touch)
        for node, (x, y, w, h, _, _) in self._boxes.items():
            if x <= touch.x <= x + w and y <= touch.y <= y + h:
                self.selected = '' if self.selected == node else node
                return True
        if self.selected:
            self.selected = ''
            return True
        return super().on_touch_down(touch)


# --- flow (steps of update) -----------------------------------------------------------------

def _hex(colour):
    return ''.join(f'{int(c * 255):02x}' for c in colour[:3])


class _Block(BoxLayout):
    """A nested group of steps with a guide line on its left."""

    def __init__(self, depth: int, **kwargs):
        super().__init__(orientation='vertical', size_hint_y=None, padding=(dp(16) if depth else 0, 0, 0, 0),
                         spacing=dp(2), **kwargs)
        self.bind(minimum_height=self.setter('height'))
        self.depth = depth
        if depth:
            self.bind(pos=self._line, size=self._line)

    def _line(self, *args):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(1, 1, 1, 0.1)
            Line(points=[self.x + dp(6), self.y + dp(2), self.x + dp(6), self.top - dp(2)], width=1)


def _row(markup: str) -> Label:
    label = Label(text=markup, markup=True, font_size=sp(12.5), color=TEXT, halign='left', valign='middle',
                  size_hint_y=None, font_name='RobotoMono-Regular')
    label.bind(width=lambda w, v: setattr(w, 'text_size', (v, None)),
               texture_size=lambda w, s: setattr(w, 'height', s[1] + dp(6)))
    return label


def build_steps(steps, depth=0) -> _Block:
    block = _Block(depth)
    for step in steps:
        kind, text = step['kind'], escape_markup(step.get('text', ''))
        if kind == 'if':
            block.add_widget(_row(f'[color={_hex(BLUE)}][b]if[/b][/color] {text}'))
            block.add_widget(build_steps(step['body'], depth + 1))
            for branch in step.get('elif') or []:
                block.add_widget(_row(f'[color={_hex(BLUE)}][b]else if[/b][/color] {escape_markup(branch["text"])}'))
                block.add_widget(build_steps(branch['body'], depth + 1))
            if step.get('else'):
                block.add_widget(_row(f'[color={_hex(BLUE)}][b]otherwise[/b][/color]'))
                block.add_widget(build_steps(step['else'], depth + 1))
        elif kind == 'loop':
            block.add_widget(_row(f'[color={_hex(BLUE)}][b]repeat[/b][/color] {text}'))
            block.add_widget(build_steps(step['body'], depth + 1))
        elif kind == 'set':
            colour = PINK if step.get('memory') else (0.8, 0.8, 0.84, 1)
            block.add_widget(_row(f'[color={_hex(colour)}]{text}[/color]'))
        elif kind == 'out':
            block.add_widget(_row(f'[color={_hex(GREEN)}][b]{text}[/b][/color]'))
        elif kind == 'stage':
            block.add_widget(_row(f'[color={_hex(ORANGE)}][b]stage[/b] {text}[/color]'))
        elif kind == 'record':
            block.add_widget(_row(f'[color=ff6b6b][b]{text}[/b][/color]'))
        elif kind == 'call':
            block.add_widget(_row(f'[color={_hex(VIOLET)}][b]call[/b] {text}[/color]'))
        elif kind in ('return', 'note'):
            block.add_widget(_row(f'[color={_hex(MUTED)}]{text}[/color]'))
        else:
            block.add_widget(_row(f'[color={_hex(MUTED)}]{text}[/color]'))
    if not steps:
        block.add_widget(_row(f'[color={_hex(MUTED)}](nothing)[/color]'))
    return block


class CodeView(ScrollView):
    graph = DictProperty({})

    def __init__(self, **kwargs):
        super().__init__(do_scroll_x=False, bar_width=dp(4), **kwargs)
        self.content = BoxLayout(orientation='vertical', size_hint_y=None, spacing=dp(8), padding=(dp(4), dp(4), dp(10), dp(10)))
        self.content.bind(minimum_height=self.content.setter('height'))
        self.add_widget(self.content)
        self.bind(graph=self._build)

    def _build(self, *args):
        self.content.clear_widgets()
        graph = self.graph
        if graph.get('error'):
            self.content.add_widget(_row(f'[color={_hex(MUTED)}]Cannot read the code: {escape_markup(graph["error"])}[/color]'))
            return
        legend = '   '.join(f'[color={_hex(KIND_COLOUR[k])}][b]{name}[/b][/color]' for k, name in
                           (('input', 'input'), ('setting', 'setting'), ('memory', 'memory'),
                            ('local', 'step'), ('output', 'output')))
        note = 'Tap a box to follow it.'
        if graph.get('folded'):
            note += f' {graph["folded"]} less connected variables and settings are folded in.'
        info = Label(text=f'{legend}     [color={_hex(MUTED)}]{note}[/color]', markup=True, font_size=sp(11.5),
                     color=TEXT, halign='left', valign='middle', size_hint_y=None, height=dp(20))
        info.bind(size=lambda w, s: setattr(w, 'text_size', s))
        self.content.add_widget(info)
        if graph.get('nodes'):
            flow = DataFlowGraph()
            flow.graph = graph
            self.content.add_widget(flow)
        else:
            self.content.add_widget(_row(f'[color={_hex(MUTED)}]Nothing reaches the outputs.[/color]'))
        for title, steps in graph.get('flows', []):
            heading = Label(text=title, bold=True, font_size=sp(12), color=PINK, halign='left', valign='bottom',
                            size_hint_y=None, height=dp(26))
            heading.bind(size=lambda w, s: setattr(w, 'text_size', s))
            self.content.add_widget(heading)
            self.content.add_widget(build_steps(steps))


# --- the preview ----------------------------------------------------------------------------

class ScriptPreview(BoxLayout):
    """Call show_plugin(code), show_plugin_file(path) or show_sequencer(script)."""
    view = StringProperty('timeline')
    sequencer = BooleanProperty(False)
    show_tabs = BooleanProperty(SHOW_CODE_VIEW)
    dac_labels = ListProperty(['', ''])
    on_ready = ObjectProperty(None, allownone=True)     # called with the timeline when it is computed

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.timeline_view = TimelineView()
        self.code_view = CodeView()
        self._generation = 0
        self._chosen = False
        self.bind(view=self._show, sequencer=self._tabs, dac_labels=self._labels)
        Clock.schedule_once(lambda dt: (self._show(), self._tabs()))

    def _labels(self, *args):
        self.timeline_view.dac_labels = list(self.dac_labels)

    def _tabs(self, *args):
        self.show_tabs = SHOW_CODE_VIEW and not self.sequencer
        if not self.show_tabs:
            self.view = 'timeline'

    def _show(self, *args):
        body = self.ids.body
        body.clear_widgets()
        body.add_widget(self.code_view if self.view == 'code' else self.timeline_view)

    def on_touch_down(self, touch):
        if self.ids.tabs.collide_point(*touch.pos):
            self._chosen = True         # the user picked a view: keep it for the next scripts
        return super().on_touch_down(touch)

    # entry points --------------------------------------------------------------------------
    def clear(self, message: str = ''):
        self._generation += 1
        self.timeline_view.data = {}
        self.timeline_view.message = message
        self.code_view.graph = {}

    def show_plugin_file(self, path: str):
        try:
            with open(path, encoding='utf-8') as f:
                code = f.read()
        except OSError:
            self.clear()
            return
        self.show_plugin(code)

    def show_plugin(self, code: str):
        self.sequencer = False
        self._generation += 1
        generation = self._generation
        self.code_view.graph = script_preview.code_graph(code)
        self.timeline_view.data = {}
        self.timeline_view.message = ' '

        def work():
            try:
                data = script_preview.timeline(code)
            except Exception as e:
                data = {'ok': False, 'error': str(e)}
            Clock.schedule_once(lambda dt: self._timeline_ready(generation, data))
        Thread(target=work, daemon=True).start()

    def _timeline_ready(self, generation, data):
        if generation != self._generation:
            return
        self.timeline_view.data = data
        if not data.get('ok'):
            self.timeline_view.message = f'The preview failed: {data.get("error", "").splitlines()[-1][:160] if data.get("error") else ""}'
            if not self._chosen and self.show_tabs:
                self.view = 'code'
        elif data.get('reactive'):
            if not self._chosen and self.show_tabs:
                self.view = 'code'
        else:
            if not self._chosen:
                self.view = 'timeline'
        if self.on_ready is not None:
            self.on_ready(data)

    def show_sequencer(self, script: str):
        self.sequencer = True
        self._generation += 1
        data = script_preview.sequencer_timeline(script)
        self.timeline_view.data = data
        self.timeline_view.message = data.get('error', '')


def short_dac_label(text: str) -> str:
    """'a buzzer: it buzzes at 0 V' -> 'a buzzer', for the timeline's lane names."""
    text = text or ''
    for mark in ':,;(':
        text = text.split(mark)[0]
    text = text.strip()
    return text[:28] + ('..' if len(text) > 28 else '')


def dac_labels_from(config) -> list[str]:
    """Short names for the DAC outputs from the AI assistant's setup fields in the app config."""
    labels = []
    for key in ('dac0', 'dac1'):
        try:
            labels.append(short_dac_label(config.get('Assistant', key)))
        except Exception:
            labels.append('')
    return labels
