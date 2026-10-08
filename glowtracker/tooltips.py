"""Hover tooltips for the main window.

Give any widget a `tooltip` text (e.g. `tooltip: 'Start recording'` in kv) and call
`Tooltips.install(root)` once. Hovering a widget for a moment shows a small rounded label next to
the pointer; it hides when the pointer moves on, or while a popup covers the window.
"""
from __future__ import annotations

from kivy.clock import Clock
from kivy.core.window import Window
from kivy.metrics import dp
from kivy.uix.label import Label

DELAY_S = 0.5


class TooltipLabel(Label):
    """The floating label; its look is in layout.kv."""


class Tooltips:
    _root = None
    _label: TooltipLabel | None = None
    _pending = None
    _target = None

    @classmethod
    def install(cls, root) -> None:
        cls._root = root
        cls._label = TooltipLabel()
        Window.bind(mouse_pos=cls._moved)
        # something opened over the main window (settings, a popup): hide at once
        Window.bind(children=lambda *args: cls._covered())

    @classmethod
    def _moved(cls, window, pos) -> None:
        if cls._pending is not None:
            cls._pending.cancel()
        target = cls._under(pos)
        if target is not cls._target:
            cls._hide()
            cls._target = target
        if target is not None and cls._label.parent is None:
            cls._pending = Clock.schedule_once(lambda dt: cls._show(target, pos), DELAY_S)

    @classmethod
    def _under(cls, pos):
        root = cls._root
        top = [w for w in Window.children if w is not cls._label]
        if root is None or not top or top[0] is not root:
            return None                     # a popup is open
        found = None
        for widget in root.walk(restrict=True):
            text = getattr(widget, 'tooltip', '')
            if text and widget.get_root_window() is not None and widget.opacity > 0 \
                    and widget.collide_point(*widget.to_widget(*pos)):
                found = widget              # the deepest match wins
        return found

    @classmethod
    def _show(cls, target, pos) -> None:
        cls._pending = None
        if target is not cls._target or not getattr(target, 'tooltip', ''):
            return
        label = cls._label
        label.text = target.tooltip
        label.texture_update()
        label.size = (label.texture_size[0] + dp(16), label.texture_size[1] + dp(10))
        x = min(pos[0] + dp(12), Window.width - label.width - dp(4))
        y = pos[1] - label.height - dp(14)
        if y < dp(4):
            y = pos[1] + dp(18)
        label.pos = (x, y)
        if label.parent is None:
            Window.add_widget(label)

    @classmethod
    def _covered(cls) -> None:
        top = [w for w in Window.children if w is not cls._label]
        if not top or top[0] is not cls._root:
            if cls._pending is not None:
                cls._pending.cancel()
            cls._target = None
            cls._hide()

    @classmethod
    def _hide(cls) -> None:
        if cls._label is not None and cls._label.parent is not None:
            cls._label.parent.remove_widget(cls._label)
