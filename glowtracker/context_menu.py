"""The right-click menu: a small list at the pointer. A click outside or Escape closes it. The look
is in layout.kv.
"""
from __future__ import annotations

from kivy.core.window import Window
from kivy.factory import Factory
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout

SEPARATOR = None


class ContextMenu(BoxLayout):
    keepsTooltips = True
    current: 'ContextMenu | None' = None

    @classmethod
    def show(cls, pos: tuple, items: list) -> 'ContextMenu':
        """items: (text, action) or (text, action, enabled), and SEPARATOR between groups."""
        cls.close()
        menu = cls()
        for item in items:
            if item is SEPARATOR:
                menu.add_widget(Factory.MenuSeparator())
                continue
            text, action, *enabled = item
            button = Factory.MenuItem(text=text, disabled=not (enabled[0] if enabled else True))
            button.bind(on_release=lambda b, action=action: menu.choose(action))
            menu.add_widget(button)
        menu.anchor = pos
        menu.bind(size=menu.place)
        Window.bind(on_keyboard=menu._key, mouse_pos=menu._hover)
        Window.add_widget(menu)
        cls.current = menu
        menu.place()
        return menu

    @classmethod
    def close(cls) -> None:
        menu = cls.current
        if menu is not None:
            cls.current = None
            Window.unbind(on_keyboard=menu._key, mouse_pos=menu._hover)
            Window.remove_widget(menu)

    def place(self, *args) -> None:
        """At the pointer, kept inside the window."""
        self.x = max(min(self.anchor[0], Window.width - self.width - dp(4)), dp(4))
        self.top = min(max(self.anchor[1], self.height + dp(4)), Window.height - dp(4))

    def choose(self, action) -> None:
        self.close()
        action()

    def _key(self, window, key, *args) -> bool:
        if key == 27:               # Escape
            self.close()
            return True
        return False

    def _hover(self, window, pos) -> None:
        for child in self.children:
            child.hover = child.collide_point(*pos) and not child.disabled

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            super().on_touch_down(touch)
        else:
            self.close()
        return True                 # the click that closes the menu does nothing else

    def on_touch_up(self, touch):
        super().on_touch_up(touch)
        return self.collide_point(*touch.pos)
