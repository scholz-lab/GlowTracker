"""Collapsible advanced rows within the existing Kivy settings groups."""

import json

from kivy.graphics import BorderImage, Color, RoundedRectangle
from kivy.metrics import dp, sp
from kivy.uix.button import Button
from kivy.uix.settings import (SettingItem, SettingOptions, SettingPath, SettingSpacer,
                               SettingString, SettingsWithSidebar)
from kivy.uix.textinput import TextInput
from kivy.uix.togglebutton import ToggleButton

# The settings window's colours (layout.kv has the same ones)
PANEL = (0.13, 0.135, 0.15, 1)
INPUT = (0.18, 0.19, 0.21, 1)
BUTTON = (0.25, 0.26, 0.29, 1)
BLUE = (0.04, 0.52, 1, 1)
TEXT = (0.93, 0.93, 0.95, 1)


def _rounded(widget, rgba, radius=dp(8), canvas=None):
    """Draw a rounded rectangle that follows the widget; returns its Color."""
    canvas = canvas if canvas is not None else widget.canvas.before
    color, rect = Color(rgba=rgba), RoundedRectangle(radius=[radius])
    canvas.add(color)
    canvas.add(rect)

    def follow(*args):
        rect.pos, rect.size = widget.pos, widget.size
    widget.bind(pos=follow, size=follow)
    follow()
    return color


def style_card(popup, title_size=sp(17), padding=dp(12)) -> None:
    """Draw a popup as a rounded dark card, with a plain title and no separator line."""
    popup.children[0].padding = padding     # Popup's own layout: title, separator, content
    popup.background, popup.background_color = '', (0, 0, 0, 0)
    popup.separator_height, popup.title_size, popup.title_color = 0, title_size, TEXT
    # the card goes right after Kivy's (now invisible) background, under the content
    index = next(i + 1 for i, c in enumerate(popup.canvas.children) if isinstance(c, BorderImage))
    color, card = Color(rgba=PANEL), RoundedRectangle(radius=[dp(14)])
    popup.canvas.insert(index, card)
    popup.canvas.insert(index, color)

    def follow(*args):
        card.pos, card.size = popup.pos, popup.size
    popup.bind(pos=follow, size=follow)
    follow()


def style_popup(popup) -> None:
    """Give a settings edit popup the settings window's look: a rounded dark card, a dark input,
    rounded buttons with the confirming one in blue."""
    style_card(popup)
    for widget in popup.content.walk(restrict=True):
        if isinstance(widget, SettingSpacer):
            widget.opacity = 0
        elif isinstance(widget, TextInput):
            widget.background_normal = widget.background_active = ''
            widget.background_color = INPUT
            widget.foreground_color, widget.cursor_color = TEXT, BLUE
            widget.font_size = sp(16)
            widget.height = dp(42)
            widget.padding = [dp(12), (dp(42) - widget.line_height) / 2]
        elif isinstance(widget, Button):
            confirm = widget.text.lower() in ('ok', 'save', 'apply', 'select')
            selected = isinstance(widget, ToggleButton) and widget.state == 'down'
            widget.background_normal = widget.background_down = ''
            widget.background_color = (0, 0, 0, 0)
            widget.color = TEXT
            if widget.parent is not None and widget.parent.height > dp(44):
                widget.parent.height = dp(44)
            fill = _rounded(widget, BLUE if confirm or selected else BUTTON)
            if isinstance(widget, ToggleButton):
                widget.bind(state=lambda w, state, fill=fill: setattr(
                    fill, 'rgba', BLUE if state == 'down' else BUTTON))


def _styled(create):
    def create_popup(self, *args):
        create(self, *args)
        if self.popup is not None:
            style_popup(self.popup)
    return create_popup


for _cls in (SettingString, SettingPath, SettingOptions):
    if '_create_popup' in vars(_cls):
        _cls._create_popup = _styled(vars(_cls)['_create_popup'])


class AdvancedSettingsToggle(ToggleButton):
    def __init__(self, count, **kwargs):
        # A quiet link-style row, in the system blue
        super().__init__(
            size_hint_y=None, height=dp(36), font_size=sp(13),
            halign='left', valign='middle', padding=(0, 0),
            background_normal='', background_down='',
            background_color=(0, 0, 0, 0), color=(0.04, 0.52, 1, 1),
            **kwargs,
        )
        self.count = count
        self.bind(size=self._update_text, state=self._update_text)
        self._update_text()

    def _update_text(self, *args):
        self.text = ('Hide advanced settings' if self.state == 'down'
                     else f'Show {self.count} advanced setting{"s" if self.count != 1 else ""}')
        self.text_size = self.size


class AdvancedSettingsWithSidebar(SettingsWithSidebar):
    def create_json_panel(self, title, config, filename=None, data=None):
        if filename is None and data is None:
            raise ValueError('You must specify either the filename or data')
        if filename is not None:
            with open(filename, encoding='utf-8') as stream:
                definitions = json.load(stream)
        else:
            definitions = json.loads(data)
        if not isinstance(definitions, list):
            raise ValueError('The first element must be a list')

        # "advanced" is our display metadata, not a Kivy setting property.
        clean = [{key: value for key, value in item.items() if key != 'advanced'}
                 for item in definitions]
        panel = super().create_json_panel(title, config, data=json.dumps(clean))
        if not any(item.get('advanced') is True for item in definitions):
            return panel

        widgets = list(reversed(panel.children))
        # Kivy adds a panel heading before the rows defined in JSON.
        prefix_count = len(widgets) - len(definitions)
        ordered = [(widget, None) for widget in widgets[:prefix_count]]
        groups = []
        group = []
        for definition, widget in zip(definitions, widgets[prefix_count:]):
            if definition['type'] == 'title' and group:
                groups.append(group)
                group = []
            group.append((definition, widget))
        if group:
            groups.append(group)

        toggles = []
        for group in groups:
            count = sum(item.get('advanced') is True for item, _ in group
                        if item['type'] != 'title')
            toggle = AdvancedSettingsToggle(count) if count else None
            if group[0][0]['type'] == 'title':
                ordered.append((group[0][1], None))
                group = group[1:]
            if toggle is not None:
                toggles.append(toggle)
                ordered.append((toggle, None))
            for definition, widget in group:
                gate = toggle if definition.get('advanced') is True else None
                ordered.append((widget, gate))

        def update_visibility(*args):
            # Keep the original row objects and order, including custom editors.
            # Detached rows consume no space and cannot receive user input.
            visible = [(widget, gate) for widget, gate in ordered
                       if gate is None or gate.state == 'down']
            for widget, gate in visible:
                if gate is not None and widget.parent is None and isinstance(widget, SettingItem):
                    widget.value = panel.get_value(widget.section, widget.key)
            panel.clear_widgets()
            for widget, _ in visible:
                panel.add_widget(widget)

        for toggle in toggles:
            toggle.bind(state=update_visibility)
        update_visibility()
        return panel
