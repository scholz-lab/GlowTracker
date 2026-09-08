"""Collapsible advanced rows within the existing Kivy settings groups."""

import json

from kivy.metrics import dp, sp
from kivy.uix.settings import SettingItem, SettingsWithSidebar
from kivy.uix.togglebutton import ToggleButton


class AdvancedSettingsToggle(ToggleButton):
    def __init__(self, count, **kwargs):
        super().__init__(
            size_hint_y=None, height=dp(36), font_size=sp(14),
            halign='left', valign='middle', padding=(dp(16), 0),
            background_normal='', background_down='',
            background_color=(64 / 255, 55 / 255, 58 / 255, 1),
            **kwargs,
        )
        self.count = count
        self.bind(size=self._update_text, state=self._update_text)
        self._update_text()

    def _update_text(self, *args):
        marker = '-' if self.state == 'down' else '+'
        self.text = f'{marker} Advanced ({self.count})'
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
