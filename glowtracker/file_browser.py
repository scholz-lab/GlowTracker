"""The app's file and folder picker: an Up key, the current folder (type a path and press Enter to
go there), a search field that filters as you type, and a Finder-like list with folder and file
icons. Used by every "load file" / "choose folder" popup; the look is in layout.kv.

`FileBrowser` keeps the API the popups already used on Kivy's FileChooserListView: `path`,
`selection`, `dirselect`, an `on_submit` event (double click), plus `patterns` such as ['*.pfs'].
"""
from __future__ import annotations

import os
from fnmatch import fnmatch

from kivy.metrics import dp, sp
from kivy.properties import BooleanProperty, ListProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.filechooser import FileChooserController, FileChooserListLayout

from Advanced_settings import style_card


class BrowserListLayout(FileChooserListLayout):
    _ENTRY_TEMPLATE = 'BrowserEntry'


class BrowserListView(FileChooserController):
    # not a FileChooserListView subclass, so Kivy's list rule (its stock layout) does not apply
    _ENTRY_TEMPLATE = 'BrowserEntry'


class FileBrowser(BoxLayout):
    path = StringProperty('')
    selection = ListProperty([])
    patterns = ListProperty([])         # e.g. ['*.pfs']; empty shows every file
    dirselect = BooleanProperty(False)
    query = StringProperty('')          # the search text

    __events__ = ('on_submit',)

    def on_kv_post(self, base_widget):
        chooser = self.ids.chooser
        chooser.filters = [self._accepts]
        chooser.filter_dirs = True
        chooser.bind(path=self._chooserMoved, selection=self.setter('selection'),
                     on_submit=lambda c, selection, touch: self.dispatch('on_submit', selection))
        self.bind(query=self._refilter, patterns=self._refilter)
        self.on_path(self, self.path)

    def on_path(self, instance, path: str) -> None:
        """Show `path`'s folder (a file path opens the folder it is in)."""
        if 'chooser' not in self.ids or not path:
            return
        path = os.path.abspath(os.path.expanduser(path))
        folder = path if os.path.isdir(path) else os.path.dirname(path)
        if os.path.isdir(folder) and self.ids.chooser.path != folder:
            self.ids.chooser.path = folder

    def _chooserMoved(self, chooser, path: str) -> None:
        self.path = path
        self.ids.search.text = ''

    def up(self) -> None:
        current = self.ids.chooser.path.rstrip(os.sep) or os.sep
        self.ids.chooser.path = os.path.dirname(current) or os.sep

    def go(self, text: str) -> None:
        """Go to a typed folder; put the field back if it is not one."""
        self.path = text.strip()
        self.ids.pathfield.text = self.ids.chooser.path

    def _accepts(self, folder: str, name: str) -> bool:
        base = os.path.basename(name.rstrip(os.sep)).lower()
        if self.query.strip() and self.query.strip().lower() not in base:
            return False
        if os.path.isdir(name):
            return True
        return not self.patterns or any(fnmatch(base, p.lower()) for p in self.patterns)

    def _refilter(self, *args) -> None:
        self.ids.chooser._trigger_update()

    def on_submit(self, selection: list) -> None:
        pass


def style_file_popup(popup) -> None:
    """The file popups' card: rounded, dark, bold title."""
    style_card(popup, title_size=sp(18), padding=dp(20))
    popup.title_font = 'Roboto-Bold'
