"""The app's activity log: short lines about what happened (recording, stage moves, hardware,
plugin output and errors), shown in the terminal panel of the main window.

Thread-safe: the camera, stage, plugin and DAQ threads add lines; listeners (the panel) are told
and must hand the update to their own thread. No Kivy import.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

INFO, WARN, ERROR = 'info', 'warn', 'error'


@dataclass(frozen=True)
class Entry:
    time: float
    source: str         # short tag: plugin, rec, live, stage, hw, daq, ai, app
    text: str
    level: str = INFO

    @property
    def clock(self) -> str:
        return time.strftime('%H:%M:%S', time.localtime(self.time))


class ActivityLog:
    def __init__(self, max_entries: int = 500):
        self._entries: deque[Entry] = deque(maxlen=max_entries)
        self._listeners: list[Callable[[Entry], None]] = []
        self._lock = threading.Lock()

    def add(self, source: str, text: str, level: str = INFO) -> None:
        text = ' '.join(str(text).split())          # one line
        if not text:
            return
        entry = Entry(time.time(), source, text, level)
        with self._lock:
            self._entries.append(entry)
            listeners = list(self._listeners)
        for listener in listeners:
            try:
                listener(entry)
            except Exception:
                pass

    def entries(self) -> list[Entry]:
        with self._lock:
            return list(self._entries)

    def subscribe(self, listener: Callable[[Entry], None]) -> None:
        with self._lock:
            self._listeners.append(listener)
