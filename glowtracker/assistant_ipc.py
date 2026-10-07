"""Connection between GlowTracker and the AI assistant window, which runs in its own process.

The assistant window never touches the app directly: it sends commands ("get_state",
"use_sequencer", ...) to a CommandServer in GlowTracker, which runs each one with a handler the app
registered and returns the result. The app can also push events to the window ("focus",
"shutdown"). Keeping the chat in its own process means its drawing never competes with live view
and tracking for the Kivy main thread, and a crash or hang there cannot affect a recording.

Transport: multiprocessing.connection on 127.0.0.1 with a random authentication key, so only a
process that was given the key (the window GlowTracker started) can connect. Messages are dicts:

    window -> app   {'id': 3, 'cmd': 'get_state', 'args': {}}
    app -> window   {'id': 3, 'ok': True, 'result': {...}}   or  {'id': 3, 'ok': False, 'error': '...'}
    app -> window   {'event': 'focus', 'args': {}}

No Kivy import, so both sides can be tested on their own.
"""
from __future__ import annotations

import itertools
import logging
import os
import secrets
import subprocess
import sys
import threading
from multiprocessing.connection import Client, Connection, Listener
from typing import Any, Callable

KEY_ENV = 'GLOWTRACKER_ASSISTANT_KEY'
ADDRESS_ENV = 'GLOWTRACKER_ASSISTANT_ADDRESS'

log = logging.getLogger(__name__)


class CommandError(Exception):
    """A command failed in the app; the message is worded for the user."""


class CommandServer:
    """Runs in GlowTracker. Accepts the assistant window's connection and answers its commands.

    `run_on_gui(fn)` must run fn on the GUI thread and return its result (or raise); handlers can
    then use widgets safely. Commands are answered one at a time per connection.
    """

    def __init__(self, run_on_gui: Callable[[Callable[[], Any]], Any] = lambda fn: fn()):
        self._handlers: dict[str, Callable[..., Any]] = {}
        self._run_on_gui = run_on_gui
        self.authkey = secrets.token_bytes(32)
        self._listener = Listener(('127.0.0.1', 0), authkey=self.authkey)
        self.address: tuple[str, int] = self._listener.address
        self._connections: list[Connection] = []
        self._send_lock = threading.Lock()
        self._closed = False
        threading.Thread(target=self._accept, name='assistant-server', daemon=True).start()

    def register(self, name: str, handler: Callable[..., Any]) -> None:
        self._handlers[name] = handler

    @property
    def connected(self) -> bool:
        return bool(self._connections)

    def notify(self, event: str, **args) -> None:
        """Push an event to every connected window."""
        for conn in list(self._connections):
            self._send(conn, {'event': event, 'args': args})

    def close(self) -> None:
        self._closed = True
        for conn in list(self._connections):
            try:
                conn.close()
            except OSError:
                pass
        try:
            self._listener.close()
        except OSError:
            pass

    def _accept(self) -> None:
        while not self._closed:
            try:
                conn = self._listener.accept()
            except Exception:           # wrong key, or the listener was closed
                if self._closed:
                    return
                continue
            self._connections.append(conn)
            threading.Thread(target=self._serve, args=(conn,), name='assistant-conn', daemon=True).start()

    def _serve(self, conn: Connection) -> None:
        try:
            while True:
                try:
                    request = conn.recv()
                except (EOFError, OSError):
                    return
                self._send(conn, self._answer(request))
        finally:
            if conn in self._connections:
                self._connections.remove(conn)

    def _answer(self, request: Any) -> dict:
        if not isinstance(request, dict) or 'cmd' not in request:
            return {'id': None, 'ok': False, 'error': 'malformed request'}
        handler = self._handlers.get(request['cmd'])
        if handler is None:
            return {'id': request.get('id'), 'ok': False, 'error': f'unknown command {request["cmd"]!r}'}
        args = request.get('args') or {}
        try:
            result = self._run_on_gui(lambda: handler(**args))
            return {'id': request.get('id'), 'ok': True, 'result': result}
        except CommandError as e:
            return {'id': request.get('id'), 'ok': False, 'error': str(e)}
        except Exception as e:
            log.exception('assistant command %s failed', request['cmd'])
            return {'id': request.get('id'), 'ok': False, 'error': f'{type(e).__name__}: {e}'}

    def _send(self, conn: Connection, message: dict) -> None:
        try:
            with self._send_lock:
                conn.send(message)
        except (OSError, ValueError):
            pass


class CommandClient:
    """Runs in the assistant window. call() is thread-safe and blocks until the app answers.

    on_event(name, args) is called from the reader thread for events the app pushes;
    on_disconnect() once when the app goes away (closed, crashed).
    """

    def __init__(self, address: tuple[str, int], authkey: bytes,
                 on_event: Callable[[str, dict], None] = lambda name, args: None,
                 on_disconnect: Callable[[], None] = lambda: None):
        self._conn = Client(address, authkey=authkey)
        self._on_event = on_event
        self._on_disconnect = on_disconnect
        self._ids = itertools.count(1)
        self._pending: dict[int, list] = {}         # id -> [threading.Event, reply]
        self._lock = threading.Lock()
        self.connected = True
        threading.Thread(target=self._read, name='assistant-client', daemon=True).start()

    @classmethod
    def from_environment(cls, **kwargs) -> 'CommandClient':
        host, port = os.environ[ADDRESS_ENV].rsplit(':', 1)
        return cls((host, int(port)), bytes.fromhex(os.environ[KEY_ENV]), **kwargs)

    def call(self, cmd: str, timeout: float = 15.0, **args) -> Any:
        if not self.connected:
            raise CommandError('GlowTracker is not reachable (was it closed?)')
        request_id = next(self._ids)
        waiter = [threading.Event(), None]
        with self._lock:
            self._pending[request_id] = waiter
            try:
                self._conn.send({'id': request_id, 'cmd': cmd, 'args': args})
            except (OSError, ValueError) as e:
                self._pending.pop(request_id, None)
                raise CommandError(f'GlowTracker is not reachable: {e}') from e
        if not waiter[0].wait(timeout):
            with self._lock:
                self._pending.pop(request_id, None)
            raise CommandError(f'GlowTracker did not answer {cmd!r} within {timeout:.0f} s')
        reply = waiter[1]
        if reply is None:
            raise CommandError('GlowTracker closed the connection')
        if not reply.get('ok'):
            raise CommandError(reply.get('error') or f'{cmd} failed')
        return reply.get('result')

    def close(self) -> None:
        try:
            self._conn.close()
        except OSError:
            pass

    def _read(self) -> None:
        try:
            while True:
                try:
                    message = self._conn.recv()
                except (EOFError, OSError):
                    return
                if 'event' in message:
                    try:
                        self._on_event(message['event'], message.get('args') or {})
                    except Exception:
                        log.exception('assistant event handler failed')
                    continue
                with self._lock:
                    waiter = self._pending.pop(message.get('id'), None)
                if waiter is not None:
                    waiter[1] = message
                    waiter[0].set()
        finally:
            self.connected = False
            with self._lock:
                waiters, self._pending = list(self._pending.values()), {}
            for waiter in waiters:
                waiter[0].set()             # reply stays None: "closed the connection"
            try:
                self._on_disconnect()
            except Exception:
                log.exception('assistant disconnect handler failed')


class WindowLauncher:
    """Runs in GlowTracker: starts the assistant window process, focuses it if already open, and
    closes it with the app. `command` is the child's command line (the address and key are passed
    in its environment, not on the command line)."""

    def __init__(self, server: CommandServer, command: list[str] | None = None):
        self.server = server
        here = os.path.dirname(os.path.abspath(__file__))
        self.command = command or [sys.executable, os.path.join(here, 'assistant_window.py')]
        self.process: subprocess.Popen | None = None

    @property
    def running(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def open(self) -> bool:
        """Start the window, or bring it to the front. Returns True if a new process was started."""
        if self.running:
            self.server.notify('focus')
            return False
        env = dict(os.environ)
        env[ADDRESS_ENV] = f'{self.server.address[0]}:{self.server.address[1]}'
        env[KEY_ENV] = self.server.authkey.hex()
        env.setdefault('KIVY_NO_ARGS', '1')
        self.process = subprocess.Popen(self.command, env=env, cwd=os.path.dirname(self.command[-1]))
        return True

    def close(self, timeout: float = 3.0) -> None:
        if not self.running:
            return
        self.server.notify('shutdown')
        try:
            self.process.wait(timeout)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            try:
                self.process.wait(2.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
