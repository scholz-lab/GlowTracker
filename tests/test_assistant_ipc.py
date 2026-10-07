"""The connection between GlowTracker and the assistant window process."""
import os
import sys
import textwrap
import threading
import time
from multiprocessing import AuthenticationError
from multiprocessing.connection import Client

import pytest

import assistant_ipc as ipc


def wait_for(condition, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError('timed out')
        time.sleep(0.01)


@pytest.fixture
def server():
    on_gui = []

    def run_on_gui(fn):
        on_gui.append(threading.current_thread().name)
        return fn()

    server = ipc.CommandServer(run_on_gui=run_on_gui)
    server.on_gui = on_gui
    yield server
    server.close()


def connect(server, **kwargs):
    return ipc.CommandClient(server.address, server.authkey, **kwargs)


def test_commands_run_through_the_gui_dispatcher_and_return_results(server):
    server.register('add', lambda a, b: a + b)
    server.register('state', lambda: {'recording': False, 'daq_mode': 'Sequencer'})
    client = connect(server)
    assert client.call('add', a=2, b=3) == 5
    assert client.call('state') == {'recording': False, 'daq_mode': 'Sequencer'}
    assert len(server.on_gui) == 2
    client.close()


def test_errors_come_back_worded_for_the_user(server):
    def refuse(code, path):
        raise ipc.CommandError('A recording is running. Stop it first.')

    server.register('use_sequencer', refuse)
    server.register('broken', lambda: 1 / 0)
    client = connect(server)
    with pytest.raises(ipc.CommandError, match='Stop it first'):
        client.call('use_sequencer', code='x', path='y')
    with pytest.raises(ipc.CommandError, match='ZeroDivisionError'):
        client.call('broken')
    with pytest.raises(ipc.CommandError, match='unknown command'):
        client.call('nope')
    assert client.call is not None and client.connected
    client.close()


def test_calls_from_several_threads_get_their_own_answers(server):
    server.register('echo', lambda value: value)
    client = connect(server)
    results = {}

    def worker(n):
        results[n] = client.call('echo', value=n)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert results == {n: n for n in range(20)}
    client.close()


def test_app_events_reach_the_window(server):
    events = []
    client = connect(server, on_event=lambda name, args: events.append((name, args)))
    wait_for(lambda: server.connected)
    server.notify('focus')
    server.notify('settings_changed', section='Assistant')
    wait_for(lambda: len(events) == 2)
    assert events == [('focus', {}), ('settings_changed', {'section': 'Assistant'})]
    client.close()


def test_a_process_without_the_key_cannot_connect(server):
    with pytest.raises(AuthenticationError):
        Client(server.address, authkey=b'wrong key')
    assert not server.connected


def test_the_window_notices_when_glowtracker_goes_away(server):
    gone = threading.Event()
    server.register('slow', lambda: time.sleep(0.3))
    client = connect(server, on_disconnect=gone.set)
    wait_for(lambda: server.connected)
    server.close()
    assert gone.wait(5)
    with pytest.raises(ipc.CommandError, match='not reachable'):
        client.call('slow')


CHILD = textwrap.dedent('''
    import os, sys, threading
    sys.path.insert(0, sys.argv[1])
    import assistant_ipc
    out = open(sys.argv[2], 'a')
    stop = threading.Event()
    def on_event(name, args):
        out.write(name + '\\n'); out.flush()
        if name == 'shutdown':
            stop.set()
    client = assistant_ipc.CommandClient.from_environment(on_event=on_event, on_disconnect=stop.set)
    out.write('hello ' + str(client.call('hello', who='window')) + '\\n'); out.flush()
    stop.wait(20)
''')


def test_launcher_starts_focuses_and_closes_the_window_process(server, tmp_path):
    script = tmp_path / 'child.py'
    script.write_text(CHILD)
    log = tmp_path / 'child.log'
    server.register('hello', lambda who: f'from app to {who}')
    here = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'glowtracker')
    launcher = ipc.WindowLauncher(server, command=[sys.executable, str(script), here, str(log)])
    assert launcher.open() is True
    wait_for(lambda: log.exists() and 'hello from app to window' in log.read_text(), timeout=15)
    assert 'KEY' not in ' '.join(launcher.process.args)         # key is passed in the environment
    assert launcher.open() is False                             # already open: focus instead
    wait_for(lambda: 'focus' in log.read_text())
    launcher.close()
    assert not launcher.running and 'shutdown' in log.read_text()
