"""GlowTracker's side of the assistant window: the commands it answers, against a fake app."""
import os
from types import SimpleNamespace

import pytest

import assistant_ipc as ipc
import DAQ_control
import GlowTracker as G

SCRIPT = 'mode: [time]\n0: [off]\n10: [on, 4.5]\n11: [off]'
PLUGIN = 'def update(state, scope):\n    scope.light_off()\n'


class FakeConfig:
    def __init__(self, values):
        self.values = values
        self.writes = 0

    def get(self, section, key, fallback=''):
        return self.values.get((section, key), fallback)

    def set(self, section, key, value):
        self.values[(section, key)] = value

    def write(self):
        self.writes += 1


def fake_app(tmp_path, recording=False):
    runtime = SimpleNamespace(
        imageacquisitionmanager=SimpleNamespace(recordbutton=SimpleNamespace(state='down' if recording else 'normal')),
        trackingcheckbox=SimpleNamespace(state='down'))
    root = SimpleNamespace(ids=SimpleNamespace(middlecolumn=SimpleNamespace(runtimecontrols=runtime),
                                               rightcolumn=SimpleNamespace(_popup=None)))
    config = FakeConfig({('Assistant', 'model'): 'm', ('Assistant', 'apikey'): 'k',
                         ('Experiment', 'exppath'): str(tmp_path)})
    plugin_host = SimpleNamespace(status='idle', is_running=lambda: False)
    return SimpleNamespace(config=config, root=root, daqControl=DAQ_control.DAQControl(), pluginHost=plugin_host)


@pytest.fixture
def host(tmp_path):
    host = G.AssistantHost(fake_app(tmp_path))
    yield host
    host.server.close()


def test_use_sequencer_saves_loads_and_selects_the_script(host, tmp_path):
    path = str(tmp_path / 'seq.txt')
    message = host.use_sequencer(code=SCRIPT, path=path)
    assert 'runs when you press Record' in message
    assert open(path).read() == SCRIPT + '\n'
    app = host.app
    assert app.config.get('DaqControl', 'sequencescript') == path
    assert app.config.get('DaqControl', 'mode') == DAQ_control.DAQMode.Sequencer.value
    assert app.daqControl.daqMode == DAQ_control.DAQMode.Sequencer
    assert list(app.daqControl.sequncerDict.items()) == [(0, ['off']), (10, ['on', 4.5]), (11, ['off'])]
    assert host.get_state()['sequencer_script'] == SCRIPT + '\n'
    host.use_sequencer(code=SCRIPT.replace('4.5', '3'), path=path)    # its own file may be replaced


def test_use_sequencer_refuses_while_recording_invalid_scripts_and_foreign_files(tmp_path):
    host = G.AssistantHost(fake_app(tmp_path, recording=True))
    try:
        with pytest.raises(ipc.CommandError, match='recording is running'):
            host.use_sequencer(code=SCRIPT, path=str(tmp_path / 'a.txt'))
        host.app.root.ids.middlecolumn.runtimecontrols.imageacquisitionmanager.recordbutton.state = 'normal'
        with pytest.raises(ipc.CommandError, match='4.95'):
            host.use_sequencer(code='mode: [time]\n1: [on, 9]', path=str(tmp_path / 'a.txt'))
        existing = tmp_path / 'mine.txt'
        existing.write_text('my own script')
        with pytest.raises(ipc.CommandError, match='already exists'):
            host.use_sequencer(code=SCRIPT, path=str(existing))
        assert existing.read_text() == 'my own script'
    finally:
        host.server.close()


def test_save_plugin_selects_it_and_current_plugin_reads_it(host, tmp_path):
    path = str(tmp_path / 'plug.py')
    assert 'press Start' in host.save_plugin(code=PLUGIN, path=path)
    assert host.app.config.get('DaqControl', 'pluginscript') == path
    assert host.current_plugin() == f'# {path}\n{PLUGIN}'
    with pytest.raises(ipc.CommandError, match='neither update'):
        host.save_plugin(code='x = 1', path=str(tmp_path / 'p2.py'))


def test_config_state_and_default_paths(host, tmp_path):
    assert host.get_config()['model'] == 'm' and host.get_config()['api_key'] == 'k'
    host.app.config.set('Assistant', 'dac1', 'a 590 nm LED')
    setup = host.get_config()['setup']
    assert set(setup) == {'subject', 'animal', 'dac0', 'dac1', 'notes'} and setup['dac1'] == 'a 590 nm LED'
    assert host.get_state()['dual_color_mode'] is False
    state = host.get_state()
    assert state['recording'] is False and state['tracking'] is True and state['daq_mode'] == 'Off'
    assert os.path.dirname(host.default_path('sequencer')) == str(tmp_path)
    assert host.default_path('plugin').endswith('.py')


def test_the_window_reaches_the_commands_over_the_connection(host, tmp_path):
    host.server._run_on_gui = lambda fn: fn()      # no Kivy loop in tests; dispatch is tested in test_assistant_ipc
    client = ipc.CommandClient(host.server.address, host.server.authkey)
    path = str(tmp_path / 'via_ipc.txt')
    assert 'Record' in client.call('use_sequencer', code=SCRIPT, path=path)
    assert client.call('get_state')['sequencer_file'] == path
    foreign = tmp_path / 'notes.py'
    foreign.write_text('# not from the assistant')
    with pytest.raises(ipc.CommandError, match='already exists'):
        client.call('save_plugin', code=PLUGIN, path=str(foreign))
    client.close()
