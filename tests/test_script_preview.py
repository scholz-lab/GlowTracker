"""The script preview: plugin timelines with simulated worms, the code graph, sequencer timelines."""
import os

import script_preview as sp

EXAMPLES = os.path.join(os.path.dirname(__file__), '..', 'examples', 'plugins')


def example(name):
    with open(os.path.join(EXAMPLES, name), encoding='utf-8') as f:
        return f.read()


def test_timed_plugin_gives_the_exact_timeline_and_does_not_react():
    data = sp.timeline(example('buzzer.py'))
    assert data['ok'] and data['reactive'] is False
    run = data['runs'][0]
    quiet = [t for t, v in zip(data['t'], run['v0']) if v == 0.0]     # buzzing at 0 V
    assert quiet and round(quiet[0]) == 10 and run['v0'] == run['v1']
    assert 'does not' not in sp.describe(data) and 'DAC0 changes 6 times' in sp.describe(data)


def test_plugin_that_follows_the_worm_is_flagged():
    code = ('def update(state, scope):\n'
            '    scope.set_voltage(4.5 if state.is_reversing else 0.0, channel=1)\n')
    data = sp.timeline(code)
    assert data['ok'] and data['reactive'] is True
    on = [t for t, v in zip(data['t'], data['runs'][0]['v1']) if v > 0]
    assert on[0] == 15.0 and all(v == 0 for v in data['runs'][1]['v1'])
    assert 'depend on the worm' in sp.describe(data)


def test_crash_is_reported_with_the_time():
    data = sp.timeline('def update(state, scope):\n    if state.time_s > 5:\n        1 / 0\n')
    assert not data['ok'] and 'ZeroDivisionError' in data['error'] and '5.' in data['error']


def test_stage_moves_and_recording_are_events():
    code = ('def update(state, scope):\n'
            '    if state.frame == 20:\n        scope.move_rel(1, 0)\n'
            '    if state.frame == 30:\n        scope.stop_recording()\n')
    events = sp.timeline(code)['runs'][0]['events']
    assert [e[1] for e in events] == ['stage', 'record'] and events[0][0] == 2.0


def test_code_graph_links_inputs_through_variables_to_outputs():
    graph = sp.code_graph(example('buzzer.py'))
    nodes, edges = graph['nodes'], {tuple(e) for e in graph['edges']}
    assert nodes['state.wall_time'] == 'input' and nodes['start'] == 'memory'
    assert nodes['DAC0'] == nodes['DAC1'] == 'output'
    assert ('state.wall_time', 't') in edges and ('t', 'buzzing') in edges and ('buzzing', 'DAC0') in edges
    title, steps = graph['flows'][0]
    assert title == 'update()' and steps[0]['kind'] == 'if'


def test_code_graph_channels_helpers_and_pruning():
    code = ('class Controller:\n'
            '    def setup(self, scope):\n        self.n = 0\n        self.unused = 1\n'
            '    def update(self, state, scope):\n'
            '        self.n += 1\n'
            '        if self._ready(state):\n            scope.set_voltage(self.n % 2, 1)\n'
            '    def _ready(self, s):\n        return s.speed > 0.1\n')
    graph = sp.code_graph(code)
    nodes, edges = graph['nodes'], {tuple(e) for e in graph['edges']}
    assert 'DAC1' in nodes and 'DAC0' not in nodes
    assert ('state.speed', 'DAC1') in edges and ('self.n', 'DAC1') in edges
    assert 'self.unused' not in nodes                       # does not reach an output
    assert [t for t, _ in graph['flows']] == ['update()', '_ready()', 'setup()']


def test_code_graph_reports_syntax_errors():
    assert 'line 1' in sp.code_graph('def update(:\n')['error']


def test_sequencer_timeline_steps():
    data = sp.sequencer_timeline('mode: [time]\n0: [off]\n10: [on, 4.5]\n11: [off]\n')
    assert data['ok'] and data['steps'] and data['unit'] == 's'
    assert max(data['runs'][0]['v0']) == 4.5 and data['seconds'] > 11
    assert not sp.sequencer_timeline('nonsense')['ok']
