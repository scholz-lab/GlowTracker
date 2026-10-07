"""Automatic checks on assistant-written plugins and sequencer scripts."""
import os

import pytest

import llm_assist as la
import plugin_check as pc


def errors(code):
    return [str(i) for i in pc.static_check(code) if i.level == 'error']


def warnings(code):
    return [str(i) for i in pc.static_check(code) if i.level == 'warning']


@pytest.mark.parametrize('name', sorted(la.plugin_examples()))
def test_shipped_example_plugins_pass(name):
    with open(os.path.join(la.PLUGIN_EXAMPLES_DIR, name), encoding='utf-8') as f:
        code = f.read()
    assert errors(code) == []
    run = pc.dry_run(code)
    assert run.ok, run.error
    assert run.frames == 300 and 'ran without errors' in run.text()


def test_unknown_api_names_are_caught_with_the_real_list():
    code = 'class Controller:\n    def update(self, s, sc):\n        if s.worm_speed > 1:\n            sc.zap()\n'
    found = errors(code)
    assert any('state.worm_speed does not exist' in e and 'speed' in e for e in found)
    assert any('scope.zap does not exist' in e for e in found)


def test_unsafe_and_blocking_code_is_caught():
    code = ('import subprocess, os\nimport time\n'
            'def update(state, scope):\n'
            '    time.sleep(1)\n    scope.set_voltage(9)\n    os.remove("x")\n    eval("1")\n'
            '    while True:\n        pass\n')
    found = ' | '.join(errors(code))
    for expected in ('import subprocess', 'time.sleep() blocks update()', 'set_voltage(9.0) is outside',
                     'os.remove', 'eval()', 'endless loop'):
        assert expected in found


def test_blocking_wait_is_fine_in_setup_and_stage_moves_are_warnings():
    code = ('def setup(scope):\n    scope.wait_for_recording()\n'
            'def update(state, scope):\n    scope.move_rel(0.1, 0)\n')
    assert errors(code) == []
    assert any('moves the stage' in w for w in warnings(code))


def test_dry_run_reaches_branches_that_only_run_sometimes():
    code = 'def update(state, scope):\n    if state.is_reversing and not state.is_tracking:\n        1 / 0\n'
    run = pc.dry_run(code)
    assert not run.ok and 'ZeroDivisionError' in run.error and 'line 3' in run.error and 'at frame' in run.error


def test_dry_run_reports_crashes_in_setup_and_slow_updates():
    assert 'NameError' in pc.dry_run('def setup(scope):\n    nope\ndef update(state, scope):\n    pass\n').error
    slow = ('def update(state, scope):\n    total = 0\n'
            '    for i in range(3_000_000):\n        total += i\n')
    run = pc.dry_run(slow, frames=3)
    assert not run.ok and 'update() took up to' in run.error


def test_dry_run_cannot_write_outside_its_folder(tmp_path):
    target = tmp_path / 'should_not_exist.txt'
    code = f'def setup(scope):\n    open({str(target)!r}, "w")\ndef update(state, scope):\n    pass\n'
    run = pc.dry_run(code)
    assert not run.ok and 'outside its run folder' in run.error
    assert not target.exists()


def test_dry_run_time_follows_frames_so_timed_branches_run():
    code = ('import time\nclass Controller:\n    def setup(self, scope):\n        self.t0 = time.time()\n'
            '    def update(self, state, scope):\n'
            '        if time.time() - self.t0 > 20:\n            raise RuntimeError("reached")\n')
    run = pc.dry_run(code)
    assert not run.ok and 'reached' in run.error


def test_sequencer_timeline():
    text = pc.sequencer_summary('mode: [time]\n0: [off]\n10: [on, 4.5]\n11: [off]\n30: [on, 4.5]\n31: [off]\n50: [on, 3]')
    assert '3 on-period(s)' in text and '10-11 at 4.5 V' in text and 'start-to-start: 20' in text
    assert 'lasts until the recording ends' in text


def test_check_combines_everything():
    problem, report, warns = la.check(la.PLUGIN, 'import os\ndef update(state, scope):\n    scope.light_off()\n')
    assert problem == '' and 'ran without errors' in report and warns
    problem, _, _ = la.check(la.PLUGIN, 'def update(state, scope):\n    scope.set_voltage(state.nope)\n')
    assert 'state.nope does not exist' in problem
