"""Automatic checks for plugins and sequencer scripts written by the AI assistant.

Three layers, each cheaper than the next is expensive:

1. static_check(code): reads the code without running it. Unknown `state.` fields and `scope.`
   methods (checked against the real WormState and Scope classes), imports outside the allowed
   list, file/process/eval access, blocking calls in update(), constant voltages out of range.
2. dry_run(code): checks that the plugin executes. It is loaded and run in a separate Python
   process, with a timeout, through the app's own PluginHost and Scope with a fake DAQ: setup,
   a few hundred update() calls with varied inputs, teardown. Reports a crash (with the line),
   a hang or a too slow update(). No hardware is touched, and writing files outside a temporary
   folder fails. Run only on code that passed static_check: it is a guard against mistakes, not
   a sandbox against hostile code.
3. sequencer_summary(script): the timeline of a sequencer script in words.

The results go back to the model and onto the proposal card for the user. No Kivy import.
"""
from __future__ import annotations

import ast
import dataclasses
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass

MAX_VOLTAGE = 4.95
ALLOWED_IMPORTS = {'math', 'random', 'collections', 'json', 'time', 'numpy', 'statistics',
                   'itertools', 'functools', 'dataclasses', 'enum', 'typing', '__future__'}
FILE_IMPORTS = {'os', 'pathlib', 'csv'}          # allowed with a warning (plugins keep tables on disk)
FORBIDDEN_CALLS = {'eval', 'exec', 'compile', '__import__', 'input', 'breakpoint',
                   'globals', 'setattr', 'delattr'}
FORBIDDEN_OS = {'system', 'popen', 'remove', 'unlink', 'rmdir', 'removedirs', 'rename', 'renames',
                'replace', 'kill', 'chmod', 'chown', 'fork', 'startfile', 'truncate'}
BLOCKING_IN_UPDATE = {('time', 'sleep'), ('scope', 'wait_for_recording')}
STAGE_MOVES = {'move_rel', 'move_abs'}
RECORDING_CONTROL = {'start_recording', 'stop_recording'}


@dataclass
class Issue:
    level: str          # 'error' (not shown to the user) or 'warning' (shown, with the note)
    line: int
    message: str

    def __str__(self) -> str:
        return f'line {self.line}: {self.message}' if self.line else self.message


def _api_names() -> tuple[set[str], set[str]]:
    """Public attributes of WormState and Scope, from the classes the app really uses."""
    import script_api
    state = {f.name for f in dataclasses.fields(script_api.WormState)}
    state |= {n for n, v in vars(script_api.WormState).items()
              if not n.startswith('_') and (isinstance(v, property) or callable(v))}
    scope = {n for n in dir(script_api.Scope) if not n.startswith('_')}
    return state, scope


# --- 1. static ------------------------------------------------------------------------------

def _update_functions(tree: ast.Module) -> list[tuple[ast.FunctionDef, str]]:
    """(function, role) for update/setup/teardown at module level or in class Controller."""
    found = []
    bodies = [tree.body] + [n.body for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Controller']
    for body in bodies:
        for node in body:
            if isinstance(node, ast.FunctionDef) and node.name in ('update', 'setup', 'teardown'):
                found.append((node, node.name))
    return found


def _param_roles(fn: ast.FunctionDef) -> dict[str, str]:
    """Map the function's parameter names to 'state' / 'scope' by position."""
    names = [a.arg for a in fn.args.args]
    if names and names[0] == 'self':
        names = names[1:]
    if fn.name == 'update':
        roles = dict(zip(names, ('state', 'scope')))
    else:
        roles = dict(zip(names, ('scope',)))
    return roles


def static_check(code: str) -> list[Issue]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [Issue('error', e.lineno or 0, f'Python syntax error: {e.msg}')]
    issues: list[Issue] = []
    state_names, scope_names = _api_names()

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or '']
            for module in modules:
                top = module.split('.')[0]
                if top in FILE_IMPORTS:
                    issues.append(Issue('warning', node.lineno, f'imports {module}: the plugin works with files'))
                elif top not in ALLOWED_IMPORTS:
                    issues.append(Issue('error', node.lineno, f'import {module} is not allowed (allowed: '
                                        f'{", ".join(sorted((ALLOWED_IMPORTS | FILE_IMPORTS) - {"__future__"}))})'))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            issues.append(Issue('error', node.lineno, f'{node.func.id}() is not allowed in a plugin'))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'open':
            issues.append(Issue('warning', node.lineno, 'open(): the plugin reads or writes a file'))
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'os' \
                and (node.attr in FORBIDDEN_OS or node.attr.startswith(('exec', 'spawn'))):
            issues.append(Issue('error', node.lineno, f'os.{node.attr} is not allowed in a plugin'))
        elif isinstance(node, ast.Attribute) and node.attr in ('unlink', 'rmdir', 'rename', 'replace', 'chmod') \
                and any(isinstance(n, ast.Name) and n.id in ('Path', 'pathlib') for n in ast.walk(node.value)):
            issues.append(Issue('error', node.lineno, f'.{node.attr}() on files is not allowed in a plugin'))
        elif isinstance(node, ast.Attribute) and node.attr.startswith('__') and node.attr != '__init__':
            issues.append(Issue('error', node.lineno, f'access to {node.attr} is not allowed'))

    functions = _update_functions(tree)
    if not any(role == 'update' for _, role in functions):
        issues.append(Issue('error', 0, 'no update(state, scope) function or Controller.update method'))

    for fn, role in functions:
        roles = _param_roles(fn)
        for node in ast.walk(fn):
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in roles:
                kind = roles[node.value.id]
                known = state_names if kind == 'state' else scope_names
                if node.attr not in known:
                    issues.append(Issue('error', node.lineno, f'{kind}.{node.attr} does not exist '
                                        f'(available: {", ".join(sorted(known))})'))
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            owner = node.func.value.id if isinstance(node.func.value, ast.Name) else ''
            method = node.func.attr
            owner_role = roles.get(owner, owner)
            if role == 'update' and (owner_role, method) in BLOCKING_IN_UPDATE:
                issues.append(Issue('error', node.lineno, f'{owner}.{method}() blocks update(), which runs '
                                    f'once per frame; count frames or compare state.time_s instead'))
            if owner_role == 'scope' and method in STAGE_MOVES:
                issues.append(Issue('warning', node.lineno, f'scope.{method}() moves the stage'))
            if owner_role == 'scope' and method in RECORDING_CONTROL:
                issues.append(Issue('warning', node.lineno, f'scope.{method}() starts or stops the recording'))
            if owner_role == 'scope' and method == 'set_voltage' and node.args:
                value = _constant(node.args[0])
                if value is not None and not 0 <= value <= MAX_VOLTAGE:
                    issues.append(Issue('error', node.lineno, f'set_voltage({value}) is outside 0..{MAX_VOLTAGE} V '
                                        f'(the DAQ would clamp it silently)'))
        if role == 'update':
            for node in ast.walk(fn):
                if isinstance(node, ast.While) and _constant(node.test) and \
                        not any(isinstance(n, (ast.Break, ast.Return)) for n in ast.walk(node)):
                    issues.append(Issue('error', node.lineno, 'endless loop in update()'))
                elif isinstance(node, ast.While):
                    issues.append(Issue('warning', node.lineno, 'while loop in update(): make sure it ends '
                                        'within a frame'))

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id.lower()
            value = _constant(node.value)
            if 'volt' in name and value is not None and not 0 <= value <= MAX_VOLTAGE:
                issues.append(Issue('error', node.lineno, f'{node.targets[0].id} = {value} is outside '
                                    f'0..{MAX_VOLTAGE} V'))
    return _dedupe(issues)


def _constant(node) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        value = _constant(node.operand)
        return -value if value is not None else None
    if isinstance(node, ast.Constant) and node.value is True:
        return 1.0
    return None


def _dedupe(issues: list[Issue]) -> list[Issue]:
    seen, out = set(), []
    for issue in issues:
        key = (issue.level, issue.message)
        if key not in seen:
            seen.add(key)
            out.append(issue)
    return out


# --- 2. dry run -----------------------------------------------------------------------------

UPDATE_BUDGET_MS = 100.0    # the app warns when update() takes longer (PluginHost.update_budget_s)


@dataclass
class DryRun:
    ok: bool
    error: str = ''         # traceback when the plugin crashed, hung or was too slow
    frames: int = 0
    update_ms_max: float = 0.0

    def text(self) -> str:
        if not self.ok:
            return f'Dry run FAILED:\n{self.error}'
        return f'Test run: {self.frames} frames ran without errors (update() max {self.update_ms_max:.1f} ms).'


def dry_run(code: str, frames: int = 300, timeout_s: float = 20.0) -> DryRun:
    """Execute the plugin in a child process: load, setup, `frames` updates, teardown."""
    here = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory(prefix='glowtracker_dryrun_') as tmp:
        path = os.path.join(tmp, 'plugin_under_test.py')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(code)
        cmd = [sys.executable, os.path.abspath(__file__), '--dry-run', path, str(frames)]
        env = dict(os.environ, PYTHONPATH=here, PYTHONDONTWRITEBYTECODE='1')
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, cwd=tmp, env=env)
        except subprocess.TimeoutExpired:
            return DryRun(False, error=f'did not finish {frames} frames within {timeout_s:.0f} s '
                                       f'(update() blocks or is very slow)')
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith('DRYRUN '):
            return DryRun(**json.loads(line[7:]))
    return DryRun(False, error=(proc.stderr or proc.stdout or 'no output').strip()[-1500:])


class Sandbox:
    """What a test run of a plugin needs, set up in the child process: file writes only inside
    the run folder, a frame clock behind time.time()/monotonic()/sleep(), a fake DAQ, a fake
    stage and recording switch that only record what was asked, and the app's PluginHost."""

    def __init__(self, fps: float = 10.0, moves_allowed: bool = False):
        import builtins
        import time as real_time

        import numpy as np
        import script_api

        run_dir = os.path.realpath(os.getcwd())
        real_open = builtins.open

        def guarded_open(file, mode='r', *args, **kwargs):
            # Reading is fine; writing only inside the temporary run folder.
            if isinstance(file, (str, bytes, os.PathLike)) and any(c in mode for c in 'wax+'):
                target = os.path.realpath(os.fsdecode(file))
                if os.path.commonpath([target, run_dir]) != run_dir:
                    raise PermissionError(f'dry run: the plugin tried to write {os.fsdecode(file)}, '
                                          f'outside its run folder')
            return real_open(file, mode, *args, **kwargs)

        builtins.open = guarded_open
        self.fps = fps
        self.clock = {'t': 0.0}
        self.start_wall = 1_700_000_000.0
        self.perf = real_time.perf_counter
        # Plugins may time things with time.time()/monotonic(); let those follow the frame clock.
        real_time.time = lambda: self.start_wall + self.clock['t']
        real_time.monotonic = lambda: self.clock['t']
        real_time.sleep = lambda s: self.clock.__setitem__('t', self.clock['t'] + max(0.0, s))
        self.events: list[tuple[float, str, str]] = []      # (time, kind, text)
        self.recording = True
        sandbox = self

        class FakeDaq:
            def __init__(self):
                self.channelVoltages = [0.0, 0.0]

            @property
            def currentVoltage(self):
                return max(self.channelVoltages)

            def set_voltage(self, v, channel=None):
                v = min(max(float(v), 0.0), MAX_VOLTAGE)
                for c in ((0, 1) if channel is None else (int(channel),)):
                    self.channelVoltages[c] = v
                return v

            def safe_off(self):
                self.channelVoltages = [0.0, 0.0]

            def isConnected(self):
                return True

        class FakeStage:
            position = [0.0, 0.0, 0.0]

            def get_cached_position(self, unit='mm'):
                return list(self.position)

            def move_rel(self, delta, unit='mm', wait_until_idle=True):
                self.position = [a + b for a, b in zip(self.position, delta)]
                sandbox.events.append((sandbox.clock['t'], 'stage', 'move by ' + ', '.join(f'{v:g}' for v in delta)))
                return True

            def move_abs(self, target, unit='mm', wait_until_idle=True):
                self.position = list(target)
                sandbox.events.append((sandbox.clock['t'], 'stage', 'move to ' + ', '.join(f'{v:g}' for v in target)))
                return True

        def recording_control(start):
            sandbox.recording = bool(start)
            sandbox.events.append((sandbox.clock['t'], 'record', 'start recording' if start else 'stop recording'))

        self.daq = FakeDaq()
        self.image = np.zeros((120, 160), dtype=np.uint8)
        self.box: dict = {}
        stage = FakeStage()
        self.host = script_api.PluginHost(
            state_provider=lambda: self.box.get('state'), frame_provider=lambda: self.image,
            daq_getter=lambda: self.daq, stage_getter=lambda: stage,
            moves_blocked_reason=(lambda: None) if moves_allowed else (lambda: 'tracking is active (dry run)'),
            recording_control=recording_control, recording_state=lambda: self.recording,
            log_dir_getter=lambda: os.getcwd())
        self.host._warn = lambda text: None
        self.host.fps = fps

    def load(self, path: str):
        """A fresh instance of the plugin; returns (controller, scope)."""
        import script_api
        self.host.load(path)
        return self.host.controller, script_api.Scope(self.host)

    def state(self, i: int, *, xy, velocity, trail, tracking: bool, reversing: bool, brightness: float):
        import numpy as np
        import script_api
        t = self.clock['t'] = i / self.fps
        self.box['state'] = state = script_api.WormState(
            frame=i, time_s=t, wall_time=self.start_wall + t, stage_xy=xy, worm_xy=xy,
            cms_offset_px=(1.5, -0.8) if tracking else (0.0, 0.0),
            trail=np.array(trail if tracking else [], dtype=float).reshape(-1, 2),
            velocity=velocity if tracking else (0.0, 0.0), is_reversing=reversing,
            is_tracking=tracking, is_recording=self.recording, voltage=self.daq.currentVoltage,
            image_shape=self.image.shape, fps=self.fps,
            analysis=script_api.BrightnessStats(0.0, 255.0, brightness, brightness - 2.0, 0.2, 10.0, 90.0))
        return state


def _child(path: str, frames: int) -> None:
    """Runs in the child process: import the plugin with the app's PluginHost and Scope, call it
    with varied inputs (so most branches run) and a fake DAQ, print the result."""
    import traceback

    box = Sandbox(fps=10.0)

    def finish(**report):
        print('DRYRUN ' + json.dumps(report), flush=True)
        os._exit(0)

    try:
        controller, scope = box.load(path)
    except Exception:
        finish(ok=False, error=_short_traceback(traceback.format_exc(), path))
    worst, trail, i = 0.0, [], 0
    try:
        if callable(getattr(controller, 'setup', None)):
            controller.setup(scope)
        for i in range(frames):
            x, y = 0.012 * i, 0.004 * i
            trail = (trail + [(x, y)])[-50:]
            tracking = i % 100 < 90                 # flip the booleans now and then
            state = box.state(i, xy=(x, y), velocity=(0.012, 0.004), trail=trail, tracking=tracking,
                              reversing=i % 50 >= 40, brightness=40.0 + i % 20)
            t0 = box.perf()
            controller.update(state, scope)
            worst = max(worst, (box.perf() - t0) * 1000.0)
        if callable(getattr(controller, 'teardown', None)):
            controller.teardown(scope)
    except Exception:
        finish(ok=False, frames=i, error=_short_traceback(traceback.format_exc(), path) + f'\n(at frame {i})')
    box.host._close_log()
    if worst > UPDATE_BUDGET_MS:
        finish(ok=False, frames=frames, update_ms_max=worst,
               error=f'update() took up to {worst:.0f} ms; it must stay well under one frame '
                     f'({UPDATE_BUDGET_MS:.0f} ms) or frames are skipped')
    finish(ok=True, frames=frames, update_ms_max=worst)


def _short_traceback(text: str, path: str) -> str:
    """Keep the frames in the plugin file and the error line."""
    lines = text.strip().splitlines()
    keep = [lines[-1]]
    for i, line in enumerate(lines):
        if path in line:
            keep.insert(-1, line.strip().replace(path, 'plugin'))
            if i + 1 < len(lines):
                keep.insert(-1, '    ' + lines[i + 1].strip())
    return '\n'.join(keep)


# --- 3. sequencer ---------------------------------------------------------------------------

def sequencer_levels(script: str) -> tuple[list[tuple[float, float, float]], str]:
    """The (trigger, DAC0 volts, DAC1 volts) after each step of a valid script, and its unit."""
    import DAQ_control
    daq = DAQ_control.DAQControl()
    daq.parseTextScript(script)
    unit = 's' if daq.sequencerMode == DAQ_control.SequencerMode.Time else 'frames'
    levels, current = [], [0.0, 0.0]
    for trigger, command in daq.sequncerDict.items():
        if command[0] == 'on':
            current = [command[1], command[1]]
        elif command[0] == 'off':
            current = [0.0, 0.0]
        else:
            current = [v if v is not None else c for v, c in zip(command[1], current)]
        levels.append((float(trigger), current[0], current[1]))
    return levels, unit


def _pulses(steps: list[tuple[float, float]]) -> tuple[list, bool]:
    pulses, start, level = [], None, 0.0
    for trigger, volts in steps:
        if start is not None and volts != level:
            pulses.append((start, trigger, level))
            start = None
        if volts > 0 and start is None:
            start, level = trigger, volts
    open_end = start is not None
    if open_end:
        pulses.append((start, None, level))
    return pulses, open_end


def _describe_pulses(pulses: list, open_end: bool, unit: str) -> str:
    if not pulses:
        return 'stays at 0 V'
    parts = [f'{a:g}-{b:g} at {v:g} V' if b is not None else f'from {a:g} at {v:g} V until the recording ends'
             for a, b, v in pulses]
    widths = sorted({round(b - a, 6) for a, b, _ in pulses if b is not None})
    gaps = sorted({round(pulses[i + 1][0] - pulses[i][0], 6) for i in range(len(pulses) - 1)})
    text = f'{len(pulses)} on-period(s): ' + '; '.join(parts[:8]) + ('...' if len(parts) > 8 else '')
    text += f'. Lengths: {", ".join(f"{w:g}" for w in widths) or "-"} {unit}'
    text += f'; start-to-start: {", ".join(f"{g:g}" for g in gaps) or "-"} {unit}.'
    if open_end:
        text += ' The last on-period has no end: it lasts until the recording ends.'
    return text


def sequencer_summary(script: str) -> str:
    """The timeline of a valid sequencer script in words, e.g. for checking pulse timing."""
    levels, unit = sequencer_levels(script)
    channels = [_pulses([(t, v0) for t, v0, _ in levels]), _pulses([(t, v1) for t, _, v1 in levels])]
    head = f'Timeline ({unit} after Record): '
    if channels[0] == channels[1]:
        pulses, open_end = channels[0]
        if not pulses:
            return head + 'the outputs stay at 0 V.'
        return head + 'both outputs, ' + _describe_pulses(pulses, open_end, unit)
    return head + ' '.join(f'DAC{i}: {_describe_pulses(p, o, unit)}' for i, (p, o) in enumerate(channels))


if __name__ == '__main__' and len(sys.argv) >= 4 and sys.argv[1] == '--dry-run':
    _child(sys.argv[2], int(sys.argv[3]))
