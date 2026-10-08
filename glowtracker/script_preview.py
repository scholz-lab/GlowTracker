"""What a plugin or sequencer script does, for the preview shown when one is loaded or proposed.
No Kivy import (the AI assistant window uses it too).

timeline(code): runs the plugin for a minute of simulated recording, twice, with two different
    simulated worms, and records the DAQ voltages frame by frame plus stage moves and recording
    starts/stops. When the two runs differ the plugin reacts to the worm, and its timeline only
    shows what it did for these simulated animals.
code_graph(code): reads update() (and the helpers it calls, setup and teardown) without running
    it: which inputs (state fields, scope readings, the clock, random numbers, settings) feed
    which variables, and which of those reach the outputs (DAC0, DAC1, the stage, recording);
    plus the structure of update() as nested steps.
sequencer_timeline(script): the voltage steps of a sequencer script, in the same shape as a
    plugin timeline.
describe(timeline): the timeline in a sentence or two, for the assistant.
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

SECONDS = 180.0
FPS = 10.0
# when the first simulated worm reverses (s)
REVERSALS = ((15.0, 18.0), (40.0, 43.0), (75.0, 78.0), (110.0, 114.0), (150.0, 153.0))


# --- timeline -------------------------------------------------------------------------------

def timeline(code: str, seconds: float = SECONDS, fps: float = FPS, timeout_s: float = 20.0) -> dict:
    """{'ok', 'error', 'unit', 't', 'runs': [{'v0', 'v1', 'events'}, ...], 'spans', 'reactive'}"""
    here = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory(prefix='glowtracker_preview_') as tmp:
        path = os.path.join(tmp, 'plugin_under_preview.py')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(code)
        cmd = [sys.executable, os.path.abspath(__file__), '--timeline', path, str(seconds), str(fps)]
        env = dict(os.environ, PYTHONPATH=here, PYTHONDONTWRITEBYTECODE='1')
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, cwd=tmp, env=env)
        except subprocess.TimeoutExpired:
            return {'ok': False, 'error': 'the preview run did not finish (update() blocks or is very slow)'}
    # the plugin may print too, even onto the report's line: find the report and read just it
    marker = proc.stdout.rfind('TIMELINE {')
    if marker >= 0:
        try:
            return json.JSONDecoder().raw_decode(proc.stdout, marker + len('TIMELINE '))[0]
        except ValueError:
            pass
    return {'ok': False, 'error': (proc.stderr or proc.stdout or 'no output').strip()[-800:]}


def _worm(kind: str, i: int, fps: float):
    """Position, velocity, reversing and brightness of a simulated worm at frame i."""
    t = i / fps
    if kind == 'a':             # crawls along x, reverses now and then, slows down for a while
        reversing = any(a <= t < b for a, b in REVERSALS)
        speed = 0.03 if 25 <= t < 32 or 125 <= t < 135 else 0.1
        direction = -1.0 if reversing else 1.0
        x = 0.1 * t - 0.4 * sum(min(max(t - a, 0), b - a) for a, b in REVERSALS)
        return (x, 0.0), (direction * speed / fps, 0.0), reversing, 60.0 + 25.0 * ((i // 30) % 2)
    # 'b': crawls along y, never reverses, stops twice, dimmer
    still = 30 <= t < 36 or 120 <= t < 130
    y = 0.06 * (t - min(max(t - 30, 0), 6) - min(max(t - 120, 0), 10))
    return (0.0, y), (0.0, 0.0 if still else 0.06 / fps), False, 35.0


def _timeline_child(path: str, seconds: float, fps: float) -> None:
    import random
    import traceback

    import numpy as np
    import plugin_check

    frames = int(seconds * fps)
    report = {'ok': True, 'error': '', 'unit': 's', 'seconds': seconds,
              't': [round(i / fps, 3) for i in range(frames)],
              'runs': [], 'spans': [list(s) for s in REVERSALS]}

    def finish():
        print('TIMELINE ' + json.dumps(report), flush=True)
        os._exit(0)

    box = plugin_check.Sandbox(fps=fps, moves_allowed=True)
    for kind in 'ab':
        random.seed(0)
        np.random.seed(0)
        box.events, box.recording = [], True
        box.daq.channelVoltages = [0.0, 0.0]
        v0, v1, trail, i = [], [], [], 0
        try:
            controller, scope = box.load(path)
            if callable(getattr(controller, 'setup', None)):
                controller.setup(scope)
            for i in range(frames):
                xy, velocity, reversing, brightness = _worm(kind, i, fps)
                trail = (trail + [xy])[-50:]
                state = box.state(i, xy=xy, velocity=velocity, trail=trail, tracking=True,
                                  reversing=reversing, brightness=brightness)
                controller.update(state, scope)
                v0.append(round(box.daq.channelVoltages[0], 4))
                v1.append(round(box.daq.channelVoltages[1], 4))
        except Exception:
            report.update(ok=False, error=plugin_check._short_traceback(traceback.format_exc(), path)
                          + f'\n(at {i / fps:.1f} s of the preview)')
            if not report['runs']:
                finish()
            break
        report['runs'].append({'v0': v0, 'v1': v1,
                               'events': [[round(t, 2), k, text] for t, k, text in box.events[:200]]})
    runs = report['runs']
    report['reactive'] = len(runs) == 2 and (
        runs[0]['v0'] != runs[1]['v0'] or runs[0]['v1'] != runs[1]['v1']
        or [e[1:] for e in runs[0]['events']] != [e[1:] for e in runs[1]['events']])
    finish()


def sequencer_timeline(script: str) -> dict:
    """A sequencer script's steps as a timeline of DAC0 and DAC1."""
    import plugin_check
    try:
        levels, unit = plugin_check.sequencer_levels(script)
    except Exception as e:
        return {'ok': False, 'error': f'the script cannot be read: {e}'}
    if not levels:
        return {'ok': False, 'error': 'the script has no steps'}
    end = max(t for t, _, _ in levels)
    end = max(end * 1.12, end + 5) if end > 0 else 10.0
    t, v0, v1 = [0.0], [0.0], [0.0]
    for trigger, a, b in levels:
        t += [trigger, trigger]
        v0 += [v0[-1], a]
        v1 += [v1[-1], b]
    t.append(end)
    v0.append(v0[-1])
    v1.append(v1[-1])
    return {'ok': True, 'error': '', 'unit': unit, 'seconds': round(end, 3), 't': t,
            'runs': [{'v0': v0, 'v1': v1, 'events': []}], 'spans': [], 'reactive': False, 'steps': True}


def describe(data: dict) -> str:
    """The timeline in words, e.g. for the assistant to check its plugin does what was asked."""
    if not data.get('ok'):
        return f'Preview: could not run ({data.get("error", "")}).'
    run, times, unit = data['runs'][0], data['t'], data['unit']
    parts = []
    for name, values in (('DAC0', run['v0']), ('DAC1', run['v1'])):
        levels = sorted({round(v, 2) for v in values})
        changes = sum(1 for a, b in zip(values, values[1:]) if a != b)
        if len(levels) == 1:
            parts.append(f'{name} stays at {levels[0]:g} V')
        else:
            first = next(times[i + 1] for i, (a, b) in enumerate(zip(values, values[1:])) if a != b)
            parts.append(f'{name} changes {changes} times between {", ".join(f"{v:g}" for v in levels[:6])} V '
                         f'(first at {first:g} {unit})')
    span = data.get('seconds', times[-1])
    text = (f'Preview ({span:g} {unit}, simulated worm): ' if not data.get('steps') else f'Timeline ({span:g} {unit}): ') + '; '.join(parts) + '.'
    kinds = [e[1] for e in run['events']]
    if kinds.count('stage'):
        text += f' {kinds.count("stage")} stage move(s).'
    if kinds.count('record'):
        text += f' {kinds.count("record")} recording start/stop(s).'
    if data.get('reactive'):
        text += ' The outputs depend on the worm (a second simulated worm gave different outputs).'
    return text


# --- code graph -----------------------------------------------------------------------------

OUTPUT_CALLS = {'set_voltage', 'light_off'}
STAGE_CALLS = {'move_rel', 'move_abs'}
RECORD_CALLS = {'start_recording', 'stop_recording'}
SCOPE_READS = {'voltage', 'voltages', 'get_position', 'get_frame', 'is_recording', 'fps', 'daq_connected',
               'is_stopping'}
MUTATORS = {'append', 'appendleft', 'extend', 'insert', 'add', 'update', 'pop', 'popleft', 'clear',
            'remove', 'discard', 'setdefault', 'sort', 'reverse'}
CLOCK = {('time', 'time'), ('time', 'monotonic'), ('time', 'perf_counter')}


@dataclass
class Graph:
    nodes: dict = field(default_factory=dict)      # id -> kind: input | setting | memory | local | output
    edges: set = field(default_factory=set)        # (source id, target id)
    flows: list = field(default_factory=list)      # [(title, steps)] for update() and its helpers
    error: str = ''

    def as_dict(self) -> dict:
        return {'nodes': self.nodes, 'edges': sorted(self.edges), 'flows': self.flows, 'error': self.error}


def _short(node, limit: int = 70) -> str:
    try:
        text = ast.unparse(node)
    except Exception:
        text = '…'
    text = ' '.join(text.split())
    return text if len(text) <= limit else text[:limit - 3] + '...'


class _Analyzer:
    def __init__(self, tree: ast.Module):
        self.graph = Graph()
        self.cls = next((n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Controller'), None)
        funcs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
        if self.cls is not None:
            funcs.update({n.name: n for n in self.cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))})
        self.funcs = funcs
        module_names = {t.id for n in tree.body if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign))
                        for t in (n.targets if isinstance(n, ast.Assign) else [n.target]) if isinstance(t, ast.Name)}
        self.globals = {name for f in funcs.values() for n in ast.walk(f) if isinstance(n, ast.Global) for name in n.names}
        self.module_settings = module_names - self.globals
        self.self_written = set()
        for f in funcs.values():
            for n in ast.walk(f):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target] if isinstance(n, (ast.AugAssign, ast.AnnAssign)) else []
                for t in targets:
                    for sub in ast.walk(t):
                        if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name) and sub.value.id == 'self':
                            self.self_written.add(sub.attr)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in MUTATORS:
                    owner = n.func.value
                    if isinstance(owner, ast.Attribute) and isinstance(owner.value, ast.Name) and owner.value.id == 'self':
                        self.self_written.add(owner.attr)
        self.stack: list[str] = []

    # node registration
    def node(self, ident: str, kind: str) -> str:
        if ident not in self.graph.nodes or kind == 'memory':
            self.graph.nodes[ident] = kind
        return ident

    def link(self, sources, target: str) -> None:
        for source in sources:
            if source != target:
                self.graph.edges.add((source, target))

    # reads ---------------------------------------------------------------------------------
    def reads(self, expr, env) -> set[str]:
        found: set[str] = set()
        if expr is None:
            return found
        if isinstance(expr, ast.Attribute):
            chain = self._chain(expr)
            if chain:
                root, rest = chain[0], chain[1:]
                if root == env['state']:
                    return {self.node('state.' + '.'.join(rest[:2]), 'input')}
                if root == env['scope'] and rest and rest[0] in SCOPE_READS:
                    return {self.node('scope.' + rest[0], 'input')}
                if root == 'self' and rest:
                    if rest[0] in self.self_written:
                        return {self.node('self.' + rest[0], 'memory')}
                    if rest[0] in self.funcs:
                        return found
                    return {self.node(rest[0], 'setting')}
        if isinstance(expr, ast.Name):
            name = expr.id
            if name in env['locals']:
                return set(env['locals'][name])
            if name in self.globals:
                return {self.node(name, 'memory')}
            if name in self.module_settings:
                return {self.node(name, 'setting')}
            return found
        if isinstance(expr, ast.Call):
            return self.call(expr, env, as_value=True)
        if isinstance(expr, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            inner = dict(env, locals=dict(env['locals']))
            for gen in expr.generators:
                sources = self.reads(gen.iter, inner)
                for target in ast.walk(gen.target):
                    if isinstance(target, ast.Name):
                        inner['locals'][target.id] = sources
                found |= sources
                for cond in gen.ifs:
                    found |= self.reads(cond, inner)
            parts = [expr.key, expr.value] if isinstance(expr, ast.DictComp) else [expr.elt]
            for part in parts:
                found |= self.reads(part, inner)
            return found
        if isinstance(expr, ast.Lambda):
            return found
        for child in ast.iter_child_nodes(expr):
            if isinstance(child, ast.expr):
                found |= self.reads(child, env)
        return found

    @staticmethod
    def _chain(expr) -> list[str] | None:
        parts = []
        while isinstance(expr, ast.Attribute):
            parts.append(expr.attr)
            expr = expr.value
        if isinstance(expr, ast.Name):
            return [expr.id] + parts[::-1]
        return None

    # calls ---------------------------------------------------------------------------------
    def call(self, node: ast.Call, env, as_value=False, ctrl=frozenset()) -> set[str]:
        func = node.func
        args = set()
        for a in list(node.args) + [k.value for k in node.keywords]:
            args |= self.reads(a, env)
        chain = self._chain(func) if isinstance(func, ast.Attribute) else None
        if chain and len(chain) == 2:
            root, name = chain
            if root == env['scope']:
                if name in OUTPUT_CALLS:
                    self._output(node, name, env, ctrl)
                    return set()
                if name in STAGE_CALLS:
                    self.link(args | set(ctrl), self.node('Stage', 'output'))
                    return set()
                if name in RECORD_CALLS:
                    self.link(set(ctrl), self.node('Recording', 'output'))
                    return set()
                if name in SCOPE_READS:
                    return {self.node('scope.' + name, 'input')} | args
                return set()
            if (root, name) in CLOCK:
                return {self.node('clock', 'input')}
            if root in ('random', 'np') or root == 'random':
                return {self.node('random', 'input')} | args
            if root == 'self' and name in self.funcs:
                return self._helper(self.funcs[name], node, env, ctrl) | (args if as_value else set())
        if chain and chain[0] == 'np' and len(chain) >= 3 and chain[1] == 'random':
            return {self.node('random', 'input')} | args
        if isinstance(func, ast.Name) and func.id in self.funcs and func.id not in ('update', 'setup', 'teardown'):
            return self._helper(self.funcs[func.id], node, env, ctrl)
        if isinstance(func, ast.Attribute) and func.attr in MUTATORS:
            owner = self.reads(func.value, env)
            for target in owner:
                if self.graph.nodes.get(target) in ('memory', 'local'):
                    self.link(args | set(ctrl), target)
            return owner | args
        if isinstance(func, ast.Attribute):
            return self.reads(func.value, env) | args
        return args

    def _output(self, node, name, env, ctrl):
        args = list(node.args)
        keywords = {k.arg: k.value for k in node.keywords}
        value = (args[0] if args else keywords.get('volts')) if name == 'set_voltage' else None
        channel = (args[1] if len(args) > 1 else keywords.get('channel')) if name == 'set_voltage' \
            else (args[0] if args else keywords.get('channel'))
        sources = set(ctrl) | (self.reads(value, env) if value is not None else set())
        if channel is None or (isinstance(channel, ast.Constant) and channel.value is None):
            targets = ['DAC0', 'DAC1']
        elif isinstance(channel, ast.Constant) and channel.value in (0, 1):
            targets = [f'DAC{channel.value}']
        else:
            targets = ['DAC0', 'DAC1']
            sources |= self.reads(channel, env)
        for target in targets:
            self.link(sources, self.node(target, 'output'))

    def _helper(self, fn, call: ast.Call, env, ctrl) -> set[str]:
        if fn.name in self.stack or len(self.stack) > 5:
            return set()
        params = [a.arg for a in fn.args.args]
        if params and params[0] == 'self':
            params = params[1:]
        local_env = {'state': None, 'scope': None, 'locals': {}, 'fn': fn.name}
        for param, arg in zip(params, call.args):
            if isinstance(arg, ast.Name) and arg.id == env['state']:
                local_env['state'] = param
            elif isinstance(arg, ast.Name) and arg.id == env['scope']:
                local_env['scope'] = param
            else:
                local_env['locals'][param] = self.reads(arg, env)
        for keyword in call.keywords:
            if keyword.arg:
                local_env['locals'][keyword.arg] = self.reads(keyword.value, env)
        self.stack.append(fn.name)
        returns: set[str] = set()
        self.walk(fn.body, local_env, frozenset(ctrl), returns)
        self.stack.pop()
        return returns

    # statements ----------------------------------------------------------------------------
    def assign_target(self, target, sources, env):
        if isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                self.assign_target(item, sources, env)
            return
        if isinstance(target, ast.Starred):
            self.assign_target(target.value, sources, env)
            return
        if isinstance(target, ast.Subscript):
            owner = target.value
            sources = sources | self.reads(target.slice, env)
            if isinstance(owner, ast.Attribute) and isinstance(owner.value, ast.Name) and owner.value.id == 'self':
                ident = self.node('self.' + owner.attr, 'memory')
                self.link(sources, ident)
            elif isinstance(owner, ast.Name):
                self.assign_target(owner, sources | self.reads(owner, env), env)
            return
        if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == 'self':
            ident = self.node('self.' + target.attr, 'memory')
            self.link(sources, ident)
            return
        if isinstance(target, ast.Name):
            if target.id in self.globals:
                ident = self.node(target.id, 'memory')
                self.link(sources, ident)
            else:
                ident = self.node(target.id, 'local')
                self.link(sources, ident)
                env['locals'][target.id] = {ident}

    def walk(self, body, env, ctrl, returns):
        for stmt in body:
            if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                value = stmt.value
                sources = self.reads(value, env) | set(ctrl) if value is not None else set(ctrl)
                targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                for target in targets:
                    if isinstance(stmt, ast.AugAssign):
                        sources = sources | self.reads(target, env)
                    self.assign_target(target, sources, env)
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                self.call(stmt.value, env, ctrl=ctrl)
            elif isinstance(stmt, (ast.If, ast.While)):
                inner = ctrl | self.reads(stmt.test, env)
                self.walk(stmt.body, env, inner, returns)
                self.walk(stmt.orelse, env, inner, returns)
            elif isinstance(stmt, (ast.For, ast.AsyncFor)):
                sources = self.reads(stmt.iter, env)
                self.assign_target(stmt.target, sources | set(ctrl), env)
                self.walk(stmt.body, env, ctrl | sources, returns)
                self.walk(stmt.orelse, env, ctrl, returns)
            elif isinstance(stmt, (ast.With, ast.AsyncWith)):
                self.walk(stmt.body, env, ctrl, returns)
            elif isinstance(stmt, ast.Try):
                for part in (stmt.body, *[h.body for h in stmt.handlers], stmt.orelse, stmt.finalbody):
                    self.walk(part, env, ctrl, returns)
            elif isinstance(stmt, ast.Return):
                returns |= self.reads(stmt.value, env) | set(ctrl)
            elif isinstance(stmt, ast.Match):
                subject = self.reads(stmt.subject, env)
                for case in stmt.cases:
                    self.walk(case.body, env, ctrl | subject, returns)

    def run(self) -> Graph:
        for name in ('setup', 'update', 'teardown'):
            fn = self.funcs.get(name)
            if fn is None:
                continue
            params = [a.arg for a in fn.args.args]
            if params and params[0] == 'self':
                params = params[1:]
            env = {'state': params[0] if name == 'update' and params else None,
                   'scope': params[-1] if params else None, 'locals': {}, 'fn': name}
            self.walk(fn.body, env, frozenset(), set())
        return self.graph


def _steps(body, analyzer: _Analyzer, helpers: list) -> list:
    """update()'s body as nested steps the preview draws: if / set / output / call / ..."""
    steps = []
    for stmt in body:
        if isinstance(stmt, ast.If):
            orelse = stmt.orelse
            step = {'kind': 'if', 'text': _short(stmt.test), 'body': _steps(stmt.body, analyzer, helpers)}
            branches = []
            while len(orelse) == 1 and isinstance(orelse[0], ast.If):
                branches.append({'text': _short(orelse[0].test), 'body': _steps(orelse[0].body, analyzer, helpers)})
                orelse = orelse[0].orelse
            step['elif'] = branches
            step['else'] = _steps(orelse, analyzer, helpers) if orelse else None
            steps.append(step)
        elif isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
            head = (f'for {_short(stmt.target, 30)} in {_short(stmt.iter, 40)}' if not isinstance(stmt, ast.While)
                    else f'while {_short(stmt.test)}')
            steps.append({'kind': 'loop', 'text': head, 'body': _steps(stmt.body, analyzer, helpers)})
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
            op = {ast.Add: '+=', ast.Sub: '-=', ast.Mult: '*=', ast.Div: '/='}.get(type(getattr(stmt, 'op', None)), '=') \
                if isinstance(stmt, ast.AugAssign) else '='
            target = ', '.join(_short(t, 30) for t in targets)
            memory = any(_short(t).startswith('self.') or _short(t) in analyzer.globals for t in targets)
            value = _short(stmt.value, 60) if stmt.value is not None else ''
            steps.append({'kind': 'set', 'text': f'{target} {op} {value}', 'memory': memory})
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            call = stmt.value
            name = call.func.attr if isinstance(call.func, ast.Attribute) else getattr(call.func, 'id', '')
            if name in OUTPUT_CALLS:
                args = list(call.args)
                keywords = {k.arg: k.value for k in call.keywords}
                if name == 'set_voltage':
                    value = args[0] if args else keywords.get('volts')
                    channel = args[1] if len(args) > 1 else keywords.get('channel')
                    volts = _short(value, 40) if value is not None else '?'
                else:
                    channel = args[0] if args else keywords.get('channel')
                    volts = '0'
                if channel is None or (isinstance(channel, ast.Constant) and channel.value is None):
                    where = 'DAC0 + DAC1'
                elif isinstance(channel, ast.Constant):
                    where = f'DAC{channel.value}'
                else:
                    where = f'DAC[{_short(channel, 20)}]'
                steps.append({'kind': 'out', 'text': f'{where} = {volts} V'})
            elif name in STAGE_CALLS:
                steps.append({'kind': 'stage', 'text': _short(call)})
            elif name in RECORD_CALLS:
                steps.append({'kind': 'record', 'text': name.replace('_', ' ')})
            elif name in ('print', 'log'):
                steps.append({'kind': 'note', 'text': _short(call)})
            elif name in analyzer.funcs and name not in ('update', 'setup', 'teardown'):
                if name not in helpers:
                    helpers.append(name)
                steps.append({'kind': 'call', 'text': f'{name}()'})
            else:
                steps.append({'kind': 'other', 'text': _short(call)})
        elif isinstance(stmt, ast.Return):
            steps.append({'kind': 'return', 'text': 'return' + (f' {_short(stmt.value, 50)}' if stmt.value else '')})
        elif isinstance(stmt, ast.Try):
            steps.extend(_steps(stmt.body, analyzer, helpers))
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            steps.extend(_steps(stmt.body, analyzer, helpers))
        elif isinstance(stmt, (ast.Global, ast.Pass, ast.Nonlocal)) or (
                isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)):
            continue        # docstrings and the like
        else:
            steps.append({'kind': 'other', 'text': _short(stmt)})
    # helper calls inside expressions (x = self._decide(state)) are listed too
    for stmt in body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, 'id', '')
                if name in analyzer.funcs and name not in ('update', 'setup', 'teardown') and name not in helpers:
                    helpers.append(name)
    return steps


def collapse_locals(nodes: dict, edges: set) -> tuple[dict, set]:
    """Drop local variables, linking what fed them straight to what they fed."""
    nodes, edges = dict(nodes), set(edges)
    for local in [n for n, k in nodes.items() if k == 'local']:
        into = {a for a, b in edges if b == local}
        out = {b for a, b in edges if a == local}
        edges = {(a, b) for a, b in edges if local not in (a, b)}
        edges |= {(a, b) for a in into for b in out if a != b}
        del nodes[local]
    return nodes, edges


def prune(nodes: dict, edges: set) -> tuple[dict, set]:
    """Keep only what reaches an output: the rest does not change what the plugin does."""
    reaches = {n for n, k in nodes.items() if k == 'output'}
    changed = True
    while changed:
        changed = False
        for a, b in edges:
            if b in reaches and a not in reaches:
                reaches.add(a)
                changed = True
    nodes = {n: k for n, k in nodes.items() if n in reaches}
    return nodes, {(a, b) for a, b in edges if a in nodes and b in nodes}


def _degree(edges: set) -> dict:
    degree: dict = {}
    for a, b in edges:
        degree[a] = degree.get(a, 0) + 1
        degree[b] = degree.get(b, 0) + 1
    return degree


def simplify(nodes: dict, edges: set, max_middle: int = 12, max_settings: int = 5) -> tuple[dict, set, int]:
    """Keep a big graph readable: fold local variables, then the least connected memory
    variables, through to what they feed; keep the most used settings. Returns how many nodes
    were folded away."""
    before = len(nodes)
    if sum(1 for k in nodes.values() if k in ('memory', 'local')) > max_middle:
        nodes, edges = collapse_locals(nodes, edges)
    middle = [n for n, k in nodes.items() if k == 'memory']
    if len(middle) > max_middle:
        degree = _degree(edges)
        for name in sorted(middle, key=lambda n: degree.get(n, 0))[:len(middle) - max_middle]:
            nodes[name] = 'local'
        nodes, edges = collapse_locals(nodes, edges)
    settings = [n for n, k in nodes.items() if k == 'setting']
    if len(settings) > max_settings:
        degree = _degree(edges)
        for name in sorted(settings, key=lambda n: degree.get(n, 0))[:len(settings) - max_settings]:
            del nodes[name]
        edges = {(a, b) for a, b in edges if a in nodes and b in nodes}
    return nodes, edges, before - len(nodes)


def code_graph(code: str, max_middle: int = 12) -> dict:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return dict(Graph(error=f'line {e.lineno}: {e.msg}').as_dict(), folded=0)
    analyzer = _Analyzer(tree)
    if 'update' not in analyzer.funcs:
        return dict(Graph(error='no update(state, scope) found').as_dict(), folded=0)
    graph = analyzer.run()
    nodes, edges = prune(graph.nodes, graph.edges)
    nodes, edges, folded = simplify(nodes, edges, max_middle)
    graph.nodes, graph.edges = nodes, edges
    helpers: list[str] = []
    graph.flows.append(('update()', _steps(analyzer.funcs['update'].body, analyzer, helpers)))
    seen = 0
    while seen < len(helpers) and seen < 6:
        name = helpers[seen]
        graph.flows.append((f'{name}()', _steps(analyzer.funcs[name].body, analyzer, helpers)))
        seen += 1
    for name in ('setup', 'teardown'):
        if name in analyzer.funcs:
            graph.flows.append((f'{name}()', _steps(analyzer.funcs[name].body, analyzer, [])))
    result = graph.as_dict()
    result['folded'] = folded
    return result


if __name__ == '__main__' and len(sys.argv) >= 5 and sys.argv[1] == '--timeline':
    _timeline_child(sys.argv[2], float(sys.argv[3]), float(sys.argv[4]))
