"""AI assistant chat: talk to a model about the experiment; it writes DAQ sequencer scripts and
plugins and hands them to you for review.

Runs on the `ox` agent runtime, vendored in glowtracker/ox (copied verbatim from the ox_microscope
project; to update, copy its src/ox/*.py over). Works with any OpenAI-compatible chat API that
supports tool calling (e.g. the GWDG / Max Planck "Chat AI" service, or OpenAI itself).

No GUI code here, so it can be tested and reused. The model gets no shell or file access, only the
GlowTracker tools defined below, and nothing it writes is executed: a proposed sequencer script
must pass the real parser and a plugin a syntax check, and then it is only shown to the user, who
decides whether to load or save it.

Settings (Settings > AI assistant): base URL, model, API key. If the key is empty, the
GLOWTRACKER_LLM_API_KEY or OPENAI_API_KEY environment variable is used.
"""
from __future__ import annotations

import ast
import asyncio
import functools
import itertools
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Protocol

import platformdirs
import structlog

from assistant_memory import MAX_FACT_CHARS, MemoryStore  # noqa: F401 (MemoryStore used by the window)
from ox.agent import AgentServer
from ox.bus import AsyncPort, Envelope, Msg
from ox.provider_http import HttpProvider
from ox.tools import ToolResolver
from ox.types import (Approval, AssistantMessage, Check, ModelSpec, SystemMessage, ToolMessage, ToolResult,
                      UserMessage)

if not structlog.is_configured():
    # ox logs every request at info level; keep the app console quiet.
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING))

DEFAULT_BASE_URL = 'https://chat-ai.academiccloud.de/v1'

# Chat models offered in Settings when the GWDG / Academic Cloud (SAIA) API is used, largest first.
# From https://docs.hpc.gwdg.de (SAIA API model names); coding, omni and embedding models left out.
# The Models button lists everything a key can use, and any other name can still be typed.
GWDG_MODELS = (
    ('qwen3.5-397b-a17b', 'Qwen 3.5, 397B mixture of experts: the largest'),
    ('mistral-medium-3.5-128b', 'Mistral Medium 3.5, 128B'),
    ('openai-gpt-oss-120b', 'OpenAI GPT-OSS, 120B open weights'),
    ('deepseek-v4-flash-0731', 'DeepSeek V4 Flash: fast'),
    ('glm-5.3-flash', 'GLM 5.3 Flash: fast'),
    ('gemma-4-31b-it', 'Gemma 4, 31B'),
    ('qwen3.8-27b', 'Qwen 3.8, 27B'),
    ('qwen3.6-35b-a3b', 'Qwen 3.6, 35B mixture of experts: small and fast'),
)


def is_gwdg(base_url: str) -> bool:
    return any(host in (base_url or '') for host in ('chat-ai.academiccloud.de', 'saia.gwdg.de'))
DEFAULT_MEMORY_FILE = os.path.join(platformdirs.user_data_dir('GlowTracker', appauthor=False),
                                   'assistant_memory.json')
KEY_ENV_VARS = ('GLOWTRACKER_LLM_API_KEY', 'OPENAI_API_KEY')

SEQUENCER = 'sequencer'
PLUGIN = 'plugin'


class AssistantError(Exception):
    """A problem talking to the API, worded for the user."""


# The user's own setup, from Settings > AI assistant: (config key, what it describes).
SETUP_FIELDS = (
    ('subject', 'Object of interest (what is tracked)'),
    ('animal', 'Animal and strain'),
    ('dac0', 'Output DAC0 is connected to'),
    ('dac1', 'Output DAC1 is connected to'),
    ('notes', 'Other notes from the lab'),
)


@dataclass
class AssistantConfig:
    base_url: str = DEFAULT_BASE_URL
    model: str = ''
    api_key: str = ''
    timeout_s: float = 120.0
    setup: dict = field(default_factory=dict)     # SETUP_FIELDS key -> the user's description
    memory_file: str = ''                         # '' -> DEFAULT_MEMORY_FILE

    def memory_path(self) -> str:
        return self.memory_file.strip() or DEFAULT_MEMORY_FILE

    def resolved_key(self) -> str:
        if self.api_key.strip():
            return self.api_key.strip()
        for name in KEY_ENV_VARS:
            value = os.environ.get(name, '').strip()
            if value:
                return value
        return ''

    def check(self) -> None:
        if not self.resolved_key():
            raise AssistantError('No API key. Enter it in Settings > AI assistant, or set the '
                                 'GLOWTRACKER_LLM_API_KEY environment variable.')
        if not self.model.strip():
            raise AssistantError('No model chosen. Pick one in Settings > AI assistant '
                                 '(the Models button lists what your key can use).')


# --- prompt ---------------------------------------------------------------------------------

SEQUENCER_GUIDE = """\
DAQ sequencer scripts drive a LabJack DAQ whose two analog outputs (DAC0 and DAC1, always set
together) control an optogenetic LED or other device. The sequence starts when the user presses
Record.

Exact format, one entry per line, nothing else:
    mode: [frame]          or    mode: [time]
    <trigger>: [on, <volts>]
    <trigger>: [off]

Rules:
- The first line chooses the clock. frame: triggers are whole frame numbers counted from the
  first recorded frame (0). time: triggers are seconds since recording started (decimals allowed).
- [on, V] sets both outputs to V volts, 0 <= V <= 4.95. [off] sets them to 0 V.
- An output keeps its value until the next command. Triggers must be unique and non-negative.
- In time mode, if several triggers pass between two camera frames only the latest one runs.
- When the recording ends the outputs return to 0 V.
- No comments, no other text and no Python: the parser accepts only this format.
- Prefer time mode unless the user explicitly wants frame-exact timing.

Example (1 s of 4.5 V light at 10 s and at 30 s):
mode: [time]
0: [off]
10: [on, 4.5]
11: [off]
30: [on, 4.5]
31: [off]
"""

PLUGIN_RULES = """\
Plugins are one Python file that the app loads and calls once per camera frame with the tracked
worm's state and a `scope` handle to set the DAQ outputs. Rules:
- Use a Controller class or module-level setup/update/teardown functions exactly as documented.
- update() must return quickly: never sleep, block or loop waiting; keep state in variables and
  count frames or compare state.wall_time instead.
- Only the standard library (math, random, collections, json, time) and numpy may be imported.
- Voltages are 0 to 4.95 V. If the hardware is active at 0 V (e.g. a buzzer that is quiet at
  4.5 V), set idle_voltage as documented.
- Never move the stage while tracking. Log the events that matter with scope.log(...).
- Keep it short, readable and commented for a biologist.
"""

GLOWTRACKER = """\
GlowTracker is a microscope tracking application that can track a small animal in bright-field,
single or dual epi-fluorescence imaging. Its interface controls the linear Zaber stages (X, Y and
Z) and the Basler camera properties. A LabJack DAQ has two analog outputs, DAC0 and DAC1, that
drive whatever the lab connected to them (see "This microscope" below).
"""

CHAT_RULES = """\
You are the assistant inside GlowTracker. You talk with a biologist about their experiment and
write DAQ sequencer scripts (fixed timing) or plugins (when the outputs must react to the animal's
behaviour, position or speed).

How to work:
- If an important detail is missing (voltage, durations, timing, which kind of output), ask a short
  question first; for small details make a sensible choice and say so.
- Call get_app_state when the current script, plugin or DAQ mode matters, e.g. when the user wants
  to change "the script". Call read_current_plugin to see the selected plugin's code.
- Before writing a plugin, read the closest example with read_plugin_example (the list is below)
  and follow its structure; use only the state fields and scope methods in the API reference.
- Deliver every script by calling propose_sequencer_script or propose_plugin. Do not paste the code
  into your message: the user sees it in the chat with Approve and Decline.
- These tools check automatically first: a sequencer script with the real parser (the result gives
  its timeline; compare it with the request); a plugin for unknown state/scope names, unsafe or
  blocking calls and out-of-range voltages, then by executing it once with a fake DAQ. If a check
  fails, fix it and call again. Mention any warnings to the user.
- Then the call waits for the user. The result tells you whether they approved (the script was
  saved and loaded), edited it before approving (the applied version is in the result), or
  declined (with their reason, if they gave one). Only describe as done what the result says was
  done. A plugin still has to be started by the user in DAQ > Plugin, and a sequencer script runs
  when they press Record.
- Answer in short, plain language: what the script does, the timing, and any assumptions.

Long-term memory:
- The lessons below were remembered on this microscope from earlier conversations. Follow them.
- When the user corrects you, or a check, GlowTracker or the user's decision shows a mistake you
  would likely repeat in another conversation, propose a lesson with remember: one short, general,
  lasting fact about this microscope, the lab's preferences or how to work here (e.g. "DAC0 drives
  a buzzer; to keep it quiet, use a plugin that holds DAC0 at 4.5 V, not a sequencer script").
  Not details of one experiment. The user approves, edits or declines it.
- Also call remember when the user asks you to remember something, and forget when a lesson is
  wrong or outdated (give its number). Do not remember what is already in the setup description.
"""


PLUGIN_EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'examples', 'plugins')


def _plugin_reference() -> str:
    """The plugin API documentation, examples/plugins/README.md, given to the model in full."""
    try:
        with open(os.path.join(PLUGIN_EXAMPLES_DIR, 'README.md'), encoding='utf-8') as f:
            return f.read()
    except OSError:
        logging.warning('assistant: examples/plugins/README.md not found; the model only gets a '
                        'short plugin API summary')
        return ('Plugin API: define update(state, scope). state.worm_xy, state.trail, state.fps, '
                'state.is_recording, state.wall_time; scope.set_voltage(v, channel=None), '
                'scope.light_off(), scope.log(**fields), scope.print(text).')


def plugin_examples() -> dict[str, str]:
    """{file name: first docstring line} of the example plugins (analysis scripts left out)."""
    examples = {}
    try:
        names = sorted(os.listdir(PLUGIN_EXAMPLES_DIR))
    except OSError:
        return examples
    for name in names:
        if not name.endswith('.py'):
            continue
        try:
            with open(os.path.join(PLUGIN_EXAMPLES_DIR, name), encoding='utf-8') as f:
                code = f.read()
            doc = ast.get_docstring(ast.parse(code)) or ''
        except (OSError, SyntaxError):
            continue
        if validate(PLUGIN, code):
            continue                  # not a plugin, e.g. a log analysis script
        examples[name] = doc.strip().splitlines()[0] if doc.strip() else ''
    return examples


def read_plugin_example(name: str) -> str:
    examples = plugin_examples()
    name = os.path.basename(name.strip())
    if name not in examples:
        return f'error: no example {name!r}. Available: {", ".join(examples) or "none"}'
    with open(os.path.join(PLUGIN_EXAMPLES_DIR, name), encoding='utf-8') as f:
        return f.read()


def setup_section(setup: dict | None) -> str:
    """'This microscope': the user's own description of their setup, and what is missing."""
    setup = setup or {}
    lines = [f'- {label}: {setup[key].strip()}' for key, label in SETUP_FIELDS if (setup.get(key) or '').strip()]
    missing = [label for key, label in SETUP_FIELDS
               if key in ('dac0', 'dac1', 'subject') and not (setup.get(key) or '').strip()]
    text = '=== This microscope (described by the user in Settings > AI assistant) ===\n'
    text += '\n'.join(lines) if lines else '(nothing described yet)'
    if missing:
        text += ('\nNot described: ' + '; '.join(missing) + '. Before writing a script that drives an '
                 'undescribed output, ask the user what it is connected to and how it responds to voltage.')
    text += ('\nTake this setup into account: e.g. a sequencer script always sets DAC0 and DAC1 '
             'together, and a device that is active at 0 V needs idle_voltage in a plugin.\n')
    return text


def system_prompt(setup: dict | None = None, memory: str = '') -> str:
    examples = plugin_examples()
    listing = '\n'.join(f'- {name}: {doc}' if doc else f'- {name}' for name, doc in examples.items())
    return (GLOWTRACKER + '\n' + setup_section(setup) + '\n' + CHAT_RULES
            + ('\n' + memory if memory else '')
            + '\n=== Sequencer scripts ===\n' + SEQUENCER_GUIDE
            + '\n=== Plugins ===\n' + PLUGIN_RULES
            + '\n--- plugin API reference (examples/plugins/README.md) ---\n' + _plugin_reference()
            + '\n--- end ---\n'
            + ('\nExample plugins (read one with read_plugin_example):\n' + listing + '\n' if listing else ''))


# --- validation -----------------------------------------------------------------------------

def validate(kind: str, code: str) -> str:
    """Return '' if the script is usable, otherwise the problem in one line."""
    if not code.strip():
        return 'the script is empty'
    if kind == SEQUENCER:
        import DAQ_control
        try:
            DAQ_control.DAQControl().parseTextScript(code)
        except Exception as e:
            return str(e)
        return ''
    if kind == PLUGIN:
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return f'Python syntax error on line {e.lineno}: {e.msg}'
        has_update = any(isinstance(n, ast.FunctionDef) and n.name == 'update' for n in tree.body)
        has_controller = any(
            isinstance(n, ast.ClassDef) and n.name == 'Controller'
            and any(isinstance(m, ast.FunctionDef) and m.name == 'update' for m in n.body)
            for n in tree.body)
        if not (has_update or has_controller):
            return 'the plugin defines neither update(state, scope) nor a Controller class with update'
        return ''
    raise ValueError(f'unknown kind {kind!r}')


_FENCE = re.compile(r'^```[\w+-]*[ \t]*\n(.*?)\n?```\s*$', re.DOTALL)
_THINK = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)


def _unfence(code: str) -> str:
    """Models sometimes wrap tool arguments in a Markdown fence; drop it."""
    match = _FENCE.match(code.strip())
    return match.group(1) if match else code


def clean_reply(text: str) -> str:
    """Drop reasoning models' <think> blocks from text shown to the user."""
    return _THINK.sub('', text).strip()


# --- GlowTracker tools ----------------------------------------------------------------------

def check(kind: str, code: str) -> tuple[str, str, list[str]]:
    """Run the automatic checks. Returns (problem, report, warnings): problem is '' when the
    script may be shown to the user; report describes what it does."""
    import plugin_check
    problem = validate(kind, code)
    if problem:
        return problem, '', []
    if kind == SEQUENCER:
        return '', plugin_check.sequencer_summary(code), []
    issues = plugin_check.static_check(code)
    errors = [str(i) for i in issues if i.level == 'error']
    warnings = [str(i) for i in issues if i.level == 'warning']
    if errors:
        return 'the static check found:\n- ' + '\n- '.join(errors), '', warnings
    run = plugin_check.dry_run(code)
    if not run.ok:
        return f'the test run failed:\n{run.error}', '', warnings
    return '', run.text(), warnings


@functools.lru_cache(maxsize=16)
def _check_cached(kind: str, code: str) -> tuple[str, str, tuple[str, ...]]:
    problem, report, warnings = check(kind, code)
    return problem, report, tuple(warnings)


def check_report(kind: str, code: str) -> Check:
    """check() as an ox Check: the error for the model, or the report shown with the approval.
    Cached, so applying an approved script does not test-run it again."""
    problem, report, warnings = _check_cached(kind, _unfence(code).strip())
    if problem:
        return Check(False, f'The {kind} is invalid, so it was not shown to the user: {problem}\n'
                            f'Fix it and call again.')
    return Check(True, '\n'.join([report] + [f'Warning: {w}' for w in warnings]).strip())


class AppBridge(Protocol):
    """What the chat needs from the app. Called from the assistant's worker thread: an
    implementation that touches the GUI must hand the work to the GUI thread itself."""

    def app_state(self) -> dict: ...
    def current_plugin(self) -> str: ...
    def apply(self, kind: str, code: str, path: str) -> str:
        """Save and load an approved script; return what was done. Raise AssistantError to refuse."""
    def default_path(self, kind: str) -> str: ...


def _register_memory_tools(tools: ToolResolver, memory: MemoryStore) -> None:
    def remember_check(args: dict) -> Check:
        problem = memory.problem(args.get('fact', ''))
        if problem:
            return Check(False, f'Not remembered: {problem}.')
        return Check(True, args.get('why', ''))

    def remember(fact: str, why: str = '') -> str:
        stored = memory.add(fact, why)
        return f'Remembered as lesson {stored.id}: {stored.text}'

    def forget_check(args: dict) -> Check:
        match = [f for f in memory.facts() if f.id == args.get('id')]
        if not match:
            return Check(False, f'There is no lesson {args.get("id")}.')
        return Check(True, f'Lesson {match[0].id}: {match[0].text}')

    def forget(id: int, why: str = '') -> str:
        removed = memory.remove(id)
        return f'Forgot lesson {removed.id}: {removed.text}'

    tools.add_function(
        'remember',
        'Propose a lasting lesson for this microscope, kept for all future conversations. The user '
        'approves, edits or declines it; waits for their decision.',
        lambda fact, why='': asyncio.to_thread(remember, fact, why),
        {'type': 'object', 'required': ['fact', 'why'], 'additionalProperties': False, 'properties': {
            'fact': {'type': 'string', 'minLength': 5, 'maxLength': MAX_FACT_CHARS,
                     'description': 'One short, general fact or rule, understandable without this conversation.'},
            'why': {'type': 'string', 'description': 'What happened that taught it, in one sentence.'}}},
        confirm=True, check=remember_check)
    tools.add_function(
        'forget',
        'Propose to delete a remembered lesson that is wrong or outdated. The user approves or declines.',
        lambda id, why='': asyncio.to_thread(forget, id, why),
        {'type': 'object', 'required': ['id', 'why'], 'additionalProperties': False, 'properties': {
            'id': {'type': 'integer', 'description': 'The lesson number.'},
            'why': {'type': 'string', 'description': 'Why it should go.'}}},
        confirm=True, check=forget_check)


def _register_tools(tools: ToolResolver, bridge: AppBridge) -> None:
    def get_app_state() -> str:
        return json.dumps(bridge.app_state(), indent=1, default=str)

    def read_current_plugin() -> str:
        return bridge.current_plugin() or '(no plugin file selected in the Plugin tab)'

    def apply(kind: str, code: str, path: str) -> ToolResult:
        try:
            done = bridge.apply(kind, _unfence(code).strip(), path or bridge.default_path(kind))
        except Exception as e:      # refused by the app, e.g. while recording
            return ToolResult(f'Not applied: {e}', is_error=True)
        report = check_report(kind, code).message
        return ToolResult(f'{done}\nAutomatic checks:\n{report}\nTell the user briefly what it does.')

    async def off_loop(fn, **kwargs):
        # The bridge may wait for the GUI thread; do not block the event loop meanwhile.
        return await asyncio.to_thread(fn, **kwargs)

    def checked(kind: str, key: str):
        return lambda args: asyncio.to_thread(check_report, kind, args.get(key, ''))

    summary = {'type': 'string', 'description': 'One sentence: what it does, shown to the user.'}
    path = {'type': 'string', 'description': 'Where to save it. Leave out to use the default folder.'}
    tools.add_function(
        'get_app_state',
        'Current GlowTracker state: recording, tracking, imaging mode, DAQ mode, the sequencer script, '
        'the selected plugin file and whether it runs.',
        lambda: off_loop(get_app_state), {'type': 'object', 'properties': {}, 'additionalProperties': False})
    tools.add_function(
        'read_current_plugin', 'Source code of the plugin file selected in the Plugin tab.',
        lambda: off_loop(read_current_plugin), {'type': 'object', 'properties': {}, 'additionalProperties': False})
    tools.add_function(
        'read_plugin_example',
        'Source code of one of the example plugins listed in the instructions. Read the closest one '
        'before writing a plugin.',
        lambda name: off_loop(read_plugin_example, name=name),
        {'type': 'object', 'required': ['name'], 'additionalProperties': False, 'properties': {
            'name': {'type': 'string', 'description': 'File name, e.g. template_controller.py'}}})
    tools.add_function(
        'propose_sequencer_script',
        'Propose a complete DAQ sequencer script. It is checked with the real parser; if valid, the '
        'user reviews it and approves (it is then saved and loaded as the sequencer script, DAQ mode '
        'Sequencer), edits it, or declines. Waits for their decision.',
        lambda script, summary='', path='': off_loop(apply, kind=SEQUENCER, code=script, path=path),
        {'type': 'object', 'required': ['script'], 'additionalProperties': False, 'properties': {
            'script': {'type': 'string', 'minLength': 1,
                       'description': 'The complete script, starting with the mode line.'},
            'summary': summary, 'path': path}},
        confirm=True, check=checked(SEQUENCER, 'script'))
    tools.add_function(
        'propose_plugin',
        'Propose a complete GlowTracker plugin file. It is checked (API names, safety, a test run); if '
        'valid, the user reviews it and approves (it is then saved and selected in DAQ > Plugin, where '
        'they press Start), edits it, or declines. Waits for their decision.',
        lambda code, summary='', path='': off_loop(apply, kind=PLUGIN, code=code, path=path),
        {'type': 'object', 'required': ['code'], 'additionalProperties': False, 'properties': {
            'code': {'type': 'string', 'minLength': 1, 'description': 'The complete Python file.'},
            'summary': summary, 'path': path}},
        confirm=True, check=checked(PLUGIN, 'code'))


# --- chat -----------------------------------------------------------------------------------

_ERROR = re.compile(r'^\[(?:LLM|Internal) Error[^\]]*\]:\s*', re.IGNORECASE)
_HINTS = {401: ' (check the API key)', 403: ' (the key has no access to this model)',
          404: ' (check the base URL and model name)', 429: ' (rate limit, try again shortly)'}


def _error_text(content: str) -> str:
    code = re.match(r'^\[LLM Error (\d+)\]', content)
    hint = _HINTS.get(int(code.group(1)), '') if code else ''
    message = _ERROR.sub('', content).strip()
    if code:
        message = f'The API returned HTTP {code.group(1)}{hint}. {message}'.strip()
    if 'tool' in message.lower() and ('support' in message.lower() or 'auto' in message.lower()):
        message += ' This model may not support tool calling; try another one.'
    return message


@dataclass
class ChatEvent:
    """What the GUI shows, in order:
    'text'      streamed answer text (text is the new piece)
    'thinking'  streamed reasoning of a reasoning model (text is the new piece)
    'message'   one assistant message is complete (text is its full answer text)
    'tool'      the model calls a tool (call_id, name; text is the JSON arguments)
    'approval'  a call waits for the user (call_id, name; data = arguments; text = check report);
                answer with ChatSession.answer()
    'result'    a tool finished (call_id, name, ok; text is what the model was told)
    'usage'     tokens used so far in this conversation (data: prompt, completion, cost or None)
    'error'     the turn failed (text is the problem, worded for the user)
    'done'      the turn is over and the user may write again (cancelled if stopped)
    """
    kind: str
    text: str = ''
    call_id: str = ''
    name: str = ''
    ok: bool = True
    cancelled: bool = False
    data: dict = field(default_factory=dict)


class ThinkSplitter:
    """Separate <think>...</think> blocks that some models stream inside the answer text.
    Tags may be split across chunks, so a possible partial tag is held back."""

    OPEN, CLOSE = '<think>', '</think>'

    def __init__(self) -> None:
        self.thinking = False
        self._pending = ''

    def feed(self, chunk: str) -> list[tuple[str, str]]:
        """Return [(kind, text)] with kind 'text' or 'thinking'."""
        out: list[tuple[str, str]] = []
        data = self._pending + chunk
        self._pending = ''
        while data:
            tag = self.CLOSE if self.thinking else self.OPEN
            index = data.lower().find(tag)
            if index >= 0:
                if index:
                    out.append(('thinking' if self.thinking else 'text', data[:index]))
                data = data[index + len(tag):]
                self.thinking = not self.thinking
                continue
            keep = next((n for n in range(min(len(tag) - 1, len(data)), 0, -1)
                         if tag.startswith(data[-n:].lower())), 0)
            if keep:
                self._pending, data = data[-keep:], data[:-keep]
            if data:
                out.append(('thinking' if self.thinking else 'text', data))
            break
        return out

    def flush(self) -> list[tuple[str, str]]:
        pending, self._pending = self._pending, ''
        return [('thinking' if self.thinking else 'text', pending)] if pending else []


def _tool_ok(content: str) -> bool:
    return not content.startswith(('ERROR', 'NOT shown', 'error:', 'Not applied'))


class Transcript:
    """One conversation as JSON lines (messages, tool calls, approvals, usage), for the lab
    record of how a script came about. The API key is never written."""

    def __init__(self, folder: str | None) -> None:
        self.folder = folder
        self.path: str | None = None
        self._lock = threading.Lock()

    def start(self, config: AssistantConfig) -> None:
        self.path = None
        if not self.folder:
            return
        os.makedirs(self.folder, exist_ok=True)
        self.path = os.path.join(self.folder, f'chat_{time.strftime("%Y%m%d_%H%M%S")}_{os.getpid()}.jsonl')
        self.write('start', base_url=config.base_url, model=config.model, setup=config.setup)

    def write(self, kind: str, **fields) -> None:
        if not self.path:
            return
        record = {'time': time.strftime('%Y-%m-%dT%H:%M:%S'), 'type': kind, **fields}
        try:
            with self._lock, open(self.path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, default=str) + '\n')
        except OSError:
            logging.exception('assistant: writing the transcript failed')


class ChatSession:
    """A chat with the model, run by an ox AgentServer on a background event loop, with
    streamed replies.

    send() returns immediately; on_event is called from the background thread for each
    ChatEvent, so a GUI must hand them to its own thread. One message at a time: wait for
    'done', or stop() the running turn. Calls that need the user's approval arrive as
    'approval' events; the turn waits until answer() is called. With transcript_dir set,
    every conversation is saved there as a .jsonl file.
    """

    def __init__(self, config: AssistantConfig, bridge: AppBridge,
                 on_event: Callable[[ChatEvent], None], stream: bool = True,
                 transcript_dir: str | None = None) -> None:
        self.config = config
        self.on_event = on_event
        self.stream = stream
        self.busy = False
        self.usage = {'prompt': 0, 'completion': 0, 'cost': None}
        self.transcript = Transcript(transcript_dir)
        self.memory = MemoryStore(config.memory_path())
        self._sessionIds = itertools.count(1)
        self._session = next(self._sessionIds)
        self._started = False             # system prompt sent in this session
        self._split = ThinkSplitter()
        self._toolNames: dict[str, str] = {}
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, args=(bridge,), daemon=True,
                                        name='assistant-chat')
        self._thread.start()
        self._ready.wait()

    def _run(self, bridge: AppBridge) -> None:
        asyncio.set_event_loop(self._loop)
        tools = ToolResolver(builtins=False)
        _register_tools(tools, bridge)
        _register_memory_tools(tools, self.memory)
        self._provider = HttpProvider(self.config.base_url, self.config.resolved_key())
        self._inbox: AsyncPort[Envelope] = AsyncPort()
        self._replies: AsyncPort[Msg] = AsyncPort()
        model = ModelSpec(id=self.config.model.strip(), temperature=0.2, max_tokens=4096)
        self._server = AgentServer(self._inbox, self._provider, model, tools, persistent=True,
                                   compactor_model=model.id, keep_reasoning=False)
        self._loop.create_task(self._server.run())
        self._loop.create_task(self._readReplies())
        self._ready.set()
        try:
            self._loop.run_forever()
        finally:
            self._loop.close()

    async def _readReplies(self) -> None:
        async for msg in self._replies:
            if msg.session_id != self._session:
                continue                  # reply to a chat that was reset
            meta = msg.metadata or {}
            request = meta.get('approval_request')
            if request:
                self.transcript.write('approval_request', **request)
                self._emit(ChatEvent('approval', request.get('details', ''), call_id=request['call_id'],
                                     name=request['name'], data=dict(request.get('arguments') or {})))
            for m in msg.data:
                if isinstance(m, ToolMessage):
                    name = self._toolNames.get(m.tool_call_id, '')
                    self.transcript.write('tool_result', call_id=m.tool_call_id, name=name, content=m.content)
                    self._emit(ChatEvent('result', m.content, call_id=m.tool_call_id, name=name,
                                         ok=_tool_ok(m.content)))
                elif isinstance(m, AssistantMessage):
                    self._onAssistant(m, meta)
            usage = meta.get('usage')
            if usage is not None and (meta.get('stream_done') or not meta.get('stream_delta')):
                self._addUsage(usage)
            if meta.get('turn_done'):
                self.busy = False
                if meta.get('cancelled'):
                    self.transcript.write('stopped')
                self._emit(ChatEvent('done', cancelled=bool(meta.get('cancelled'))))

    def _addUsage(self, usage) -> None:
        self.usage['prompt'] += int(getattr(usage, 'prompt_tokens', 0) or 0)
        self.usage['completion'] += int(getattr(usage, 'completion_tokens', 0) or 0)
        cost = getattr(usage, 'cost', None)
        if cost is not None:
            self.usage['cost'] = (self.usage['cost'] or 0.0) + float(cost)
        self.transcript.write('usage', prompt_tokens=getattr(usage, 'prompt_tokens', 0),
                              completion_tokens=getattr(usage, 'completion_tokens', 0), cost=cost)
        self._emit(ChatEvent('usage', data=dict(self.usage)))

    def _onAssistant(self, m: AssistantMessage, meta: dict) -> None:
        text = m.content if isinstance(m.content, str) else \
            ''.join(getattr(p, 'text', '') for p in (m.content or []))
        if meta.get('reasoning_delta'):
            self._emit(ChatEvent('thinking', m.reasoning or ''))
            return
        if meta.get('stream_delta'):
            for kind, piece in self._split.feed(text or ''):
                self._emit(ChatEvent(kind, piece))
            return
        if _ERROR.match(text or ''):
            self._split = ThinkSplitter()
            self.transcript.write('error', content=text)
            self._emit(ChatEvent('error', _error_text(text)))
            return
        # A complete message: the end of a stream, or a non-streamed reply.
        if meta.get('stream_done'):
            for kind, piece in self._split.flush():
                self._emit(ChatEvent(kind, piece))
        else:
            if m.reasoning or m.reasoning_content:
                self._emit(ChatEvent('thinking', m.reasoning or m.reasoning_content))
            for kind, piece in ThinkSplitter().feed(text or ''):
                if kind == 'thinking':
                    self._emit(ChatEvent('thinking', piece))
        self._split = ThinkSplitter()
        answer = clean_reply(text or '')
        if answer:
            self.transcript.write('assistant', content=answer)
        self._emit(ChatEvent('message', answer))
        for call in m.tool_calls or []:
            name = call.function.name if call.function else '?'
            self._toolNames[call.id] = name
            arguments = call.function.arguments if call.function else ''
            self.transcript.write('tool_call', call_id=call.id, name=name, arguments=arguments)
            self._emit(ChatEvent('tool', arguments, call_id=call.id, name=name))

    def _emit(self, event: ChatEvent) -> None:
        try:
            self.on_event(event)
        except Exception:
            logging.exception('assistant chat event handler failed')

    def send(self, text: str) -> None:
        """Send the user's message. Raises AssistantError if it cannot be sent now."""
        text = text.strip()
        if not text:
            raise AssistantError('Write a message first.')
        if self.busy:
            raise AssistantError('Wait for the answer to the previous message.')
        self.config.check()
        data = [UserMessage(content=text)]
        if not self._started:
            data.insert(0, SystemMessage(content=system_prompt(self.config.setup, self.memory.prompt_section())))
            self._started = True
            self.transcript.start(self.config)
        self.transcript.write('user', content=text)
        self.busy = True
        self._split = ThinkSplitter()
        envelope = Envelope(msg=Msg(session_id=self._session, data=data,
                                    metadata={'stream': True} if self.stream else None),
                            reply=self._replies)
        asyncio.run_coroutine_threadsafe(self._inbox.send(envelope), self._loop)

    def answer(self, call_id: str, approved: bool, arguments: dict | None = None, note: str = '') -> None:
        """The user's decision on an 'approval' event. `arguments`: the edited arguments, if any."""
        self.transcript.write('approval', call_id=call_id, approved=approved, edited=arguments is not None,
                              arguments=arguments, note=note)
        approval = Approval(approved=approved, arguments=arguments, note=note)
        asyncio.run_coroutine_threadsafe(
            self._server.answer_approval(self._session, call_id, approval), self._loop)

    def stop(self) -> None:
        """Stop the running answer; 'done' with cancelled=True follows."""
        if self.busy:
            asyncio.run_coroutine_threadsafe(self._server.cancel(self._session), self._loop)

    def reset(self) -> None:
        """Start a new conversation. A pending answer to the old one is stopped and ignored."""
        self.stop()
        self._session = next(self._sessionIds)
        self._started = False
        self.busy = False
        self.usage = {'prompt': 0, 'completion': 0, 'cost': None}

    def close(self) -> None:
        async def shutdown():
            # Also the per-session workers the AgentServer started, not only our own tasks.
            tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self._provider.close()
            self._loop.stop()
        if self._loop.is_running():
            asyncio.run_coroutine_threadsafe(shutdown(), self._loop)
        self._thread.join(timeout=5)


# --- model list -----------------------------------------------------------------------------

def list_models(config: AssistantConfig) -> list[str]:
    key = config.resolved_key()
    if not key:
        raise AssistantError('No API key. Enter it in Settings > AI assistant, or set the '
                             'GLOWTRACKER_LLM_API_KEY environment variable.')
    url = config.base_url.rstrip('/') + '/models'
    req = urllib.request.Request(url, headers={'Authorization': f'Bearer {key}',
                                               'Accept': 'application/json'})
    try:
        with urllib.request.urlopen(req, timeout=config.timeout_s) as response:
            reply = json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        raise AssistantError(f'The API returned HTTP {e.code}{_HINTS.get(e.code, "")}.') from e
    except urllib.error.URLError as e:
        raise AssistantError(f'Could not reach {url}: {e.reason}') from e
    except TimeoutError as e:
        raise AssistantError(f'No answer from {url} within {config.timeout_s:.0f} s') from e
    except json.JSONDecodeError as e:
        raise AssistantError(f'The API at {url} did not return JSON') from e
    return sorted(m.get('id', '') for m in reply.get('data', []) if m.get('id'))
