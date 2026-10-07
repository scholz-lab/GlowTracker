"""Write DAQ sequencer scripts and plugins from a plain-language description, using any
OpenAI-compatible chat API (e.g. the GWDG / Max Planck "Chat AI" service, or OpenAI itself).

No GUI code here, so it can be tested and reused. Nothing generated is ever executed: sequencer
scripts are checked with the real parser, plugins are only syntax-checked, and the user reviews
the result before using it.

Settings (Settings > AI assistant): base URL, model, API key. If the key is empty, the
GLOWTRACKER_LLM_API_KEY or OPENAI_API_KEY environment variable is used.
"""
from __future__ import annotations

import ast
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field

DEFAULT_BASE_URL = 'https://chat-ai.academiccloud.de/v1'
KEY_ENV_VARS = ('GLOWTRACKER_LLM_API_KEY', 'OPENAI_API_KEY')

SEQUENCER = 'sequencer'
PLUGIN = 'plugin'


class AssistantError(Exception):
    """A problem talking to the API, worded for the user."""


@dataclass
class AssistantConfig:
    base_url: str = DEFAULT_BASE_URL
    model: str = ''
    api_key: str = ''
    timeout_s: float = 120.0

    def resolved_key(self) -> str:
        if self.api_key.strip():
            return self.api_key.strip()
        for name in KEY_ENV_VARS:
            value = os.environ.get(name, '').strip()
            if value:
                return value
        return ''


@dataclass
class Result:
    code: str = ''
    explanation: str = ''
    error: str = ''              # validation error left after the repair attempts, '' if valid
    attempts: int = 0
    transcript: list = field(default_factory=list)


# --- prompts --------------------------------------------------------------------------------

SEQUENCER_GUIDE = """\
You write DAQ sequencer scripts for GlowTracker, a tracking microscope for C. elegans. The script
drives a LabJack DAQ whose two analog outputs (DAC0 and DAC1, always set together) control an
optogenetic LED or other device. The sequence starts when the user presses Record.

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
- If the user gives a duration in seconds but asks for frame mode, ask yourself whether time mode
  is simpler; prefer time mode unless they explicitly want frame-exact timing.

Example (1 s of 4.5 V light at 10 s and at 30 s):
mode: [time]
0: [off]
10: [on, 4.5]
11: [off]
30: [on, 4.5]
31: [off]
"""

PLUGIN_RULES = """\
You write GlowTracker plugins: one Python file that the app loads and calls once per camera frame
with the tracked worm's state and a `scope` handle to set the DAQ outputs. The API reference
follows. Rules:
- Use a Controller class or module-level setup/update/teardown functions exactly as documented.
- update() must return quickly: never sleep, block or loop waiting; keep state in variables and
  count frames or compare state.wall_time instead.
- Only the standard library (math, random, collections, json, time) and numpy may be imported.
- Voltages are 0 to 4.95 V. If the hardware is active at 0 V (e.g. a buzzer that is quiet at
  4.5 V), set idle_voltage as documented.
- Never move the stage while tracking. Log the events that matter with scope.log(...).
- Keep it short, readable and commented for a biologist.
"""

OUTPUT_FORMAT = """\
Reply with exactly one fenced code block containing the complete {what}, followed by an
explanation of 2-5 short sentences in plain language: what it does, the timing, and any
assumption you made. If the request is ambiguous, make a sensible choice and state it.
"""


def _plugin_reference() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    readme = os.path.join(here, '..', 'examples', 'plugins', 'README.md')
    try:
        with open(readme, encoding='utf-8') as f:
            return f.read()
    except OSError:
        return ('Plugin API: define update(state, scope). state.worm_xy, state.trail, state.fps, '
                'state.is_recording, state.wall_time; scope.set_voltage(v, channel=None), '
                'scope.light_off(), scope.log(**fields), scope.print(text).')


def system_prompt(kind: str) -> str:
    if kind == SEQUENCER:
        return SEQUENCER_GUIDE + '\n' + OUTPUT_FORMAT.format(what='sequencer script')
    if kind == PLUGIN:
        return (PLUGIN_RULES + '\n--- API reference ---\n' + _plugin_reference() + '\n--- end ---\n\n'
                + OUTPUT_FORMAT.format(what='Python plugin file'))
    raise ValueError(f'unknown kind {kind!r}')


def user_prompt(request: str, current: str | None = None) -> str:
    request = request.strip()
    if current and current.strip():
        return f'Current script:\n```\n{current.strip()}\n```\n\nChange it as follows: {request}'
    return request


# --- API ------------------------------------------------------------------------------------

def _request(config: AssistantConfig, path: str, payload: dict | None = None):
    key = config.resolved_key()
    if not key:
        raise AssistantError('No API key. Enter it in Settings > AI assistant, or set the '
                             'GLOWTRACKER_LLM_API_KEY environment variable.')
    url = config.base_url.rstrip('/') + path
    data = None if payload is None else json.dumps(payload).encode('utf-8')
    req = urllib.request.Request(url, data=data, method='GET' if payload is None else 'POST')
    req.add_header('Authorization', f'Bearer {key}')
    req.add_header('Content-Type', 'application/json')
    req.add_header('Accept', 'application/json')
    try:
        with urllib.request.urlopen(req, timeout=config.timeout_s) as response:
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        detail = ''
        try:
            detail = e.read().decode('utf-8', 'replace')[:300]
        except Exception:
            pass
        hint = {401: ' (check the API key)', 403: ' (the key has no access to this model)',
                404: ' (check the base URL and model name)', 429: ' (rate limit, try again shortly)'}.get(e.code, '')
        raise AssistantError(f'The API returned HTTP {e.code}{hint}. {detail}'.strip()) from e
    except urllib.error.URLError as e:
        raise AssistantError(f'Could not reach {url}: {e.reason}') from e
    except TimeoutError as e:
        raise AssistantError(f'No answer from {url} within {config.timeout_s:.0f} s') from e
    except json.JSONDecodeError as e:
        raise AssistantError(f'The API at {url} did not return JSON') from e


def list_models(config: AssistantConfig) -> list[str]:
    reply = _request(config, '/models')
    return sorted(m.get('id', '') for m in reply.get('data', []) if m.get('id'))


def chat(config: AssistantConfig, messages: list[dict]) -> str:
    if not config.model.strip():
        raise AssistantError('No model chosen. Pick one in Settings > AI assistant '
                             '(the Models button lists what your key can use).')
    reply = _request(config, '/chat/completions', {
        'model': config.model.strip(),
        'messages': messages,
        'temperature': 0.2,
    })
    try:
        content = reply['choices'][0]['message']['content']
    except (KeyError, IndexError, TypeError) as e:
        raise AssistantError(f'Unexpected reply from the API: {str(reply)[:300]}') from e
    return content or ''


# --- parsing and validation -------------------------------------------------------------------

_THINK = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)
_FENCE = re.compile(r'```[ \t]*([\w+-]*)[ \t]*\n(.*?)```', re.DOTALL)


def extract(reply: str) -> tuple[str, str]:
    """Split a reply into (code, explanation). Reasoning models' <think> blocks are dropped."""
    reply = _THINK.sub('', reply).strip()
    match = _FENCE.search(reply)
    if not match:
        return reply.strip(), ''
    code = match.group(2).strip()
    explanation = (reply[:match.start()] + reply[match.end():]).strip()
    return code, explanation


def validate(kind: str, code: str) -> str:
    """Return '' if the script is usable, otherwise the problem in one line."""
    if not code.strip():
        return 'the reply contained no script'
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


def generate(config: AssistantConfig, kind: str, request: str, current: str | None = None,
             max_repairs: int = 1) -> Result:
    """Ask the model for a script; if it does not validate, send the error back up to
    `max_repairs` times. API problems raise AssistantError."""
    if not request.strip():
        raise AssistantError('Describe what the script should do first.')
    messages = [{'role': 'system', 'content': system_prompt(kind)},
                {'role': 'user', 'content': user_prompt(request, current)}]
    result = Result()
    for attempt in range(max_repairs + 1):
        reply = chat(config, messages)
        result.attempts = attempt + 1
        result.transcript.append(reply)
        result.code, result.explanation = extract(reply)
        result.error = validate(kind, result.code)
        if not result.error:
            break
        messages += [{'role': 'assistant', 'content': reply},
                     {'role': 'user', 'content': f'That script is not valid: {result.error}. '
                                                 f'Reply with the corrected complete script in the same format.'}]
    return result
