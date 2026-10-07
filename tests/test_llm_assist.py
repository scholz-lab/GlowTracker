"""llm_assist chat against a local fake OpenAI-compatible server (no network, no real key)."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import llm_assist as la


def text(content):
    return {'role': 'assistant', 'content': content}


def call(tool, **args):
    return {'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': f'call_{tool}', 'type': 'function',
         'function': {'name': tool, 'arguments': json.dumps(args)}}]}


def chunks(message):
    """Split a reply message into streaming deltas the way OpenAI-style servers do: text and
    reasoning in pieces, each tool call as a header (id, name) and then its arguments."""
    out = []
    for key in ('reasoning_content', 'content'):
        value = message.get(key) or ''
        for i in range(0, len(value), 7):
            out.append({key: value[i:i + 7]})
    for i, tc in enumerate(message.get('tool_calls') or []):
        out.append({'tool_calls': [{'index': i, 'id': tc['id'], 'type': 'function',
                                    'function': {'name': tc['function']['name'], 'arguments': ''}}]})
        args = tc['function']['arguments']
        half = len(args) // 2
        out.append({'tool_calls': [{'index': i, 'function': {'arguments': args[:half]}}]})
        out.append({'tool_calls': [{'index': i, 'function': {'arguments': args[half:]}}]})
    return out


class FakeAPI:
    def __init__(self, replies, status=200, delay=0.0):
        self.replies = list(replies)
        self.status = status
        self.delay = delay              # seconds between streamed chunks
        self.requests = []
        api = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code, body):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                api.requests.append(('GET', self.path, dict(self.headers), None))
                self._send(api.status, {'data': [{'id': 'model-b'}, {'id': 'model-a'}]})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                api.requests.append(('POST', self.path, dict(self.headers), body))
                if api.status != 200:
                    self._send(api.status, {'error': {'message': 'invalid api key'}})
                    return
                message = api.replies.pop(0)
                if body.get('stream'):
                    self._stream(message)
                    return
                self._send(200, {'id': 'x', 'object': 'chat.completion', 'model': 'model-a',
                                 'choices': [{'index': 0, 'message': message,
                                              'finish_reason': 'stop'}],
                                 'usage': {'prompt_tokens': 10, 'completion_tokens': 5,
                                           'total_tokens': 15}})

            def _stream(self, message):
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                try:
                    for delta in [{'role': 'assistant'}] + chunks(message):
                        self._event({'id': 'x', 'object': 'chat.completion.chunk', 'model': 'model-a',
                                     'choices': [{'index': 0, 'delta': delta}]})
                        time.sleep(api.delay)
                    self._event({'id': 'x', 'object': 'chat.completion.chunk', 'model': 'model-a',
                                 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}],
                                 'usage': {'prompt_tokens': 10, 'completion_tokens': 5, 'total_tokens': 15}})
                    self.wfile.write(b'data: [DONE]\n\n')
                except (BrokenPipeError, ConnectionResetError):
                    pass                    # the client stopped reading

            def _event(self, body):
                self.wfile.write(b'data: ' + json.dumps(body).encode() + b'\n\n')
                self.wfile.flush()

        self.server = HTTPServer(('127.0.0.1', 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def bodies(self):
        return [r[3] for r in self.requests if r[0] == 'POST']

    def __enter__(self):
        self.thread.start()
        return la.AssistantConfig(base_url=f'http://127.0.0.1:{self.server.server_port}/v1',
                                  model='model-a', api_key='secret', timeout_s=5)

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


class FakeApp:
    def __init__(self):
        self.proposals = []

    def app_state(self):
        return {'recording': False, 'daq_mode': 'Sequencer', 'sequencer_script': 'mode: [time]\n5: [on, 2]'}

    def current_plugin(self):
        return 'def update(state, scope):\n    pass\n'

    def propose(self, proposal):
        self.proposals.append(proposal)


class Chat:
    """A ChatSession plus the events it emitted; say() waits for the end of the turn."""

    def __init__(self, config, stream=True):
        self.app = FakeApp()
        self.events = []
        self._done = threading.Event()
        self.session = la.ChatSession(config, self.app, self._onEvent, stream=stream)

    def _onEvent(self, event):
        self.events.append(event)
        if event.kind == 'done':
            self._done.set()

    def say(self, message, wait=True):
        self._done.clear()
        self.session.send(message)
        if wait:
            self.wait()

    def wait(self):
        assert self._done.wait(10), 'no answer'

    def of(self, kind):
        return [e.text for e in self.events if e.kind == kind]

    def replies(self):
        return [t for t in self.of('message') if t]

    def streamed(self):
        return ''.join(self.of('text'))

    def kinds(self):
        """Event kinds in order, with repeated streaming deltas collapsed."""
        out = []
        for e in self.events:
            if not (out and e.kind in ('text', 'thinking') and out[-1] == e.kind):
                out.append(e.kind)
        return out

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.session.close()


GOOD = 'mode: [time]\n0: [off]\n10: [on, 4.5]\n11: [off]'
TOO_BRIGHT = 'mode: [time]\n10: [on, 9.0]'


def test_request_is_valid_openai_chat_with_only_glowtracker_tools():
    api = FakeAPI([text('Hello! What should the light do?')])
    with api as config, Chat(config) as chat:
        chat.say('hi')
    assert chat.replies() == ['Hello! What should the light do?']
    method, path, headers, body = api.requests[0]
    assert (method, path) == ('POST', '/v1/chat/completions')
    assert headers['Authorization'] == 'Bearer secret'
    assert body['model'] == 'model-a'
    assert [m['role'] for m in body['messages']] == ['system', 'user']
    assert 'mode: [frame]' in body['messages'][0]['content']
    assert 'idle_voltage' in body['messages'][0]['content']        # plugin README included
    assert all('type' not in m for m in body['messages'])          # role is the tag, not "type"
    assert body['stream'] is True and body['stream_options'] == {'include_usage': True}
    assert chat.streamed() == 'Hello! What should the light do?'
    names = {t['function']['name'] for t in body['tools']}
    assert names == {'get_app_state', 'read_current_plugin', 'read_plugin_example',
                     'propose_sequencer_script', 'propose_plugin'}


def test_proposal_goes_through_the_parser_and_reaches_the_app():
    api = FakeAPI([call('propose_sequencer_script', script=f'```\n{GOOD}\n```', summary='One pulse'),
                   text('One 1 s pulse at 10 s; press Use in Sequencer.')])
    with api as config, Chat(config) as chat:
        chat.say('1 s of light at 10 s')
    assert [(p.kind, p.code, p.summary) for p in chat.app.proposals] == [(la.SEQUENCER, GOOD, 'One pulse')]
    assert [e.name for e in chat.events if e.kind == 'tool'] == ['propose_sequencer_script']
    assert [(e.name, e.ok) for e in chat.events if e.kind == 'result'] == [('propose_sequencer_script', True)]
    assert chat.replies() == ['One 1 s pulse at 10 s; press Use in Sequencer.']
    tool_result = api.bodies()[1]['messages'][-1]
    assert tool_result['role'] == 'tool' and tool_result['tool_call_id'] == 'call_propose_sequencer_script'
    assert tool_result['content'].startswith('Valid.')


def test_invalid_script_is_returned_to_the_model_not_shown():
    api = FakeAPI([call('propose_sequencer_script', script=TOO_BRIGHT),
                   call('propose_sequencer_script', script=GOOD),
                   text('Fixed: 4.5 V instead of 9 V.')])
    with api as config, Chat(config) as chat:
        chat.say('bright pulse at 10 s')
    assert [p.code for p in chat.app.proposals] == [GOOD]
    feedback = api.bodies()[1]['messages'][-1]['content']
    assert 'NOT shown' in feedback and '4.95' in feedback


def test_plugin_proposal_is_syntax_checked():
    api = FakeAPI([call('propose_plugin', code='def setup(scope):\n    pass\n'),
                   call('propose_plugin', code='def update(state, scope):\n    scope.light_off()\n'),
                   text('Keeps the light off.')])
    with api as config, Chat(config) as chat:
        chat.say('keep the light off')
    assert [p.kind for p in chat.app.proposals] == [la.PLUGIN]
    assert 'neither update' in api.bodies()[1]['messages'][-1]['content']


def test_follow_up_keeps_the_history_and_can_read_the_app_state():
    api = FakeAPI([text('Which voltage?'),
                   call('get_app_state'),
                   text('<think>the script has 2 V</think>Your script switches on 2 V at 5 s.')])
    with api as config, Chat(config) as chat:
        chat.say('change my script')
        chat.say('what does it do now?')
    assert chat.replies() == ['Which voltage?', 'Your script switches on 2 V at 5 s.']
    second = api.bodies()[1]['messages']
    assert [m['role'] for m in second] == ['system', 'user', 'assistant', 'user']
    assert second[2]['content'] == 'Which voltage?'
    state = api.bodies()[2]['messages'][-1]
    assert state['role'] == 'tool' and '5: [on, 2]' in state['content']


def test_reset_starts_a_new_conversation():
    api = FakeAPI([text('one'), text('two')])
    with api as config, Chat(config) as chat:
        chat.say('first')
        chat.session.reset()
        chat.say('second')
    assert [m['role'] for m in api.bodies()[1]['messages']] == ['system', 'user']


def test_bad_key_gives_a_readable_error_and_the_chat_continues():
    api = FakeAPI([], status=401)
    with api as config, Chat(config) as chat:
        chat.say('anything')
        assert len(chat.of('error')) == 1 and 'check the API key' in chat.of('error')[0]
        api.status = 200
        api.replies = [text('ok now')]
        chat.say('again')
    assert chat.replies() == ['ok now']


def test_one_message_at_a_time_and_missing_settings_are_explained(monkeypatch):
    for name in la.KEY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    with Chat(la.AssistantConfig(model='m')) as chat:
        with pytest.raises(la.AssistantError, match='No API key'):
            chat.session.send('hi')
        chat.session.config = la.AssistantConfig(api_key='k')
        with pytest.raises(la.AssistantError, match='No model'):
            chat.session.send('hi')
        chat.session.busy = True
        with pytest.raises(la.AssistantError, match='Wait'):
            chat.session.send('hi')


def test_list_models_and_key_from_environment(monkeypatch):
    api = FakeAPI([])
    with api as config:
        config.api_key = ''
        monkeypatch.setenv('OPENAI_API_KEY', 'from-env')
        assert la.list_models(config) == ['model-a', 'model-b']
        assert api.requests[0][2]['Authorization'] == 'Bearer from-env'


def test_streamed_tool_call_arguments_are_reassembled():
    """Tool calls arrive as a header chunk plus argument pieces without id or name."""
    script = 'mode: [time]\n0: [off]\n10: [on, 4.5]\n11: [off]'
    api = FakeAPI([call('propose_sequencer_script', script=script, summary='One pulse'),
                   text('Done.')])
    with api as config, Chat(config) as chat:
        chat.say('pulse')
    assert [p.code for p in chat.app.proposals] == [script]
    sent_back = api.bodies()[1]['messages'][-2]['tool_calls'][0]
    assert sent_back['id'] == 'call_propose_sequencer_script'
    assert json.loads(sent_back['function']['arguments'])['script'] == script
    assert chat.kinds() == ['message', 'tool', 'result', 'text', 'message', 'done']


def test_thinking_is_separated_from_the_answer_and_not_sent_back():
    api = FakeAPI([text('<think>The user wants one pulse.</think>One pulse at 10 s.'),
                   {'role': 'assistant', 'reasoning_content': 'Plan: reuse it.', 'content': 'Sure.'},
                   text('ok')])
    with api as config, Chat(config) as chat:
        chat.say('pulse')
        chat.say('again')
        chat.say('thanks')
    assert chat.replies() == ['One pulse at 10 s.', 'Sure.', 'ok']
    assert ''.join(chat.of('thinking')) == 'The user wants one pulse.Plan: reuse it.'
    assert '<think>' not in chat.streamed()
    history = api.bodies()[2]['messages']
    assert all('reasoning_content' not in m and 'reasoning' not in m for m in history)


def test_stop_ends_the_turn_and_keeps_a_valid_history():
    api = FakeAPI([text('A very long answer that streams slowly ' * 5), text('Short.')], delay=0.05)
    with api as config, Chat(config) as chat:
        chat.say('tell me everything', wait=False)
        deadline = time.monotonic() + 5
        while not chat.of('text') and time.monotonic() < deadline:
            time.sleep(0.01)
        chat.session.stop()
        chat.wait()
        assert chat.events[-1].kind == 'done' and chat.events[-1].cancelled
        assert not chat.session.busy
        api.delay = 0
        chat.say('short version')
    assert chat.replies()[-1] == 'Short.'
    roles = [m['role'] for m in api.bodies()[1]['messages']]
    assert roles == ['system', 'user', 'assistant', 'user']
    assert 'Stopped' in api.bodies()[1]['messages'][2]['content']


def test_non_streaming_mode_still_works():
    api = FakeAPI([call('get_app_state'), text('<think>hm</think>Your script uses 2 V.')])
    with api as config, Chat(config, stream=False) as chat:
        chat.say('what is loaded?')
    assert not api.bodies()[0].get('stream')
    assert chat.replies() == ['Your script uses 2 V.'] and chat.of('thinking') == ['hm']
    assert [(e.name, e.ok) for e in chat.events if e.kind == 'result'] == [('get_app_state', True)]


def test_think_splitter_handles_tags_split_across_chunks():
    splitter = la.ThinkSplitter()
    out = []
    for piece in ['Hi <th', 'ink>pla', 'n</thi', 'nk>Answer <', 'b>']:
        out += splitter.feed(piece)
    out += splitter.flush()
    joined = {}
    for kind, text in out:
        joined[kind] = joined.get(kind, '') + text
    assert joined == {'text': 'Hi Answer <b>', 'thinking': 'plan'}


def test_the_full_plugin_readme_and_example_list_are_in_the_prompt():
    with open(f'{la.PLUGIN_EXAMPLES_DIR}/README.md', encoding='utf-8') as f:
        readme = f.read()
    prompt = la.system_prompt()
    assert readme in prompt
    examples = la.plugin_examples()
    assert 'template_controller.py' in examples and 'buzzer.py' in examples
    assert 'plugin_log_summary.py' not in examples       # analysis script, not a plugin
    assert all(f'- {name}' in prompt for name in examples)


def test_model_can_read_an_example_plugin():
    api = FakeAPI([call('read_plugin_example', name='template_controller.py'), text('Read it.')])
    with api as config, Chat(config) as chat:
        chat.say('write a plugin')
    result = api.bodies()[1]['messages'][-1]
    assert result['role'] == 'tool' and 'def update' in result['content']
    assert 'no example' in la.read_plugin_example('../../secret.py')


def test_plugin_proposals_are_checked_and_test_run_before_they_are_shown():
    crashes = 'def update(state, scope):\n    if state.is_reversing:\n        scope.set_voltage(1 / 0)\n'
    plugin = 'def update(state, scope):\n    scope.set_voltage(3.0 if state.is_reversing else 0.0)\n'
    api = FakeAPI([call('propose_plugin', code='def update(state, scope):\n    scope.set_voltage(state.worm_speed)\n'),
                   call('propose_plugin', code=crashes),
                   call('propose_plugin', code=plugin, summary='light during reversals'),
                   text('Done.')])
    with api as config, Chat(config) as chat:
        chat.say('light while reversing')
    static = api.bodies()[1]['messages'][-1]['content']
    assert 'NOT shown' in static and 'state.worm_speed does not exist' in static
    crashed = api.bodies()[2]['messages'][-1]['content']
    assert 'NOT shown' in crashed and 'ZeroDivisionError' in crashed and 'line 3' in crashed
    accepted = api.bodies()[3]['messages'][-1]['content']
    assert 'ran without errors' in accepted
    [proposal] = chat.app.proposals
    assert proposal.code == plugin.strip() and 'ran without errors' in proposal.checks
