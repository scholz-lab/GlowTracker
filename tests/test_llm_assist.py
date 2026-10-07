"""llm_assist against a local fake OpenAI-compatible server (no network, no real key)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

import llm_assist as la


class FakeAPI:
    def __init__(self, replies, status=200):
        self.replies = list(replies)
        self.status = status
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
                content = api.replies.pop(0)
                self._send(200, {'choices': [{'message': {'role': 'assistant', 'content': content}}]})

        self.server = HTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return la.AssistantConfig(base_url=f'http://127.0.0.1:{self.server.server_port}/v1',
                                  model='model-a', api_key='secret', timeout_s=5)

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


GOOD_SEQ = 'Here you go.\n```\nmode: [time]\n0: [off]\n10: [on, 4.5]\n11: [off]\n```\nOne 1 s pulse at 10 s.'
BAD_SEQ = '```\nmode: [time]\n10: [on, 9.0]\n```\nToo bright.'


def test_sequencer_request_and_reply():
    api = FakeAPI([GOOD_SEQ])
    with api as config:
        result = la.generate(config, la.SEQUENCER, '1 s of light at 10 s')
    assert result.error == '' and result.attempts == 1
    assert result.code.startswith('mode: [time]') and '10: [on, 4.5]' in result.code
    assert result.explanation.startswith('Here you go.') and 'One 1 s pulse' in result.explanation
    method, path, headers, body = api.requests[0]
    assert (method, path) == ('POST', '/v1/chat/completions')
    assert headers['Authorization'] == 'Bearer secret'
    assert body['model'] == 'model-a'
    assert body['messages'][0]['role'] == 'system' and 'mode: [frame]' in body['messages'][0]['content']
    assert body['messages'][1]['content'] == '1 s of light at 10 s'


def test_invalid_script_is_sent_back_once_for_repair():
    api = FakeAPI([BAD_SEQ, GOOD_SEQ])
    with api as config:
        result = la.generate(config, la.SEQUENCER, 'pulse at 10 s')
    assert result.attempts == 2 and result.error == ''
    repair = api.requests[1][3]['messages'][-1]['content']
    assert 'not valid' in repair and '4.95' in repair


def test_still_invalid_after_repair_is_reported_not_hidden():
    api = FakeAPI([BAD_SEQ, BAD_SEQ])
    with api as config:
        result = la.generate(config, la.SEQUENCER, 'pulse at 10 s')
    assert result.attempts == 2 and '4.95' in result.error


def test_plugin_validation_and_think_tags():
    reply = '<think>let me plan</think>\n```python\ndef setup(scope):\n    pass\n```\nNo update.'
    fixed = '```python\ndef update(state, scope):\n    scope.set_voltage(0.0)\n```\nKeeps the light off.'
    api = FakeAPI([reply, fixed])
    with api as config:
        result = la.generate(config, la.PLUGIN, 'keep the light off')
    assert result.attempts == 2 and result.error == ''
    assert 'def update' in result.code and 'think' not in result.explanation
    assert 'idle_voltage' in api.requests[0][3]['messages'][0]['content']     # README reference included


def test_editing_an_existing_script_sends_it_along():
    api = FakeAPI([GOOD_SEQ])
    with api as config:
        la.generate(config, la.SEQUENCER, 'move the pulse to 20 s', current='mode: [time]\n10: [on, 4.5]')
    assert 'Current script' in api.requests[0][3]['messages'][1]['content']


def test_bad_key_gives_a_readable_error():
    api = FakeAPI([], status=401)
    with api as config:
        with pytest.raises(la.AssistantError, match='check the API key'):
            la.generate(config, la.SEQUENCER, 'anything')


def test_list_models_and_key_from_environment(monkeypatch):
    api = FakeAPI([])
    with api as config:
        config.api_key = ''
        monkeypatch.setenv('OPENAI_API_KEY', 'from-env')
        assert la.list_models(config) == ['model-a', 'model-b']
        assert api.requests[0][2]['Authorization'] == 'Bearer from-env'


def test_missing_key_and_model_are_explained(monkeypatch):
    for name in la.KEY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(la.AssistantError, match='No API key'):
        la.chat(la.AssistantConfig(model='m'), [])
    with pytest.raises(la.AssistantError, match='No model'):
        la.chat(la.AssistantConfig(api_key='k'), [])
