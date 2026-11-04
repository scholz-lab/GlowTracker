"""Earlier conversations: listing them, continuing them, switching model, attached files."""
import base64
import json
import time
import zipfile

import numpy as np
import pytest
from PIL import Image

import assistant_chats as chats
import assistant_documents as documents
import llm_assist as la
from test_llm_assist import GOOD, Chat, FakeAPI, call, text


def records(*items):
    return [dict(type=kind, **fields) for kind, fields in items]


def roles(messages):
    return [type(m).__name__.replace('Message', '').lower() for m in messages]


# --- continuing a chat ----------------------------------------------------------------------

def test_a_saved_chat_continues_with_what_was_said(tmp_path):
    api = FakeAPI([call('propose_sequencer_script', script=GOOD), text('Saved it.'), text('Hello.'),
                   text('Yes, the pulse at 10 s.')])
    with api as config, Chat(config, transcript_dir=str(tmp_path)) as chat:
        chat.say('pulse at 10 s')
        [path] = [str(p) for p in tmp_path.glob('*.jsonl')]
        chat.session.reset()
        chat.say('another chat')                    # its own file
        chat.session.resume(path, chats.read(path))
        assert chat.session.usage['prompt'] > 0     # the old chat's tokens, not the other chat's
        chat.say('what did we do?')
    body = api.bodies()[3]
    assert [m['role'] for m in body['messages']] == ['system', 'user', 'assistant', 'tool', 'assistant', 'user']
    assert body['messages'][1]['content'] == 'pulse at 10 s'
    assert body['messages'][2]['tool_calls'][0]['type'] == 'function'
    assert body['messages'][3]['tool_call_id'] == body['messages'][2]['tool_calls'][0]['id']
    assert body['messages'][-1]['content'] == 'what did we do?'
    written = [r for r in chats.read(path) if r['type'] == 'user']
    assert [r['content'] for r in written] == ['pulse at 10 s', 'what did we do?']     # same file
    assert len(list(tmp_path.glob('*.jsonl'))) == 2


def test_tool_calls_without_a_result_are_left_out():
    history = chats.history(records(
        ('user', {'content': 'pulse'}),
        ('assistant', {'content': 'Checking.'}),
        ('tool_call', {'call_id': 'a', 'name': 'get_app_state', 'arguments': ''}),
        ('tool_call', {'call_id': 'b', 'name': 'propose_plugin', 'arguments': '{}'}),
        ('tool_result', {'call_id': 'a', 'content': '{}'})))       # the window closed during b
    assert roles(history) == ['user', 'assistant', 'tool', 'assistant']
    assert [c.id for c in history[1].tool_calls] == ['a'] and history[1].tool_calls[0].function.arguments == '{}'
    assert history[-1].content == chats.NO_ANSWER


def test_a_stopped_turn_is_rebuilt_as_the_running_chat_had_it():
    history = chats.history(records(
        ('user', {'content': 'one'}),
        ('tool_call', {'call_id': 'a', 'name': 'x', 'arguments': '{}'}),
        ('tool_result', {'call_id': 'a', 'content': 'ok'}),
        ('stopped', {}),
        ('user', {'content': 'two'}),
        ('error', {'content': '[LLM Error 500]: boom'}),
        ('user', {'content': 'three'}),
        ('assistant', {'content': 'done'})))
    assert roles(history) == ['user', 'assistant', 'user', 'assistant', 'user', 'assistant']
    assert history[1].content == chats.STOPPED and history[3].content == chats.NO_ANSWER


def test_list_shows_chats_newest_first_and_skips_broken_ones(tmp_path):
    def chat(name, first, age):
        path = tmp_path / name
        path.write_text('\n'.join(json.dumps(r) for r in records(
            ('start', {'model': 'model-a'}), ('user', {'content': first}))) + '\n')
        stamp = time.time() - age
        import os
        os.utime(path, (stamp, stamp))
    chat('chat_old.jsonl', 'old one', 500)
    chat('chat_new.jsonl', 'x ' * 100, 10)
    (tmp_path / 'chat_empty.jsonl').write_text(json.dumps({'type': 'start'}) + '\n')
    (tmp_path / 'chat_broken.jsonl').write_text('{not json\n')
    (tmp_path / 'notes.txt').write_text('no')
    listed = chats.list_chats(str(tmp_path))
    assert [c.path.split('chat_')[-1] for c in listed] == ['new.jsonl', 'old.jsonl']
    assert len(listed[0].title) <= chats.TITLE_CHARS and listed[1].title == 'old one' and listed[1].model == 'model-a'
    assert chats.list_chats(str(tmp_path / 'nothing here')) == []


def test_two_chats_in_the_same_second_get_their_own_files(tmp_path, monkeypatch):
    monkeypatch.setattr(la.time, 'strftime', lambda *args: '20261104_120000')
    first, second = la.Transcript(str(tmp_path)), la.Transcript(str(tmp_path))
    first.start(la.AssistantConfig())
    second.start(la.AssistantConfig())
    assert first.path != second.path and len(list(tmp_path.glob('*.jsonl'))) == 2


def test_a_transcript_cut_off_by_a_crash_can_still_be_read(tmp_path):
    path = tmp_path / 'chat.jsonl'
    path.write_text(json.dumps({'type': 'user', 'content': 'hi'}) + '\n{"type": "assi')
    assert [r['type'] for r in chats.read(str(path))] == ['user']


# --- switching model ------------------------------------------------------------------------

def test_switching_model_keeps_the_conversation(tmp_path):
    api = FakeAPI([text('One.'), text('Two.')])
    with api as config, Chat(config, transcript_dir=str(tmp_path)) as chat:
        chat.say('first')
        chat.session.set_model('model-b')
        chat.say('second')
    first, second = api.bodies()
    assert (first['model'], second['model']) == ('model-a', 'model-b')
    assert [m['role'] for m in second['messages']] == ['system', 'user', 'assistant', 'user']
    [path] = tmp_path.glob('*.jsonl')
    assert any(r['type'] == 'model' and r['model'] == 'model-b' for r in chats.read(str(path)))


def test_the_model_is_not_switched_during_an_answer():
    api = FakeAPI([text('A long answer.')], delay=0.05)
    with api as config, Chat(config) as chat:
        chat.say('hello', wait=False)
        with pytest.raises(la.AssistantError, match='Wait'):
            chat.session.set_model('model-b')
        chat.wait()


# --- attachments ----------------------------------------------------------------------------

def test_text_and_word_documents_are_sent_as_text(tmp_path):
    notes = tmp_path / 'notes.csv'
    notes.write_bytes('strain,n﻿\nN2,5\n'.encode('utf-8'))
    assert documents.load(str(notes))['text'].startswith('strain,n')
    latin = tmp_path / 'old.txt'
    latin.write_bytes('Temperatur 20 \xb0C'.encode('latin-1'))
    assert '\xb0C' in documents.load(str(latin))['text']
    word = tmp_path / 'protocol.docx'
    with zipfile.ZipFile(word, 'w') as z:
        z.writestr('word/document.xml', '<w:document><w:p><w:r><w:t>Wash &amp; dry</w:t></w:r></w:p>'
                                        '<w:p><w:r><w:t xml:space="preserve">twice</w:t></w:r></w:p></w:document>')
    assert documents.load(str(word))['text'].split() == ['Wash', '&', 'dry', 'twice']


def test_pdf_text_is_extracted(tmp_path):
    pypdf = pytest.importorskip('pypdf')
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(300, 300)
    font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'),
                             NameObject('/BaseFont'): NameObject('/Helvetica')})
    page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): font})})
    stream = DecodedStreamObject()
    stream.set_data(b'BT /F1 12 Tf 20 200 Td (Hold DAC0 at 4.5 V) Tj ET')
    page[NameObject('/Contents')] = writer._add_object(stream)
    path = tmp_path / 'paper.pdf'
    writer.write(str(path))
    assert 'Hold DAC0 at 4.5 V' in documents.load(str(path))['text']


def test_a_pdf_without_text_and_other_unreadable_files_are_explained(tmp_path):
    pypdf = pytest.importorskip('pypdf')
    writer = pypdf.PdfWriter()
    writer.add_blank_page(100, 100)
    path = tmp_path / 'scan.pdf'
    writer.write(str(path))
    with pytest.raises(documents.AttachmentError, match='scanned'):
        documents.load(str(path))
    binary = tmp_path / 'data.bin'
    binary.write_bytes(b'\0\1\2')
    with pytest.raises(documents.AttachmentError, match='not a text file'):
        documents.load(str(binary))
    with pytest.raises(documents.AttachmentError, match='Could not read'):
        documents.load(str(tmp_path / 'gone.txt'))


def test_long_documents_are_cut_and_say_so(tmp_path):
    path = tmp_path / 'long.txt'
    path.write_text('x' * (documents.MAX_TEXT_CHARS + 10))
    loaded = documents.load(str(path))
    assert len(loaded['text']) == documents.MAX_TEXT_CHARS and loaded['note']
    assert loaded['note'] in documents.user_content('read this', [loaded])


def test_16_bit_microscope_images_become_visible_jpegs(tmp_path):
    frame = np.zeros((2000, 1000), np.uint16)
    frame[:, 500:] = 4000                   # dim in 16 bits: would look black if only cut to 8 bits
    path = tmp_path / 'frame.tif'
    Image.fromarray(frame).save(path)
    loaded = documents.load(str(path))
    assert loaded['kind'] == 'image' and loaded['url'].startswith('data:image/jpeg;base64,')
    import io
    image = Image.open(io.BytesIO(base64.b64decode(loaded['url'].split(',')[1])))
    assert max(image.size) == documents.MAX_IMAGE_SIDE
    assert np.asarray(image.convert('L'))[:, -1].mean() > 200


def test_attachments_reach_the_model_and_survive_continuing_the_chat(tmp_path):
    document = {'name': 'protocol.txt', 'kind': 'text', 'text': 'Keep at 20 C.', 'note': ''}
    image = {'name': 'plate.png', 'kind': 'image', 'url': 'data:image/jpeg;base64,AAAA', 'note': ''}
    api = FakeAPI([text('Read.'), text('Still know it.')])
    with api as config, Chat(config, transcript_dir=str(tmp_path)) as chat:
        chat._done.clear()
        chat.session.send('what temperature?', [document, image])
        chat.wait()
        [path] = [str(p) for p in tmp_path.glob('*.jsonl')]
        chat.session.resume(path, chats.read(path))
        chat.say('and again?')
    for body in api.bodies():
        content = body['messages'][1]['content']
        assert content[0]['type'] == 'text' and 'Keep at 20 C.' in content[0]['text']
        assert 'what temperature?' in content[0]['text'] and '[Attached image: plate.png]' in content[0]['text']
        assert content[1] == {'type': 'image_url', 'image_url': {'url': image['url']}}


def test_a_document_alone_can_be_sent():
    document = {'name': 'a.txt', 'kind': 'text', 'text': 'body', 'note': ''}
    api = FakeAPI([text('ok')])
    with api as config, Chat(config) as chat:
        chat._done.clear()
        with pytest.raises(la.AssistantError):
            chat.session.send('   ')
        chat.session.send('', [document])
        chat.wait()
    assert 'body' in api.bodies()[0]['messages'][1]['content']
