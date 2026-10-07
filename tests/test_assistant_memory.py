"""Long-term memory: lessons the assistant proposes and the user approves."""
import json

import pytest

import llm_assist as la
from assistant_memory import MAX_FACT_CHARS, MemoryStore
from test_llm_assist import Chat, FakeAPI, call, text


def test_store_adds_removes_and_persists(tmp_path):
    path = tmp_path / 'sub' / 'memory.json'
    store = MemoryStore(str(path))
    assert store.facts() == [] and '(none yet)' in store.prompt_section()
    a = store.add('DAC0 drives a buzzer that is quiet at 4.5 V.', 'it buzzed during a test')
    b = store.add('Prefer time mode for sequencer scripts.')
    assert (a.id, b.id) == (1, 2)
    again = MemoryStore(str(path))                      # a new process sees the same facts
    assert [f.text for f in again.facts()] == [a.text, b.text]
    assert '- [1] DAC0 drives a buzzer' in again.prompt_section()
    again.remove(1)
    assert [f.id for f in store.facts()] == [2]
    assert store.add('Third.').id == 3                  # ids are not reused
    assert json.loads(path.read_text())['facts'][0]['text'] == 'Prefer time mode for sequencer scripts.'


def test_store_rejects_duplicates_empty_and_long_facts(tmp_path):
    store = MemoryStore(str(tmp_path / 'm.json'))
    store.add('Keep the buzzer quiet.')
    assert 'already remembered as fact 1' in store.problem('keep the   BUZZER quiet')
    assert store.problem('  ') == 'the fact is empty'
    assert 'keep it under' in store.problem('x' * (MAX_FACT_CHARS + 1))
    with pytest.raises(ValueError):
        store.add('Keep the buzzer quiet')
    with pytest.raises(KeyError):
        store.remove(42)


def test_prompt_keeps_the_newest_facts_when_too_many(tmp_path, monkeypatch):
    import assistant_memory
    monkeypatch.setattr(assistant_memory, 'MAX_PROMPT_CHARS', 60)
    store = MemoryStore(str(tmp_path / 'm.json'))
    for i in range(10):
        store.add(f'Lesson number {i} about this microscope.')
    section = store.prompt_section()
    assert 'Lesson number 9' in section and 'Lesson number 0' not in section and 'older facts left out' in section


LESSON = 'DAC0 drives a buzzer; keep it at 4.5 V with a plugin instead of a sequencer script.'


def test_an_approved_lesson_opens_every_later_conversation():
    api = FakeAPI([call('remember', fact=LESSON, why='the sequencer script made the buzzer sound'),
                   text('Noted.'), text('Hello again.')])
    with api as config, Chat(config) as chat:
        chat.say('the buzzer went off the whole time, remember that')
        assert [f.text for f in chat.session.memory.facts()] == [LESSON]
        told = api.bodies()[1]['messages'][-1]['content']
        assert 'Remembered as lesson 1' in told
        chat.session.reset()
        chat.say('new experiment')
    system = api.bodies()[2]['messages'][0]['content']
    assert f'- [1] {LESSON}' in system and 'Lessons remembered on this microscope' in system


def test_declined_and_edited_lessons():
    api = FakeAPI([call('remember', fact='Always use 4.5 V.', why='x'), text('ok'),
                   call('remember', fact='Use 3 V for the LED.', why='y'), text('ok')])
    with api as config, Chat(config) as chat:
        chat.decide = lambda event: (False, None)
        chat.say('one')
        assert chat.session.memory.facts() == []
        chat.decide = lambda event: (True, dict(event.data, fact='Use 3 V for the 590 nm LED unless told otherwise.'))
        chat.say('two')
    assert [f.text for f in chat.session.memory.facts()] == ['Use 3 V for the 590 nm LED unless told otherwise.']
    assert 'after changing: fact' in api.bodies()[3]['messages'][-1]['content']


def test_duplicates_are_refused_without_asking_and_forget_works():
    api = FakeAPI([call('remember', fact=LESSON, why='x'), text('ok'),
                   call('remember', fact=LESSON, why='again'), text('already known'),
                   call('forget', id=1, why='outdated'), text('gone')])
    with api as config, Chat(config) as chat:
        chat.say('remember')
        chat.say('remember again')
        assert len([e for e in chat.events if e.kind == 'approval']) == 1     # the duplicate never reached the user
        assert 'already remembered as fact 1' in api.bodies()[3]['messages'][-1]['content']
        chat.say('forget it')
        [forget] = [e for e in chat.events if e.kind == 'approval' and e.name == 'forget']
        assert LESSON in forget.text
    assert chat.session.memory.facts() == []


def test_memory_location_comes_from_settings(tmp_path):
    config = la.AssistantConfig(memory_file=str(tmp_path / 'shared.json'))
    assert config.memory_path() == str(tmp_path / 'shared.json')
    assert la.AssistantConfig().memory_path() == la.DEFAULT_MEMORY_FILE
