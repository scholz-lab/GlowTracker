"""Markdown from chat models rendered as Kivy markup."""
import chat_markup as cm


def test_inline_styles_and_escaping():
    [(kind, markup)] = cm.segments('This is **bold**, *italic*, `a*b*` and [x], 2*3*4.')
    assert kind == 'text'
    assert '[b]bold[/b]' in markup and '[i]italic[/i]' in markup
    assert f'[font={cm.MONO}]' in markup and 'a*b*' in markup      # no styling inside code
    assert '&bl;x&br;' in markup                                   # user brackets are not markup
    assert '2*3*4' in markup


def test_lists_and_headings():
    [(_, markup)] = cm.segments('# Plan\n- one\n  - two\n1. first')
    assert markup.splitlines() == ['[b]Plan[/b]', '  • one', '      • two', '  1. first']


def test_code_blocks_are_separate_and_raw():
    text = 'Here:\n```\nmode: [time]\n0: [off]\n```\nThat is all.'
    assert cm.segments(text) == [('text', 'Here:'), ('code', 'mode: [time]\n0: [off]'),
                                 ('text', 'That is all.')]


def test_unclosed_fence_while_streaming_is_code_so_far():
    assert cm.segments('Plugin:\n```python\ndef update(state, scope):') == [
        ('text', 'Plugin:'), ('code', 'def update(state, scope):')]


def test_symbols_the_font_cannot_draw_are_replaced():
    [(_, markup)] = cm.segments('Go to DAQ \u2192 Plugin \u2705')
    assert markup == 'Go to DAQ > Plugin (ok)'


def test_bold_around_inline_code_and_bold_italic():
    [(_, markup)] = cm.segments('**Press `Record`** then ***go***')
    assert markup.startswith('[b]Press [font=') and markup.endswith('[b][i]go[/i][/b]')
    assert '*' not in markup


def test_markers_not_closed_yet_are_hidden_while_streaming():
    [(_, markup)] = cm.segments('Light at **4.5')
    assert markup == 'Light at 4.5'


def test_tables_become_aligned_monospace_blocks():
    parts = cm.segments('Pulses:\n| t (s) | **V** |\n|---|:-:|\n| 10 | 4.5 |\nDone.')
    assert parts == [('text', 'Pulses:'), ('code', 't (s)  V\n-----  ---\n10     4.5'), ('text', 'Done.')]
