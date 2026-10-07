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
