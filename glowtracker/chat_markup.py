"""Render the Markdown that chat models write as Kivy label markup.

Covers what assistants actually use: paragraphs, headings, bullet and numbered lists, **bold**,
*italic*, `inline code` and fenced code blocks. Code blocks are returned separately so the GUI can
show them in a selectable monospace box. No Kivy import, so it can be tested on its own.
"""
from __future__ import annotations

import re

MONO = 'RobotoMono-Regular'
CODE_COLOR = 'f0c27b'

_FENCE = re.compile(r'^[ \t]*```[ \t]*([\w+.-]*)[ \t]*$', re.MULTILINE)
_HEADING = re.compile(r'^(#{1,6})\s+(.*)$')
_BULLET = re.compile(r'^(\s*)[-*+]\s+(.*)$')
_NUMBERED = re.compile(r'^(\s*)(\d+)[.)]\s+(.*)$')
_BOLD = re.compile(r'(\*\*|__)(?=\S)(.+?)(?<=\S)\1')
_ITALIC = re.compile(r'(?<![\w*])([*_])(?=\S)(.+?)(?<=\S)\1(?![\w*])')


# Symbols models like to use that the default Kivy font (Roboto) cannot draw.
_GLYPHS = str.maketrans({
    '\u2192': '>', '\u2190': '<', '\u21d2': '=>', '\u2194': '<->', '\u279c': '>',
    '\u2713': '(ok)', '\u2714': '(ok)', '\u2705': '(ok)', '\u2717': '(x)', '\u274c': '(x)',
    '\u26a0': '(!)', '\ufe0f': None, '\u2264': '<=', '\u2265': '>=', '\u2248': '~',
})


def escape(text: str) -> str:
    """Same as kivy.utils.escape_markup."""
    return text.replace('&', '&amp;').replace('[', '&bl;').replace(']', '&br;')


def segments(text: str) -> list[tuple[str, str]]:
    """Split into [('text', markup) | ('code', raw code)]. An unclosed fence (a reply still
    streaming) runs to the end of the text."""
    text = text.translate(_GLYPHS)
    out: list[tuple[str, str]] = []
    pos = 0
    fences = list(_FENCE.finditer(text))
    i = 0
    while i < len(fences):
        start = fences[i]
        before = text[pos:start.start()]
        if before.strip():
            out.append(('text', inline_block(before.strip('\n'))))
        end = fences[i + 1] if i + 1 < len(fences) else None
        code_end = end.start() if end else len(text)
        out.append(('code', text[start.end():code_end].strip('\n')))
        pos = end.end() if end else len(text)
        i += 2
    rest = text[pos:]
    if rest.strip():
        out.append(('text', inline_block(rest.strip('\n'))))
    return out


def inline_block(text: str) -> str:
    """Markup for text without code fences, line by line."""
    lines = []
    for line in text.split('\n'):
        if m := _HEADING.match(line):
            lines.append(f'[b]{_inline(m.group(2))}[/b]')
        elif m := _BULLET.match(line):
            indent = '    ' * (len(m.group(1).expandtabs(4)) // 2)
            lines.append(f'{indent}  • {_inline(m.group(2))}')
        elif m := _NUMBERED.match(line):
            indent = '    ' * (len(m.group(1).expandtabs(4)) // 2)
            lines.append(f'{indent}  {m.group(2)}. {_inline(m.group(3))}')
        elif re.fullmatch(r'\s*([-*_])(\s*\1){2,}\s*', line):
            lines.append('[color=666666]———[/color]')
        else:
            lines.append(_inline(line))
    return '\n'.join(lines)


def _inline(text: str) -> str:
    # Inline code first, so ** or * inside it stay literal.
    parts = re.split(r'(`[^`\n]+`)', text)
    out = []
    for part in parts:
        if len(part) > 1 and part.startswith('`') and part.endswith('`'):
            out.append(f'[font={MONO}][color={CODE_COLOR}]{escape(part[1:-1])}[/color][/font]')
        else:
            part = escape(part)
            part = _BOLD.sub(r'[b]\2[/b]', part)
            part = _ITALIC.sub(r'[i]\2[/i]', part)
            out.append(part)
    return ''.join(out)
