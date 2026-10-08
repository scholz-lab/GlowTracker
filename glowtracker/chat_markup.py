"""Render the Markdown that chat models write as Kivy label markup.

Covers what assistants actually use: paragraphs, headings, bullet and numbered lists, **bold**,
*italic*, `inline code`, fenced code blocks and tables. Code blocks and tables are returned
separately so the GUI can show them in a selectable monospace box. Markers that are not closed yet
(a reply still streaming) are hidden rather than shown as asterisks. No Kivy import, so it can be
tested on its own.
"""
from __future__ import annotations

import re

MONO = 'RobotoMono-Regular'
CODE_COLOR = 'f0c27b'

_FENCE = re.compile(r'^[ \t]*```[ \t]*([\w+.-]*)[ \t]*$', re.MULTILINE)
_HEADING = re.compile(r'^(#{1,6})\s+(.*)$')
_BULLET = re.compile(r'^(\s*)[-*+]\s+(.*)$')
_NUMBERED = re.compile(r'^(\s*)(\d+)[.)]\s+(.*)$')
_BOLD_ITALIC = re.compile(r'(\*\*\*|___)(?=\S)(.+?)(?<=\S)\1')
_BOLD = re.compile(r'(\*\*|__)(?=\S)(.+?)(?<=\S)\1')
_ITALIC = re.compile(r'(?<![\w*])([*_])(?=\S)(.+?)(?<=\S)\1(?![\w*])')
_CODE = re.compile(r'`([^`\n]+)`')
_TABLE_ROW = re.compile(r'^\s*\|.*\|\s*$')
_TABLE_RULE = re.compile(r'^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$')


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
    return _split_tables(out)


def _split_tables(parts: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Pull Markdown tables out of the text parts as aligned monospace ('code') parts."""
    out: list[tuple[str, str]] = []
    for kind, content in parts:
        if kind != 'text' or '|' not in content:
            out.append((kind, content))
            continue
        lines, block = [], []

        def flush_text():
            if lines and '\n'.join(lines).strip():
                out.append(('text', '\n'.join(lines).strip('\n')))
            lines.clear()

        for line in content.split('\n') + ['']:
            if _TABLE_ROW.match(line):
                block.append(line)
                continue
            if len(block) >= 2:
                flush_text()
                out.append(('code', _table(block)))
            else:
                lines.extend(block)
            block = []
            lines.append(line)
        flush_text()
    return out


def _table(rows: list[str]) -> str:
    """A Markdown table as plain aligned text. Rows arrive already converted to markup, so the
    markup is removed again for the monospace box."""
    cells = []
    for row in rows:
        if _TABLE_RULE.match(_plain(row)):
            continue
        cells.append([_plain(c).strip() for c in row.strip().strip('|').split('|')])
    widths = [max(len(r[i]) if i < len(r) else 0 for r in cells) for i in range(max(map(len, cells)))]
    out = []
    for n, row in enumerate(cells):
        out.append('  '.join(c.ljust(widths[i]) for i, c in enumerate(row)).rstrip())
        if n == 0 and len(cells) > 1:
            out.append('  '.join('-' * w for w in widths))
    return '\n'.join(out)


def _plain(markup: str) -> str:
    text = re.sub(r'\[/?(?:b|i|font|color)(?:=[^\]]*)?\]', '', markup)
    return text.replace('&bl;', '[').replace('&br;', ']').replace('&amp;', '&')


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
    # Inline code is set aside first (so ** or * inside it stay literal), but bold and italic are
    # applied to the whole line, so **`code` in bold** works.
    codes: list[str] = []

    def keep(match: re.Match) -> str:
        codes.append(f'[font={MONO}][color={CODE_COLOR}]{escape(match.group(1))}[/color][/font]')
        return f'\x00{len(codes) - 1}\x00'

    line = escape(_CODE.sub(keep, text))
    line = _BOLD_ITALIC.sub(r'[b][i]\2[/i][/b]', line)
    line = _BOLD.sub(r'[b]\2[/b]', line)
    line = _ITALIC.sub(r'[i]\2[/i]', line)
    line = re.sub(r'(?<!\*)\*\*\*?(?=\S)|(?<=\S)\*\*\*?(?!\*)', '', line)   # unclosed (still streaming)
    line = line.replace('`', '')
    return re.sub('\x00(\\d+)\x00', lambda m: codes[int(m.group(1))], line)
