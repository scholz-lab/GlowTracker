"""Long-term memory of the AI assistant: short facts about this microscope and how to work with
it, learned from mistakes and corrections ("DAC0 drives a buzzer, so never use a sequencer
script when the buzzer must stay quiet").

The model proposes a fact with the `remember` tool; it is stored only after the user approves it
(they can edit it first). Every new conversation starts with the stored facts. The file is plain
JSON, so it can also be read, backed up or shared; a lab can point several machines at one file.
No Kivy import.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from dataclasses import asdict, dataclass

MAX_FACT_CHARS = 400
MAX_PROMPT_CHARS = 6000


@dataclass
class Fact:
    id: int
    text: str
    why: str = ''           # what happened that taught it
    created: str = ''


class MemoryStore:
    """Facts in a JSON file. Thread-safe within a process; writes are atomic (temp file + rename),
    and the file is re-read before each change, so a second window does not lose facts."""

    def __init__(self, path: str):
        self.path = os.path.abspath(os.path.expanduser(path))
        self._lock = threading.Lock()

    def facts(self) -> list[Fact]:
        try:
            with open(self.path, encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            return []
        except (OSError, ValueError) as e:
            raise MemoryFileError(f'cannot read the memory file {self.path}: {e}') from e
        return [Fact(**{k: item.get(k, '') for k in ('id', 'text', 'why', 'created')})
                for item in data.get('facts', []) if isinstance(item, dict) and item.get('text')]

    def problem(self, text: str) -> str:
        """Why `text` should not be stored, or ''."""
        text = text.strip()
        if not text:
            return 'the fact is empty'
        if len(text) > MAX_FACT_CHARS:
            return f'the fact is {len(text)} characters long; keep it under {MAX_FACT_CHARS}'
        for fact in self.facts():
            if _normal(fact.text) == _normal(text):
                return f'this is already remembered as fact {fact.id}'
        return ''

    def add(self, text: str, why: str = '') -> Fact:
        with self._lock:
            problem = self.problem(text)
            if problem:
                raise ValueError(problem)
            facts = self.facts()
            fact = Fact(id=max((f.id for f in facts), default=0) + 1, text=text.strip(), why=why.strip(),
                        created=time.strftime('%Y-%m-%d'))
            self._save(facts + [fact])
            return fact

    def remove(self, fact_id: int) -> Fact:
        with self._lock:
            facts = self.facts()
            match = [f for f in facts if f.id == int(fact_id)]
            if not match:
                raise KeyError(f'there is no fact {fact_id}')
            self._save([f for f in facts if f.id != int(fact_id)])
            return match[0]

    def prompt_section(self) -> str:
        """The facts for the system prompt, newest kept if they do not all fit."""
        try:
            facts = self.facts()
        except MemoryFileError as e:
            return f'=== Lessons remembered on this microscope ===\n(unavailable: {e})\n'
        if not facts:
            return ('=== Lessons remembered on this microscope ===\n(none yet)\n')
        lines, size = [], 0
        for fact in reversed(facts):
            line = f'- [{fact.id}] {fact.text}'
            if size + len(line) > MAX_PROMPT_CHARS:
                lines.append(f'- ... {len(facts) - len(lines)} older facts left out; consider merging some '
                             f'(forget + remember).')
                break
            lines.append(line)
            size += len(line)
        return ('=== Lessons remembered on this microscope (follow them) ===\n'
                + '\n'.join(reversed(lines)) + '\n')

    def _save(self, facts: list[Fact]) -> None:
        folder = os.path.dirname(self.path)
        os.makedirs(folder, exist_ok=True)
        data = {'facts': [asdict(f) for f in facts]}
        fd, tmp = tempfile.mkstemp(prefix='.memory_', dir=folder, text=True)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=1, ensure_ascii=False)
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


class MemoryFileError(Exception):
    """The memory file could not be read."""


def _normal(text: str) -> str:
    return ' '.join(text.lower().split()).rstrip('.')
