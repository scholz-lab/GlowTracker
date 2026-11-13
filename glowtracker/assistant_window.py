"""The AI assistant window: GlowTracker's chat assistant as its own app in its own process.

GlowTracker starts this file (Assistant button) through assistant_ipc.WindowLauncher and passes the
connection address and key in the environment. Everything the window needs from the microscope
app (API settings, app state, the selected plugin, using a proposal) goes through
assistant_ipc.CommandClient, so the chat never runs on GlowTracker's Kivy thread. The window closes
itself when GlowTracker goes away.

Run on its own for UI work with a fake app: python assistant_window.py --demo
"""
# ruff: noqa: E402  (the Kivy window size must be configured before Kivy is imported)
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
import time
from threading import Thread

os.environ.setdefault('KIVY_NO_ARGS', '1')
from kivy.config import Config

Config.set('graphics', 'width', '560')
Config.set('graphics', 'height', '860')
Config.set('graphics', 'minimum_width', '380')
Config.set('graphics', 'minimum_height', '420')
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')    # no red dots on right click

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.factory import Factory
from kivy.lang import Builder
from kivy.metrics import dp
from kivy.properties import BooleanProperty, NumericProperty, ObjectProperty, StringProperty
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget
from kivy.utils import escape_markup

import platformdirs

from Advanced_settings import style_card
import assistant_chats
import assistant_documents
import assistant_ipc
import chat_markup
import llm_assist
from script_preview_ui import ScriptPreview, short_dac_label  # noqa: F401  (ScriptPreview is used in assistant.kv)

# Every conversation is saved here as a .jsonl file (messages, tool calls, approvals, usage).
TRANSCRIPT_DIR = os.path.join(platformdirs.user_data_dir('GlowTracker', appauthor=False), 'assistant_chats')

# The system's own file dialog, in a process of its own: it needs a main thread and Kivy has that one.
FILE_DIALOG = '''
import json, tkinter
from tkinter import filedialog
root = tkinter.Tk()
root.withdraw()
root.attributes('-topmost', True)
print(json.dumps(list(filedialog.askopenfilenames(title='Attach to your message'))))
'''


class ChatInput(TextInput):
    """Message box: Enter sends, Shift+Enter starts a new line."""
    send_callback = ObjectProperty(None, allownone=True)

    def keyboard_on_key_down(self, window, keycode, text, modifiers):
        if keycode[1] in ('enter', 'numpadenter') and 'shift' not in modifiers \
                and self.send_callback is not None:
            self.send_callback()
            return True
        return super().keyboard_on_key_down(window, keycode, text, modifiers)


class UserBubble(Label):
    """The user's message: a bubble on the right, as wide as its text up to max_width."""
    max_width = NumericProperty(dp(400))

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(text=self._fit, max_width=self._fit)
        self._fit()

    def _fit(self, *args) -> None:
        self.text_size = (None, None)
        self.texture_update()
        if self.texture_size[0] > self.max_width:
            self.text_size = (self.max_width - 2 * self.padding[0], None)
            self.texture_update()
        self.size = self.texture_size


class UserRow(BoxLayout):
    def __init__(self, text: str, **kwargs):
        super().__init__(**kwargs)
        self.bubble = UserBubble(text=escape_markup(text))
        self.add_widget(Widget())
        self.add_widget(self.bubble)
        self.bind(width=lambda *a: setattr(self.bubble, 'max_width', self.width * 0.8))
        self.bubble.bind(height=lambda *a: setattr(self, 'height', self.bubble.height))
        self.height = self.bubble.height


class MessageBody(BoxLayout):
    """An assistant answer, re-rendered from its Markdown as it streams in: text paragraphs
    as markup labels, code blocks in selectable monospace boxes."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.raw = ''
        self._parts: list = []
        self._pending = None

    def append(self, text: str) -> None:
        self.raw += text
        if self._pending is None:     # re-render at most ~15 times a second
            self._pending = Clock.schedule_once(self._render, 0.06)

    def setText(self, text: str) -> None:
        self.raw = text
        self._render()

    def _render(self, *args) -> None:
        if self._pending is not None:
            self._pending.cancel()
            self._pending = None
        wanted = chat_markup.segments(self.raw)
        for i, (kind, content) in enumerate(wanted):
            cls = Factory.ChatCode if kind == 'code' else Factory.ChatText
            if i < len(self._parts) and isinstance(self._parts[i], cls):
                if self._parts[i].text != content:
                    self._parts[i].text = content
                continue
            for old in self._parts[i:]:
                self.remove_widget(old)
            del self._parts[i:]
            widget = cls(text=content)
            self._parts.append(widget)
            self.add_widget(widget)
        for old in self._parts[len(wanted):]:
            self.remove_widget(old)
        del self._parts[len(wanted):]


class ThinkingBlock(BoxLayout):
    """A reasoning model's thinking: collapsed to one line, tap to read it."""
    is_open = BooleanProperty(False)
    header = StringProperty('Thinking')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.raw = ''
        self.running = True
        self._started = time.monotonic()

    def append(self, text: str) -> None:
        self.raw += text
        if self.is_open:
            self.ids.content.text = escape_markup(self.raw.strip())

    def finish(self) -> None:
        if self.running:
            self.running = False
            self.header = f'Thought for {max(1, round(time.monotonic() - self._started))} s'

    def toggle(self) -> None:
        self.is_open = not self.is_open
        self.ids.content.text = escape_markup(self.raw.strip()) if self.is_open else ''

    def tick(self, phase: int) -> None:
        if self.running:
            self.header = 'Thinking' + '.' * (phase % 3 + 1)


class ToolChip(BoxLayout):
    """One tool call: what the assistant is doing, then whether it worked. Tap for details."""
    LABELS = {
        'get_app_state': ('Reading the app state', 'Read the app state', 'Could not read the app state'),
        'read_current_plugin': ('Reading the plugin', 'Read the plugin', 'Could not read the plugin'),
        'read_plugin_example': ('Reading an example', 'Read an example', 'Could not read the example'),
        'remember': ('Proposing to remember', 'Saved to memory', 'Not saved to memory'),
        'forget': ('Proposing to forget', 'Removed from memory', 'Not removed from memory'),
        'get_settings': ('Reading the settings', 'Read the settings', 'Could not read the settings'),
        'change_setting': ('Checking the setting', 'Setting changed', 'Setting not changed'),
        'propose_sequencer_script': ('Checking the script', 'Saved and loaded', 'Script invalid, fixing it'),
        'propose_plugin': ('Checking and test-running', 'Saved and selected', 'Plugin invalid, fixing it'),
    }
    is_open = BooleanProperty(False)
    header = StringProperty('')
    state = StringProperty('running')   # running, waiting, ok, failed

    def __init__(self, name: str, arguments: str, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.labels = self.LABELS.get(name, (f'Using {name}', f'Used {name}', f'{name} failed'))
        self.arguments = arguments
        self.result = ''
        self.running = True
        self.tick(2)

    def waiting(self) -> None:
        """The checks passed and the call waits for the user's decision on the card below."""
        self.running = False
        self.state = 'waiting'
        self.header = 'Waiting for approval'

    def finish(self, ok: bool, result: str) -> None:
        self.running = False
        self.result = result
        self.state = 'ok' if ok else 'failed'
        text = self.labels[1 if ok else 2]
        if ok and result.startswith('Approved by the user after changing'):
            text += ' (with your edits)'
        if not ok and 'declined' in result:
            note = result.split('Their note:', 1)[1].strip() if 'Their note:' in result else ''
            self.header = 'Declined' + (f'  [color=bbbbbb]({escape_markup(note[:160])})[/color]' if note else '')
            self._refreshDetails()
            return
        if not ok and 'Not applied' in result:
            reason = result.split('Not applied:', 1)[-1].strip().splitlines()[0]
            self.header = f'Not applied  [color=bbbbbb]({escape_markup(reason[:160])})[/color]'
            self._refreshDetails()
            return
        if not ok:
            reason = result.split('invalid:', 1)[-1].split('Fix it and call', 1)[0]
            lines = [x.strip(' -') for x in reason.splitlines() if x.strip() and not x.strip().endswith(':')]
            text += f'  [color=bbbbbb]({escape_markup(lines[-1][:160])})[/color]' if lines else ''
        self.header = text
        self._refreshDetails()

    def tick(self, phase: int) -> None:
        if self.running:
            self.header = self.labels[0] + '.' * (phase % 3 + 1)

    def toggle(self) -> None:
        self.is_open = not self.is_open
        self._refreshDetails()

    def _refreshDetails(self) -> None:
        if not self.is_open:
            self.ids.details.text = ''
            return
        try:
            args = json.dumps(json.loads(self.arguments or '{}'), indent=1)
            args = args.replace('\\n', '\n')
        except (ValueError, TypeError):
            args = self.arguments
        self.ids.details.text = f'call {self.name}\n{args}' + (f'\n\nresult\n{self.result}' if self.result else '')


class AssistantTurn(BoxLayout):
    """Everything the assistant does for one user message, in order: thinking, answer text,
    tool calls and proposals, then possibly more text."""

    def __init__(self, owner: 'AssistantWidget', **kwargs):
        super().__init__(**kwargs)
        self.owner = owner
        self._body: MessageBody | None = None
        self._thinking: ThinkingBlock | None = None
        self._chips: dict[str, ToolChip] = {}
        self._typing = Factory.TypingIndicator()
        self.add_widget(self._typing)

    def _add(self, widget) -> None:
        if self._typing is not None and self._typing.parent is self:
            self.remove_widget(self._typing)
        self.add_widget(widget)

    def _showTyping(self) -> None:
        """Waiting for the model again (e.g. after a tool call)."""
        if self._typing is not None and self._typing.parent is None:
            self.add_widget(self._typing)

    def addThinking(self, text: str) -> None:
        if self._thinking is None:
            self._thinking = ThinkingBlock()
            self.owner._animated.add(self._thinking)
            self._add(self._thinking)
        self._thinking.append(text)

    def addText(self, text: str) -> None:
        if self._thinking is not None:
            self._thinking.finish()
        if self._body is None:
            if not text.strip():
                return
            self._body = MessageBody()
            self._add(self._body)
        self._body.append(text)

    def endMessage(self, text: str) -> None:
        if self._thinking is not None:
            self._thinking.finish()
            self._thinking = None
        if self._body is not None:
            self._body.setText(text)
        elif text.strip():
            self.addText(text)
            self._body.setText(text)
        self._body = None

    def toolStart(self, call_id: str, name: str, arguments: str) -> None:
        chip = ToolChip(name, arguments)
        self._chips[call_id] = chip
        self.owner._animated.add(chip)
        self._add(chip)

    def toolWaiting(self, call_id: str) -> None:
        chip = self._chips.get(call_id)
        if chip is not None:
            chip.waiting()
        if self._typing is not None and self._typing.parent is self:
            self.remove_widget(self._typing)

    def toolEnd(self, call_id: str, ok: bool, result: str) -> None:
        chip = self._chips.get(call_id)
        if chip is not None:
            chip.finish(ok, result)
        if all(not c.running for c in self._chips.values()):
            self._showTyping()

    def addCard(self, card) -> None:
        self._add(card)

    def addNote(self, text: str, color: str) -> None:
        self._add(Factory.ChatText(text=f'[color={color}]{escape_markup(text)}[/color]'))

    def finish(self, cancelled: bool) -> None:
        if self._typing is not None and self._typing.parent is self:
            self.remove_widget(self._typing)
        self.owner._animated.discard(self._typing)
        for chip in self._chips.values():
            if chip.running or chip.state == 'waiting':
                chip.finish(False, 'stopped')
        if self._thinking is not None:
            self._thinking.finish()
        if self._body is not None:
            self._body._render()
        if cancelled:
            self.addNote('Stopped.', '999999')


class ProposalCard(BoxLayout):
    """A script the assistant wants to use, waiting for the user's decision. The code and file
    name can be edited before approving; ox checks the edited version again before it is
    applied, and the assistant is told what was changed."""

    def __init__(self, call_id: str, name: str, arguments: dict, details: str, owner: 'AssistantWidget',
                 **kwargs):
        self.call_id = call_id
        self.owner = owner
        self.arguments = dict(arguments)
        self.kind = llm_assist.PLUGIN if name == 'propose_plugin' else llm_assist.SEQUENCER
        self.codeKey = 'code' if self.kind == llm_assist.PLUGIN else 'script'
        super().__init__(**kwargs)
        plugin = self.kind == llm_assist.PLUGIN
        summary = arguments.get('summary') or ''
        summary = f'  [color=bbbbbb]{escape_markup(summary)}[/color]' if summary else ''
        self.ids.title.text = f'[b]{"Plugin" if plugin else "Sequencer script"}[/b]{summary}'
        self.ids.code.text = llm_assist._unfence(arguments.get(self.codeKey, '')).strip()
        self.originalCode = self.ids.code.text
        # the check report is for the model; the card shows only what the user must know
        warnings = [escape_markup(line) for line in details.splitlines() if line.startswith('Warning:')]
        self.ids.checks.text = '\n'.join(f'[color=f0c27b]{line}[/color]' for line in warnings)
        self.ids.usebutton.text = 'Approve'
        self.ids.path.text = arguments.get('path') or owner.defaultPath(
            self.kind, arguments.get('name') or arguments.get('summary') or '')
        self.originalPath = self.ids.path.text
        self.decided = False
        # what the script does, again after the user edits it
        try:
            setup = owner._config().setup
        except Exception:
            setup = {}
        self.ids.preview.dac_labels = [short_dac_label(setup.get('dac0', '')), short_dac_label(setup.get('dac1', ''))]
        self._previewEvent = Clock.create_trigger(self.refreshPreview, 1.0)
        self.ids.code.bind(text=lambda *args: self._previewEvent())
        self.refreshPreview()

    def refreshPreview(self, *args) -> None:
        code = self.ids.code.text.strip()
        if self.kind == llm_assist.PLUGIN:
            self.ids.preview.show_plugin(code)
        else:
            self.ids.preview.show_sequencer(code)

    def approve(self) -> None:
        code, path = self.ids.code.text.strip(), self.ids.path.text.strip()
        problem = llm_assist.validate(self.kind, code)
        if problem:
            self.setStatus(f'Not approved: {problem}', error=True)
            return
        # Only what the user changed counts as an edit. A default path the card filled in is
        # used as shown (default_path() hands it to the tool) without being reported as an edit.
        edited = dict(self.arguments)
        if code != self.originalCode:
            edited[self.codeKey] = code
        if path != self.originalPath:
            edited['path'] = path
        elif not self.arguments.get('path'):
            self.owner.approvedDefaultPaths[self.kind] = path
        self._decide()
        self.setStatus('Saving...')
        self.owner.answer(self.call_id, True, edited if edited != self.arguments else None)

    def decline(self) -> None:
        self._decide()
        self.setStatus('Declined', error=True)
        self.owner.answer(self.call_id, False, None)

    def _decide(self) -> None:
        self.decided = True
        self.ids.usebutton.disabled = True
        self.ids.declinebutton.disabled = True

    def result(self, ok: bool, text: str) -> None:
        """What happened after the decision: GlowTracker's answer (saved, or refused and why)."""
        if 'declined' in text:
            return
        lines = [x for x in text.replace('ERROR', '').splitlines()
                 if x.strip() and not x.startswith(('Approved by the user', 'Their note', '{', '}', ' ', '"'))]
        self.setStatus(lines[0] if lines else text.strip().splitlines()[0], error=not ok)

    def stopped(self) -> None:
        if not self.decided:
            self._decide()
            self.setStatus('Stopped', error=True)

    def setStatus(self, text: str, error: bool = False) -> None:
        self.ids.status.text = text
        self.ids.status.color = (1, 0.45, 0.45, 1) if error else (0.6, 0.9, 0.6, 1)


class MemoryCard(BoxLayout):
    """The assistant wants to remember (or forget) a fact for all future conversations.
    The fact can be edited before approving."""

    def __init__(self, call_id: str, name: str, arguments: dict, details: str, owner: 'AssistantWidget',
                 **kwargs):
        self.call_id = call_id
        self.owner = owner
        self.arguments = dict(arguments)
        self.forgetting = name == 'forget'
        super().__init__(**kwargs)
        why = escape_markup(arguments.get('why') or '')
        if self.forgetting:
            self.ids.title.text = f'[b]Forget this?[/b]  [color=bbbbbb]{why}[/color]'
            self.ids.fact.text = details
            self.ids.fact.readonly = True
            self.ids.approvebutton.text = 'Forget'
        else:
            self.ids.title.text = ('[b]Remember this?[/b]'
                                   + (f'  [color=bbbbbb]{why}[/color]' if why else ''))
            self.ids.fact.text = arguments.get('fact', '')
            self.ids.approvebutton.text = 'Remember'
        self.decided = False

    def approve(self) -> None:
        edited = None
        if not self.forgetting and self.ids.fact.text.strip() != (self.arguments.get('fact') or '').strip():
            edited = dict(self.arguments, fact=self.ids.fact.text.strip())
        self._decide()
        self.owner.answer(self.call_id, True, edited)

    def decline(self) -> None:
        self._decide()
        self.setStatus('Declined', error=True)
        self.owner.answer(self.call_id, False, None)

    def _decide(self) -> None:
        self.decided = True
        self.ids.approvebutton.disabled = True
        self.ids.declinebutton.disabled = True
        self.ids.fact.readonly = True

    def result(self, ok: bool, text: str) -> None:
        if 'declined' in text:
            return
        lines = [x for x in text.replace('ERROR', '').splitlines()
                 if x.strip() and not x.startswith(('Approved by the user', 'Their note', '{', '}', ' ', '"'))]
        self.setStatus(lines[0] if lines else text.strip(), error=not ok)

    def stopped(self) -> None:
        if not self.decided:
            self._decide()
            self.setStatus('Stopped', error=True)

    def setStatus(self, text: str, error: bool = False) -> None:
        self.ids.status.text = text
        self.ids.status.color = (1, 0.45, 0.45, 1) if error else (0.6, 0.9, 0.6, 1)


class SettingCard(MemoryCard):
    """The assistant wants to change a GlowTracker setting. Shows what changes; the new value can
    be edited before approving. (Uses MemoryCard's layout.)"""

    def __init__(self, call_id: str, name: str, arguments: dict, details: str, owner: 'AssistantWidget',
                 **kwargs):
        super().__init__(call_id, 'remember', arguments, details, owner, **kwargs)
        why = escape_markup(arguments.get('why') or '')
        self.ids.title.text = (f'[b]Change setting?[/b]  {escape_markup(details)}'
                               + (f'\n[color=bbbbbb]{why}[/color]' if why else ''))
        self.ids.fact.text = str(arguments.get('value', ''))
        self.ids.approvebutton.text = 'Approve'

    def approve(self) -> None:
        value = self.ids.fact.text.strip()
        edited = dict(self.arguments, value=value) if value != str(self.arguments.get('value', '')) else None
        self._decide()
        self.owner.answer(self.call_id, True, edited)


class MemoryList(BoxLayout):
    """Popup content: everything remembered, each with Delete, and a field to add a fact. The
    user's own housekeeping; the model is not involved."""

    def __init__(self, store, close, **kwargs):
        super().__init__(**kwargs)
        self.store = store
        self.close = close
        self.ids.where.text = f'Stored in {escape_markup(store.path)}'
        self.refresh()

    def refresh(self) -> None:
        rows = self.ids.rows
        rows.clear_widgets()
        try:
            facts = self.store.facts()
        except Exception as e:
            rows.add_widget(Factory.ChatText(text=f'[color=ff7373]{escape_markup(str(e))}[/color]'))
            return
        if not facts:
            rows.add_widget(Factory.ChatText(text='[color=999999]Nothing remembered yet.[/color]'))
        for fact in facts:
            row = Factory.MemoryRow()
            row.ids.text.text = (f'[b]{fact.id}.[/b] {escape_markup(fact.text)}'
                                 + (f'\n[color=888888]{escape_markup(fact.why)} · {fact.created}[/color]'
                                    if fact.why else f'\n[color=888888]{fact.created}[/color]'))
            row.ids.delete.bind(on_release=lambda button, fid=fact.id: self.delete(fid))
            rows.add_widget(row)

    def delete(self, fact_id: int) -> None:
        self.store.remove(fact_id)
        self.refresh()

    def add(self) -> None:
        text = self.ids.newfact.text.strip()
        if not text:
            return
        try:
            self.store.add(text, 'added by hand')
        except (ValueError, OSError) as e:
            self.ids.where.text = f'[color=ff7373]{escape_markup(str(e))}[/color]'
            return
        self.ids.newfact.text = ''
        self.ids.where.text = f'Stored in {escape_markup(self.store.path)}'
        self.refresh()


class ChatList(BoxLayout):
    """Popup content: the saved conversations, newest first. Tap one to continue it."""

    def __init__(self, owner: 'AssistantWidget', close, **kwargs):
        super().__init__(**kwargs)
        self.owner = owner
        self.close = close
        self.chats = assistant_chats.list_chats(TRANSCRIPT_DIR)
        self.refresh()

    def refresh(self, *args) -> None:
        rows = self.ids.rows
        rows.clear_widgets()
        query = self.ids.search.text.strip().lower()
        chats = [c for c in self.chats if query in c.title.lower() or query in c.model.lower()]
        if not chats:
            text = 'No chat matches.' if query else 'No saved chats yet.'
            rows.add_widget(Factory.ChatText(text=f'[color=999999]{text}[/color]'))
        current = self.owner.currentChat()
        for chat in chats:
            row = Factory.ChatRow()
            row.ids.open.text = (f'{"[b]" if chat.path == current else ""}{escape_markup(chat.title)}'
                                 f'{"[/b]" if chat.path == current else ""}\n'
                                 f'[size=11sp][color=888888]{chat.when}  ·  {escape_markup(chat.model)}[/color][/size]')
            row.ids.open.bind(on_release=lambda b, path=chat.path: self.choose(path))
            row.ids.delete.bind(on_release=lambda b, c=chat: self.delete(b, c))
            rows.add_widget(row)

    def choose(self, path: str) -> None:
        self.close()
        self.owner.openChat(path)

    def delete(self, button, chat) -> None:
        """The transcript is also the record of how a script came about: ask once more."""
        if button.text != 'Sure?':
            button.text = 'Sure?'
            return
        try:
            os.remove(chat.path)
        except OSError as e:
            self.owner._addInfo(f'Could not delete the chat: {e}', 'ff7373')
            return
        if chat.path == self.owner.currentChat():
            self.owner.newChat()
        self.chats.remove(chat)
        self.refresh()


class ModelList(BoxLayout):
    """Popup content: the models to choose from. The recommended ones come first, then what the
    API lists for this key; any other name can be typed."""

    def __init__(self, owner: 'AssistantWidget', config: llm_assist.AssistantConfig, close, **kwargs):
        super().__init__(**kwargs)
        self.owner = owner
        self.close = close
        self.current = config.model.strip()
        self.recommended = dict(llm_assist.GWDG_MODELS) if llm_assist.is_gwdg(config.base_url) else {}
        self.available: list[str] | None = None
        self.ids.status.text = 'Asking the API which models your key can use...'
        self.refresh()
        Thread(target=self._fetch, args=(config,), daemon=True).start()

    def _fetch(self, config) -> None:
        try:
            models, problem = llm_assist.list_models(config), ''
        except Exception as e:
            models, problem = None, str(e)
        Clock.schedule_once(lambda dt: self._fetched(models, problem))

    def _fetched(self, models: list[str] | None, problem: str) -> None:
        self.available = models
        self.ids.status.text = f'[color=ff7373]{escape_markup(problem)}[/color]' if problem else ''
        self.refresh()

    def names(self) -> list[str]:
        if self.available is None:
            return list(self.recommended)
        first = [name for name in self.recommended if name in self.available]
        return first + [name for name in self.available if name not in first]

    def refresh(self, *args) -> None:
        rows = self.ids.rows
        rows.clear_widgets()
        query = self.ids.search.text.strip()
        names = [n for n in self.names() if query.lower() in n.lower()]
        for name in names:
            about = self.recommended.get(name, '')
            row = Factory.ChatSuggestion(tint=(0.18, 0.3, 0.5, 1) if name == self.current else (0.14, 0.15, 0.17, 1))
            row.markup = True
            row.text = escape_markup(name) + (f'\n[size=11sp][color=999999]{escape_markup(about)}[/color][/size]'
                                              if about else '')
            row.bind(on_release=lambda b, model=name: self.choose(model))
            rows.add_widget(row)
        if query and query not in names:
            row = Factory.ChatSuggestion(text=f'Use "{escape_markup(query)}"', markup=True)
            row.bind(on_release=lambda b: self.choose(query))
            rows.add_widget(row)

    def typed(self) -> None:
        """Enter in the search field: the only match, or the name as typed."""
        query = self.ids.search.text.strip()
        names = [n for n in self.names() if query.lower() in n.lower()]
        if query:
            self.choose(names[0] if len(names) == 1 else query)

    def choose(self, model: str) -> None:
        self.close()
        self.owner.chooseModel(model)


class AssistantWidget(BoxLayout):
    """The chat: answers stream in, tool calls show as chips, proposed sequencer scripts and plugins
    appear as cards. A card's button asks GlowTracker to save and load the script; GlowTracker
    checks it again and refuses e.g. while recording. Implements llm_assist.AppBridge by asking
    GlowTracker over the connection.
    """
    busy = BooleanProperty(False)
    connected = BooleanProperty(True)
    attachmentCount = NumericProperty(0)

    SUGGESTIONS = (
        '5 pulses of 4.5 V, 1 s long, every 20 s, starting 10 s after Record',
        'Switch the light on at 3 V while the worm moves forward',
        'What does the current sequencer script do?',
    )

    def init(self, client):
        """client: an assistant_ipc.CommandClient (or anything with call(cmd, **args))."""
        self.client = client
        self._chat: llm_assist.ChatSession | None = None
        self._turn: AssistantTurn | None = None
        self._cards: dict[str, ProposalCard] = {}
        self._usage: dict = {}
        self.approvedDefaultPaths: dict[str, str] = {}     # kind -> default path shown on an approved card
        self._animated: set = set()
        self._phase = 0
        self._spacer = Widget(size_hint_y=None, height=0)   # see _fitSpacer
        self._anchor = None
        self._topOffset = 0.0       # the view's distance from the top of the chat log
        self._holding = False
        self._userScrolling = False
        self._holdSoon = Clock.create_trigger(lambda dt: self._holdView(), -1)
        self._userScrollEnds = Clock.create_trigger(lambda dt: setattr(self, '_userScrolling', False), 0.2)
        self.attachments: list[dict] = []
        self._picking = False
        Window.bind(on_drop_file=lambda window, path, *args: self.addFiles([path.decode('utf-8')]))
        self.ids.message.send_callback = self.sendOrStop
        self.ids.chatscroll.bind(scroll_y=self._onScroll)
        self.ids.chatlog.bind(height=self._onContentHeight)
        self.ids.chatscroll.bind(height=self._onContentHeight)
        self._showEmptyState()
        self._updateHeader()

    def _config(self) -> llm_assist.AssistantConfig:
        try:
            settings = self.client.call('get_config')
        except assistant_ipc.CommandError as e:
            raise llm_assist.AssistantError(str(e)) from e
        return llm_assist.AssistantConfig(base_url=settings['base_url'], model=settings['model'],
                                          api_key=settings['api_key'], setup=settings.get('setup') or {},
                                          memory_file=settings.get('memory_file') or '')

    def _updateHeader(self, config: llm_assist.AssistantConfig | None = None) -> None:
        if config is None:
            try:
                config = self._config()
            except llm_assist.AssistantError:
                config = llm_assist.AssistantConfig()
        model = config.model.strip()
        self.ids.modelbutton.text = (model if len(model) <= 26 else model[:24] + '..') or 'Choose a model'
        usage = ''
        if self._usage.get('prompt') or self._usage.get('completion'):
            usage = (f'[color=777777]in {_tokens(self._usage["prompt"])} · '
                     f'out {_tokens(self._usage["completion"])}')
            if self._usage.get('cost') is not None:
                usage += f' · ${self._usage["cost"]:.4f}'
            usage += '[/color]'
        self.ids.header.text = usage

    def _session(self) -> llm_assist.ChatSession:
        """The chat session, restarted when the API settings changed in GlowTracker."""
        config = self._config()
        self._updateHeader(config)
        if self._chat is not None and self._chat.config != config:
            if dataclasses.replace(self._chat.config, model=config.model) == config:
                self._chat.set_model(config.model)      # the conversation can carry on
            else:
                self._chat.close()
                self._chat = None
                self._addInfo('Settings changed: new conversation.')
        if self._chat is None:
            self._chat = llm_assist.ChatSession(
                config, self, lambda event: Clock.schedule_once(lambda dt: self._onEvent(event)),
                transcript_dir=TRANSCRIPT_DIR)
        return self._chat

    def disconnected(self) -> None:
        self.connected = False
        if self._chat is not None:
            self._chat.stop()
        self._addInfo('GlowTracker was closed.', 'ff7373')

    def close(self) -> None:
        if self._chat is not None:
            self._chat.close()

    # --- chat log --------------------------------------------------------------------------
    def _append(self, widget) -> None:
        log = self.ids.chatlog
        if self._empty is not None:
            log.remove_widget(self._empty)
            self._empty = None
        if self._spacer.parent is None:
            log.add_widget(self._spacer)
        log.add_widget(widget, index=1)     # above the spacer, which stays last

    def _addInfo(self, text: str, color: str = '999999') -> None:
        self._append(Factory.ChatText(text=f'[color={color}]{escape_markup(text)}[/color]'))

    def _showEmptyState(self) -> None:
        self._empty = Factory.ChatEmptyState()
        for suggestion in self.SUGGESTIONS:
            button = Factory.ChatSuggestion(text=suggestion)
            button.bind(on_release=lambda b: self._useSuggestion(b.text))
            self._empty.ids.suggestions.add_widget(button)
        try:
            setup = self._config().setup
        except llm_assist.AssistantError:
            setup = {}
        missing = [label for key, label in llm_assist.SETUP_FIELDS
                   if key in ('subject', 'dac0', 'dac1') and not (setup.get(key) or '').strip()]
        if missing:
            self._empty.ids.reminder.text = ('[color=f0c27b]Describe your setup in Settings > AI assistant '
                                             '(missing: ' + escape_markup(', '.join(missing)) + ').[/color]')
        scroll = self.ids.chatscroll
        def fit(*args):
            if self._empty is not None:
                self._empty.height = scroll.height - dp(20)
        fit()
        scroll.bind(height=fit)
        self.ids.chatlog.add_widget(self._empty)

    def _useSuggestion(self, text: str) -> None:
        self.ids.message.text = text
        self.sendOrStop()

    # The newest message goes to the top of the view and its answer fills in below it. A spacer
    # under the answer takes up the rest of the view and shrinks as the answer grows, so the height
    # of the log stays the same and nothing moves. A longer answer runs on below the view.
    def _scrollRange(self) -> float:
        return max(self.ids.chatlog.height - self.ids.chatscroll.height, 0.0)

    def userScrolled(self) -> None:
        """Wheel or drag: only this moves the view. The ScrollView also rewrites scroll_y by itself
        when the log's height changes, e.g. a new text is very tall until it gets its width."""
        self._userScrolling = True
        self._userScrollEnds()

    def _onScroll(self, scroll, value) -> None:
        if self._userScrolling:
            self._topOffset = (1 - value) * self._scrollRange()
        elif not self._holding:
            self._holdSoon()

    def _onContentHeight(self, *args) -> None:
        self._fitSpacer()
        self._holdView()

    def _holdView(self) -> None:
        """scroll_y is a fraction of the log's height: keep the distance from the top instead."""
        if self._userScrolling:
            return
        scrollRange = self._scrollRange()
        offset = min(self._topOffset, scrollRange)
        self._holding = True
        self.ids.chatscroll.scroll_y = 1 - offset / scrollRange if scrollRange > 0 else 1
        self._holding = False

    def _rows(self) -> list:
        return [w for w in reversed(self.ids.chatlog.children) if w is not self._spacer]

    def _fitSpacer(self) -> None:
        log, rows = self.ids.chatlog, self._rows()
        if self._anchor not in rows:
            self._spacer.height = 0
            return
        below = rows[rows.index(self._anchor):]
        used = sum(w.height for w in below) + log.spacing * len(below) + log.padding[3]
        self._spacer.height = max(self.ids.chatscroll.height - used - log.padding[1], 0)

    def _showAtTop(self, widget) -> None:
        """Bring this row to the top of the view, once, without animation."""
        self._anchor = widget

        def go(dt):
            log, rows = self.ids.chatlog, self._rows()
            if widget in rows:
                above = rows[:rows.index(widget)]
                self._topOffset = sum(w.height for w in above) + log.spacing * len(above)
                self._fitSpacer()
                self._holdView()

        Clock.schedule_once(go, 0.05)

    def _showEnd(self) -> None:
        """Bring the end of the log into view: an approval card must not wait out of sight."""
        def go(dt):
            self._topOffset = self._scrollRange()
            self._holdView()

        Clock.schedule_once(go, 0.1)

    def _tick(self, dt) -> None:
        self._phase += 1
        for widget in list(self._animated):
            if widget.parent is None and not getattr(widget, 'running', False):
                self._animated.discard(widget)
                continue
            widget.tick(self._phase)
        if not self.busy:
            self._animated = {w for w in self._animated if getattr(w, 'running', False)}

    def _setBusy(self, busy: bool) -> None:
        self.busy = busy
        if busy and getattr(self, '_ticker', None) is None:
            self._ticker = Clock.schedule_interval(self._tick, 0.4)
        elif not busy and getattr(self, '_ticker', None) is not None:
            self._ticker.cancel()
            self._ticker = None

    def _onEvent(self, event: llm_assist.ChatEvent) -> None:
        turn = self._turn
        if turn is None:
            return
        kind = event.kind
        if kind == 'text':
            turn.addText(event.text)
        elif kind == 'thinking':
            turn.addThinking(event.text)
        elif kind == 'message':
            turn.endMessage(event.text)
        elif kind == 'tool':
            turn.toolStart(event.call_id, event.name, event.text)
        elif kind == 'approval':
            turn.toolWaiting(event.call_id)
            cls = {'remember': MemoryCard, 'forget': MemoryCard,
                   'change_setting': SettingCard}.get(event.name, ProposalCard)
            card = cls(event.call_id, event.name, event.data, event.text, self)
            self._cards[event.call_id] = card
            turn.addCard(card)
            self._showEnd()
        elif kind == 'result':
            turn.toolEnd(event.call_id, event.ok, event.text)
            card = self._cards.pop(event.call_id, None)
            if card is not None:
                card.result(event.ok, event.text)
        elif kind == 'usage':
            self._usage = event.data
            self._updateHeader(self._chat.config if self._chat is not None else None)
        elif kind == 'error':
            turn.addNote(event.text, 'ff7373')
        elif kind == 'done':
            turn.finish(event.cancelled)
            for card in self._cards.values():
                card.stopped()
            self._cards.clear()
            self._turn = None
            self._setBusy(False)

    # --- actions ---------------------------------------------------------------------------
    def sendOrStop(self) -> None:
        if self.busy:
            if self._chat is not None:
                self._chat.stop()
            return
        text = self.ids.message.text.strip()
        attachments = list(self.attachments)
        if not text and not attachments:
            return
        try:
            chat = self._session()
            chat.send(text, attachments)
        except llm_assist.AssistantError as e:
            self._addInfo(str(e), 'ff7373')
            return
        self.ids.message.text = ''
        self._clearAttachments()
        first = self._showUser(text, attachments)
        self._turn = AssistantTurn(self)
        self._animated.add(self._turn._typing)
        self._append(self._turn)
        self._showAtTop(first)
        self._setBusy(True)

    def _showUser(self, text: str, attachments: list[dict]):
        """Add the user's message to the log and return its first row."""
        first = None
        if attachments:
            names = ', '.join(a['name'] for a in attachments)
            first = Factory.ChatText(text=f'[color=999999]{escape_markup(names)}[/color]', halign='right',
                                     font_size='12sp')
            self._append(first)
        if text:
            row = UserRow(text)
            self._append(row)
            first = first or row
        return first

    def _clearLog(self) -> None:
        self._turn = None
        self._cards.clear()
        self._usage = {}
        self._setBusy(False)
        self._animated.clear()
        self.ids.chatlog.clear_widgets()
        self._empty = None
        self._anchor = None
        self._topOffset = 0.0

    def newChat(self) -> None:
        if self._chat is not None:
            self._chat.reset()
        self._clearLog()
        self._showEmptyState()
        self._updateHeader()

    def currentChat(self) -> str | None:
        """The transcript of the conversation on screen."""
        return self._chat.transcript.path if self._chat is not None else None

    def showChats(self) -> None:
        self._openList('Chats', lambda close: ChatList(self, close))

    def openChat(self, path: str) -> None:
        """Show a saved conversation and carry on with it."""
        if path == self.currentChat():
            return
        try:
            records = assistant_chats.read(path)
            chat = self._session()
        except (OSError, llm_assist.AssistantError) as e:
            self._addInfo(f'Could not open the chat: {e}', 'ff7373')
            return
        self._clearLog()
        chat.resume(path, records)
        self._usage = dict(chat.usage)
        self._replay(records)
        self._updateHeader(chat.config)

    def _replay(self, records: list[dict]) -> None:
        """Rebuild the chat log from a transcript. Approval cards are not shown again: what was
        decided is in the tool's chip."""
        turn = None
        last = None
        for record in records:
            kind = record.get('type')
            if kind == 'user':
                if turn is not None:
                    turn.finish(False)
                    turn = None
                last = self._showUser(record.get('content') or '', record.get('attachments') or [])
            elif kind in ('assistant', 'tool_call', 'tool_result', 'error', 'stopped'):
                if turn is None:
                    turn = AssistantTurn(self)
                    self._append(turn)
                if kind == 'assistant':
                    turn.endMessage(record.get('content') or '')
                elif kind == 'tool_call':
                    turn.toolStart(record.get('call_id', ''), record.get('name', ''), record.get('arguments') or '')
                elif kind == 'tool_result':
                    content = record.get('content') or ''
                    turn.toolEnd(record.get('call_id', ''), llm_assist._tool_ok(content), content)
                elif kind == 'error':
                    turn.addNote(llm_assist._error_text(record.get('content') or ''), 'ff7373')
                else:
                    turn.finish(True)
                    turn = None
        if turn is not None:
            turn.finish(False)
        self._animated.clear()
        if last is not None:
            self._showAtTop(last)       # where the conversation left off

    def _openList(self, title: str, content) -> None:
        """A popup holding a list; content is called with the function that closes it."""
        popup = Popup(title=title, size_hint=(0.92, 0.8))
        style_card(popup, padding=dp(16))
        popup.title_font = 'Roboto-Bold'
        popup.content = content(popup.dismiss)
        popup.open()

    # --- model -----------------------------------------------------------------------------
    def showModels(self) -> None:
        try:
            config = self._config()
        except llm_assist.AssistantError as e:
            self._addInfo(str(e), 'ff7373')
            return
        self._openList('Model', lambda close: ModelList(self, config, close))

    def chooseModel(self, model: str) -> None:
        """Use this model from now on, in this conversation too, and keep it in the settings."""
        if self.busy:
            self._addInfo('Wait for the answer before switching the model.', 'ff7373')
            return
        try:
            self._call('set_model', model=model)
            if self._chat is not None:
                self._chat.set_model(model)
        except llm_assist.AssistantError as e:
            self._addInfo(str(e), 'ff7373')
            return
        if self._empty is None:
            self._addInfo(f'Answers now come from {model}.')
        self._updateHeader()

    # --- attachments -----------------------------------------------------------------------
    def pickFiles(self) -> None:
        if self._picking:
            return
        self._picking = True

        def work():
            try:
                done = subprocess.run([sys.executable, '-c', FILE_DIALOG], capture_output=True, text=True)
                paths = json.loads(done.stdout)
            except (OSError, ValueError):
                paths = None
            Clock.schedule_once(lambda dt: self._picked(paths))

        Thread(target=work, daemon=True).start()

    def _picked(self, paths: list | None) -> None:
        self._picking = False
        if paths is None:
            self._addInfo('Could not open a file dialog here. Drag the files onto this window instead.', 'ff7373')
            return
        self.addFiles(paths)

    def addFiles(self, paths: list[str]) -> None:
        """Read the files for the next message. Off the GUI thread: a long PDF takes a while."""
        def work():
            for path in paths:
                try:
                    attachment = assistant_documents.load(path)
                except assistant_documents.AttachmentError as e:
                    message = str(e)
                    Clock.schedule_once(lambda dt, message=message: self._addInfo(message, 'ff7373'))
                    continue
                Clock.schedule_once(lambda dt, a=attachment: self._attach(a))

        Thread(target=work, daemon=True).start()

    def _attach(self, attachment: dict) -> None:
        self.attachments.append(attachment)
        self.attachmentCount = len(self.attachments)
        note = f'  ({attachment["note"]})' if attachment.get('note') else ''
        chip = Factory.AttachmentChip(text=f'{attachment["name"]}{note}   x')
        chip.bind(on_release=lambda button: self._detach(attachment, button))
        self.ids.attachments.add_widget(chip)

    def _detach(self, attachment: dict, chip) -> None:
        self.attachments.remove(attachment)
        self.attachmentCount = len(self.attachments)
        self.ids.attachments.remove_widget(chip)

    def _clearAttachments(self) -> None:
        self.attachments = []
        self.attachmentCount = 0
        self.ids.attachments.clear_widgets()

    def showMemory(self) -> None:
        try:
            store = llm_assist.MemoryStore(self._config().memory_path())
        except llm_assist.AssistantError as e:
            self._addInfo(str(e), 'ff7373')
            return
        self._openList('Memory', lambda close: MemoryList(store, close))

    # --- llm_assist.AppBridge: called from the chat's thread, answered by GlowTracker ----------
    def app_state(self) -> dict:
        return self.client.call('get_state')

    def current_plugin(self) -> str:
        return self.client.call('current_plugin')

    def apply(self, kind: str, code: str, path: str) -> str:
        """Save and load an approved script in GlowTracker, which checks it again."""
        command = 'use_sequencer' if kind == llm_assist.SEQUENCER else 'save_plugin'
        try:
            return self.client.call(command, code=code, path=path)
        except assistant_ipc.CommandError as e:
            raise llm_assist.AssistantError(str(e)) from e

    def settings(self, section: str) -> str:
        return self._call('get_settings', section=section)

    def check_setting(self, section: str, key: str, value) -> dict:
        return self._call('check_setting', section=section, key=key, value=value)

    def change_setting(self, section: str, key: str, value) -> str:
        return self._call('change_setting', section=section, key=key, value=value)

    def _call(self, command: str, **args):
        try:
            return self.client.call(command, **args)
        except assistant_ipc.CommandError as e:
            raise llm_assist.AssistantError(str(e)) from e

    def default_path(self, kind: str, name: str = '') -> str:
        """The path for a script the model gave none for: the one the approved card showed."""
        return self.approvedDefaultPaths.pop(kind, None) or self.defaultPath(kind, name)

    # --- the user's decision on a proposal -----------------------------------------------------
    def answer(self, call_id: str, approved: bool, arguments: dict | None) -> None:
        if self._chat is not None:
            self._chat.answer(call_id, approved, arguments)

    def defaultPath(self, kind: str, name: str = '') -> str:
        try:
            return self.client.call('default_path', kind=kind, name=name)
        except assistant_ipc.CommandError:
            ext = '.txt' if kind == llm_assist.SEQUENCER else '.py'
            return os.path.join(os.path.expanduser('~'), f'assistant_{kind}_{time.strftime("%Y%m%d_%H%M%S")}{ext}')


def _tokens(n: int) -> str:
    return f'{n / 1000:.1f}k' if n >= 1000 else str(n)


class AssistantWindowApp(App):
    title = 'GlowTracker AI assistant'

    def __init__(self, client_factory, **kwargs):
        super().__init__(**kwargs)
        self._client_factory = client_factory

    def build(self):
        Window.clearcolor = (0.075, 0.08, 0.09, 1)
        Builder.load_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assistant.kv'))
        self.widget = AssistantWidget()
        return self.widget

    def on_start(self):
        client = self._client_factory(
            on_event=lambda name, args: Clock.schedule_once(lambda dt: self._onAppEvent(name)),
            on_disconnect=lambda: Clock.schedule_once(lambda dt: self._onDisconnect()))
        self.widget.init(client)

    def _onAppEvent(self, name: str) -> None:
        if name == 'focus':
            Window.restore()
            Window.raise_window()
        elif name == 'shutdown':
            self.stop()

    def _onDisconnect(self) -> None:
        self.widget.disconnected()
        Clock.schedule_once(lambda dt: self.stop(), 1.5)

    def on_stop(self):
        self.widget.close()


class DemoClient:
    """Stands in for GlowTracker (python assistant_window.py --demo). API settings come from
    OX_BASE_URL / OX_API_KEY / OX_MODEL."""

    def __init__(self, on_event=None, on_disconnect=None):
        self.connected = True

    def call(self, cmd, timeout=15.0, **args):
        if cmd == 'get_config':
            return {'base_url': os.environ.get('OX_BASE_URL', llm_assist.DEFAULT_BASE_URL),
                    'model': os.environ.get('OX_MODEL', ''), 'api_key': os.environ.get('OX_API_KEY', '')}
        if cmd == 'get_state':
            return {'recording': False, 'tracking': True, 'daq_mode': 'Sequencer', 'daq_connected': False,
                    'sequencer_script': 'mode: [time]\n0: [off]\n5: [on, 2.0]\n15: [off]', 'plugin_file': ''}
        if cmd == 'current_plugin':
            return ''
        if cmd == 'default_path':
            return os.path.join(os.path.expanduser('~'), f'demo_{args["kind"]}.txt')
        if cmd == 'set_model':
            os.environ['OX_MODEL'] = args['model']
            return args['model']
        if cmd in ('use_sequencer', 'save_plugin'):
            return f'(demo) would {cmd.replace("_", " ")} to {args["path"]}'
        if cmd == 'get_settings':
            return 'Camera.framerate = 10  (Framerate; numeric)\nTracking.showtrackingoverlay = 1  (Tracking overlay; bool)'
        if cmd == 'check_setting':
            return {'section': args['section'], 'key': args['key'], 'title': args['key'], 'current': '10',
                    'new': str(args['value'])}
        if cmd == 'change_setting':
            return f'(demo) would change {args["section"]}.{args["key"]} to {args["value"]}'
        raise assistant_ipc.CommandError(f'unknown command {cmd}')

    def close(self):
        pass


def main() -> None:
    if '--demo' in sys.argv:
        factory = DemoClient
    else:
        factory = assistant_ipc.CommandClient.from_environment
    AssistantWindowApp(factory).run()


if __name__ == '__main__':
    main()
