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

import json
import os
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
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget
from kivy.utils import escape_markup

import assistant_ipc
import chat_markup
import llm_assist


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
        'get_app_state': ('Looking at the current settings', 'Looked at the current settings',
                          'Could not read the settings'),
        'read_current_plugin': ('Reading the selected plugin', 'Read the selected plugin',
                                'Could not read the plugin'),
        'read_plugin_example': ('Reading an example plugin', 'Read an example plugin',
                                'Could not read the example'),
        'propose_sequencer_script': ('Checking the script with the sequencer parser',
                                     'Script is valid, see the proposal below',
                                     'Script rejected by the parser; the assistant is fixing it'),
        'propose_plugin': ('Checking the plugin and test-running it',
                           'Plugin passed the checks and the test run, see below',
                           'Plugin rejected by the checks; the assistant is fixing it'),
    }
    is_open = BooleanProperty(False)
    header = StringProperty('')
    state = StringProperty('running')   # running, ok, failed

    def __init__(self, name: str, arguments: str, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.labels = self.LABELS.get(name, (f'Using {name}', f'Used {name}', f'{name} failed'))
        self.arguments = arguments
        self.result = ''
        self.running = True
        self.tick(2)

    def finish(self, ok: bool, result: str) -> None:
        self.running = False
        self.result = result
        self.state = 'ok' if ok else 'failed'
        text = self.labels[1 if ok else 2]
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

    def toolEnd(self, call_id: str, ok: bool, result: str) -> None:
        chip = self._chips.get(call_id)
        if chip is not None:
            chip.finish(ok, result)
        if all(not c.running for c in self._chips.values()):
            self._showTyping()

    def addCard(self, card) -> None:
        self._add(card)
        if any(c.running for c in self._chips.values()):
            self._showTyping()

    def addNote(self, text: str, color: str) -> None:
        self._add(Factory.ChatText(text=f'[color={color}]{escape_markup(text)}[/color]'))

    def finish(self, cancelled: bool) -> None:
        if self._typing is not None and self._typing.parent is self:
            self.remove_widget(self._typing)
        self.owner._animated.discard(self._typing)
        for chip in self._chips.values():
            if chip.running:
                chip.finish(False, 'stopped')
        if self._thinking is not None:
            self._thinking.finish()
        if self._body is not None:
            self._body._render()
        if cancelled:
            self.addNote('Stopped.', '999999')


class ProposalCard(BoxLayout):
    """A script the assistant proposed, shown in the chat for review. The code and the file name
    can be edited before pressing the button; GlowTracker checks it again when it is used."""

    def __init__(self, proposal: llm_assist.Proposal, owner: 'AssistantWidget', **kwargs):
        self.proposal = proposal
        self.owner = owner
        super().__init__(**kwargs)
        plugin = proposal.kind == llm_assist.PLUGIN
        kindText = 'Plugin' if plugin else 'Sequencer script'
        summary = f'  [color=bbbbbb]{escape_markup(proposal.summary)}[/color]' if proposal.summary else ''
        self.ids.title.text = f'[b]{kindText}[/b]{summary}'
        self.ids.code.text = proposal.code
        checks = escape_markup(proposal.checks)
        for warning in proposal.warnings:
            checks += f'\n[color=f0c27b]Warning: {escape_markup(warning)}[/color]'
        self.ids.checks.text = checks.strip()
        self.ids.usebutton.text = 'Save plugin and select it' if plugin else 'Save and use in Sequencer'
        self.ids.path.text = owner.defaultPath(proposal.kind)

    def setStatus(self, text: str, error: bool = False) -> None:
        self.ids.status.text = text
        self.ids.status.color = (1, 0.45, 0.45, 1) if error else (0.6, 0.9, 0.6, 1)


class AssistantWidget(BoxLayout):
    """The chat: answers stream in, tool calls show as chips, proposed sequencer scripts and plugins
    appear as cards. A card's button asks GlowTracker to save and load the script; GlowTracker
    checks it again and refuses e.g. while recording. Implements llm_assist.AppBridge by asking
    GlowTracker over the connection.
    """
    busy = BooleanProperty(False)
    connected = BooleanProperty(True)

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
        self._animated: set = set()
        self._phase = 0
        self._stick = True
        self.ids.message.send_callback = self.sendOrStop
        self.ids.chatscroll.bind(scroll_y=self._onScroll)
        self.ids.chatlog.bind(height=self._onContentHeight)
        self._showEmptyState()
        self._updateHeader()

    def _config(self) -> llm_assist.AssistantConfig:
        try:
            settings = self.client.call('get_config')
        except assistant_ipc.CommandError as e:
            raise llm_assist.AssistantError(str(e)) from e
        return llm_assist.AssistantConfig(base_url=settings['base_url'], model=settings['model'],
                                          api_key=settings['api_key'], setup=settings.get('setup') or {})

    def _updateHeader(self, config: llm_assist.AssistantConfig | None = None) -> None:
        if config is None:
            try:
                config = self._config()
            except llm_assist.AssistantError:
                config = llm_assist.AssistantConfig()
        model = config.model.strip()
        self.ids.header.text = f'[b]AI assistant[/b]  [color=999999]{escape_markup(model or "no model set")}[/color]'

    def _session(self) -> llm_assist.ChatSession:
        """The chat session, restarted when the API settings changed in GlowTracker."""
        config = self._config()
        self._updateHeader(config)
        if self._chat is not None and self._chat.config != config:
            self._chat.close()
            self._chat = None
            self._addInfo('API settings changed: this is a new conversation for the model.')
        if self._chat is None:
            self._chat = llm_assist.ChatSession(
                config, self, lambda event: Clock.schedule_once(lambda dt: self._onEvent(event)))
        return self._chat

    def disconnected(self) -> None:
        self.connected = False
        if self._chat is not None:
            self._chat.stop()
        self._addInfo('GlowTracker was closed; this window closes too.', 'ff7373')

    def close(self) -> None:
        if self._chat is not None:
            self._chat.close()

    # --- chat log --------------------------------------------------------------------------
    def _append(self, widget) -> None:
        log = self.ids.chatlog
        if self._empty is not None:
            log.remove_widget(self._empty)
            self._empty = None
        log.add_widget(widget)

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
            self._empty.ids.reminder.text = (
                '[color=f0c27b]Describe your setup in GlowTracker: Settings > AI assistant > Your setup. '
                'Still empty: ' + escape_markup(', '.join(missing)) + '.[/color]')
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

    def _onScroll(self, scroll, value) -> None:
        # Follow new text only while the user is at the bottom.
        self._stick = value <= 0.02 or self.ids.chatlog.height <= scroll.height

    def _onContentHeight(self, *args) -> None:
        if self._stick:
            self.ids.chatscroll.scroll_y = 0

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
        elif kind == 'result':
            turn.toolEnd(event.call_id, event.ok, event.text)
        elif kind == 'error':
            turn.addNote(event.text, 'ff7373')
        elif kind == 'done':
            turn.finish(event.cancelled)
            self._turn = None
            self._setBusy(False)

    # --- actions ---------------------------------------------------------------------------
    def sendOrStop(self) -> None:
        if self.busy:
            if self._chat is not None:
                self._chat.stop()
            return
        text = self.ids.message.text.strip()
        if not text:
            return
        try:
            chat = self._session()
            chat.send(text)
        except llm_assist.AssistantError as e:
            self._addInfo(str(e), 'ff7373')
            return
        self.ids.message.text = ''
        self._stick = True
        self._append(UserRow(text))
        self._turn = AssistantTurn(self)
        self._animated.add(self._turn._typing)
        self._append(self._turn)
        self._setBusy(True)

    def newChat(self) -> None:
        if self._chat is not None:
            self._chat.reset()
        self._turn = None
        self._setBusy(False)
        self._animated.clear()
        self.ids.chatlog.clear_widgets()
        self._showEmptyState()
        self._updateHeader()

    def listModels(self) -> None:
        config = self._config()
        self._addInfo(f'Listing models at {config.base_url}...')

        def work():
            try:
                models = llm_assist.list_models(config)
                text = ('Models your key can use (copy one into Settings > AI assistant > Model): '
                        + ', '.join(models)) if models else 'The API listed no models.'
                Clock.schedule_once(lambda dt: self._addInfo(text, 'bbbbbb'))
            except Exception as e:
                message = str(e)
                Clock.schedule_once(lambda dt: self._addInfo(message, 'ff7373'))

        Thread(target=work, daemon=True).start()

    # --- llm_assist.AppBridge: called from the chat's thread, answered by GlowTracker ----------
    def app_state(self) -> dict:
        return self.client.call('get_state')

    def current_plugin(self) -> str:
        return self.client.call('current_plugin')

    def propose(self, proposal: llm_assist.Proposal) -> None:
        def show(dt):
            card = ProposalCard(proposal, self)
            if self._turn is not None:
                self._turn.addCard(card)
            else:
                self._append(card)
        Clock.schedule_once(show)

    # --- using a proposal ------------------------------------------------------------------
    def defaultPath(self, kind: str) -> str:
        try:
            return self.client.call('default_path', kind=kind)
        except assistant_ipc.CommandError:
            ext = '.txt' if kind == llm_assist.SEQUENCER else '.py'
            return os.path.join(os.path.expanduser('~'), f'assistant_{kind}_{time.strftime("%Y%m%d_%H%M%S")}{ext}')

    def useProposal(self, card: ProposalCard) -> None:
        code, path, kind = card.ids.code.text, card.ids.path.text, card.proposal.kind
        problem = llm_assist.validate(kind, code)
        if problem:
            card.setStatus(f'Not used: {problem}', error=True)
            return
        command = 'use_sequencer' if kind == llm_assist.SEQUENCER else 'save_plugin'
        card.ids.usebutton.disabled = True

        def work():
            try:
                message, error = self.client.call(command, code=code, path=path), False
            except assistant_ipc.CommandError as e:
                message, error = str(e), True

            def done(dt):
                card.ids.usebutton.disabled = False
                card.setStatus(message, error=error)
            Clock.schedule_once(done)

        Thread(target=work, daemon=True).start()


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
        if cmd in ('use_sequencer', 'save_plugin'):
            return f'(demo) would {cmd.replace("_", " ")} to {args["path"]}'
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
