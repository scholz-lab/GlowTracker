"""ox: tool argument validation and calls that need the user's approval."""
import asyncio

from ox.schema import validate
from ox.tools import ToolResolver
from ox.types import Approval, Check

SCHEMA = {'type': 'object', 'required': ['volts'], 'additionalProperties': False, 'properties': {
    'volts': {'type': 'number', 'minimum': 0, 'maximum': 4.95},
    'channel': {'type': 'integer', 'enum': [0, 1]},
    'label': {'type': 'string', 'maxLength': 5},
    'times': {'type': 'array', 'items': {'type': 'number'}}}}


def test_schema_errors_are_short_and_specific():
    assert validate(SCHEMA, {'volts': 3.0, 'channel': 1}) == []
    assert validate(SCHEMA, {}) == ["'volts' is required"]
    assert validate(SCHEMA, {'volts': 9}) == ["'volts' must be <= 4.95"]
    assert validate(SCHEMA, {'volts': True}) == ["'volts' must be number, got bool"]
    assert validate(SCHEMA, {'volts': 1, 'channel': 2}) == ["'channel' must be one of [0, 1]"]
    assert validate(SCHEMA, {'volts': 1, 'colour': 'red'})[0].startswith("'colour' is not a known argument")
    assert validate(SCHEMA, {'volts': 1, 'times': [1, 'x']}) == ["'times[1]' must be number, got str"]
    assert validate(SCHEMA, {'volts': 1, 'channel': 1.0}) == []          # 1.0 is an integer in JSON


def run(coro):
    return asyncio.run(coro)


def resolver(confirm=False, check=None, calls=None):
    calls = calls if calls is not None else []
    tools = ToolResolver(builtins=False)
    tools.add_function('set_voltage', 'Set a DAC output.', lambda **a: calls.append(a) or f'set {a}',
                       SCHEMA, confirm=confirm, check=check)
    return tools, calls


def test_invalid_arguments_never_reach_the_function():
    tools, calls = resolver()
    result = run(tools.execute('set_voltage', '{"volts": 7}'))
    assert result.is_error and 'invalid arguments for set_voltage' in result.content and '<= 4.95' in result.content
    result = run(tools.execute('set_voltage', '{volts: 1'))
    assert result.is_error and 'not valid JSON' in result.content
    assert calls == []
    assert not run(tools.execute('set_voltage', '{"volts": 2}')).is_error and calls == [{'volts': 2}]


def test_confirmed_tools_wait_for_the_user_and_report_edits():
    asked = []

    async def approve(name, args, details):
        asked.append((name, args, details))
        return Approval(True, arguments=dict(args, volts=1.5), note='dimmer please')

    tools, calls = resolver(confirm=True, check=lambda a: Check(True, f'will set {a["volts"]} V'))
    result = run(tools.execute('set_voltage', '{"volts": 3}', approve=approve))
    assert asked == [('set_voltage', {'volts': 3}, 'will set 3 V')]
    assert calls == [{'volts': 1.5}]
    assert 'after changing: volts' in result.content and 'dimmer please' in result.content


def test_failed_checks_declines_and_missing_approver():
    asked = []

    async def approve(name, args, details):
        asked.append(name)
        return Approval(False, note='not now')

    tools, calls = resolver(confirm=True, check=lambda a: Check(a['volts'] < 4, 'too bright for this LED'))
    result = run(tools.execute('set_voltage', '{"volts": 4.5}', approve=approve))
    assert result.is_error and result.content == 'too bright for this LED' and asked == []
    result = run(tools.execute('set_voltage', '{"volts": 1}', approve=approve))
    assert result.is_error and 'declined' in result.content and 'not now' in result.content and calls == []
    result = run(tools.execute('set_voltage', '{"volts": 1}'))
    assert result.is_error and "needs the user's approval" in result.content


def test_edited_arguments_are_validated_again():
    async def approve(name, args, details):
        return Approval(True, arguments={'volts': 12})

    tools, calls = resolver(confirm=True)
    result = run(tools.execute('set_voltage', '{"volts": 1}', approve=approve))
    assert result.is_error and 'edited version is invalid' in result.content and calls == []
