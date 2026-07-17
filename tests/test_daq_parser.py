import pytest

import DAQ_control as DAQ


def test_stage_program_accepts_zero_exterior_constant():
    program = DAQ.DAQStageProgram()
    program.update(exteriorConstant=2.5)
    program.update(exteriorConstant=0)
    assert program.exteriorConstant == 0


def test_frame_script_is_validated_and_sorted():
    control = DAQ.DAQControl()
    control.parseTextScript('mode: [frame]\n10: [off]\n2: [on, 4.5]')
    assert control.sequencerMode is DAQ.SequencerMode.Frame
    assert list(control.sequncerDict.items()) == [
        (2, ['on', 4.5]),
        (10, ['off']),
    ]


def test_time_script_accepts_fractional_triggers():
    control = DAQ.DAQControl()
    control.parseTextScript('mode: [time]\n0.25: [on, 1]\n1.5: [off]')
    assert control.sequencerMode is DAQ.SequencerMode.Time
    assert list(control.sequncerDict) == [0.25, 1.5]


def test_invalid_script_does_not_replace_active_sequence():
    control = DAQ.DAQControl()
    control.parseTextScript('mode: [frame]\n1: [off]')
    with pytest.raises(ValueError):
        control.parseTextScript('mode: [frame]\n2: [on, 5]')
    assert list(control.sequncerDict.items()) == [(1, ['off'])]


@pytest.mark.parametrize('script', [
    "mode: [frame]\n0: [on, __import__('os').getcwd()]",
    'mode: [frame]\n0.5: [off]',
    'mode: [time]\n-1: [off]',
    'mode: [frame]\n0: [on, 5]',
    'mode: [frame]\n0: [off, 1]',
    '0: [off]',
])
def test_invalid_or_executable_scripts_are_rejected(script):
    with pytest.raises(ValueError):
        DAQ.DAQControl().parseTextScript(script)
