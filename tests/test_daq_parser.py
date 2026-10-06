import pytest

import DAQ_control as DAQ


class FakeDaq:
    def __init__(self):
        self.feedback = []

    def voltageToDACBits(self, volts, dacNumber, is16Bits):
        return dacNumber, volts

    def getFeedback(self, *commands):
        self.feedback.append(commands)


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


def test_daq_voltage_state_and_safe_off_cover_both_outputs(monkeypatch):
    monkeypatch.setattr(DAQ.u3, 'DAC0_8', lambda value: ('dac0', value))
    monkeypatch.setattr(DAQ.u3, 'DAC1_8', lambda value: ('dac1', value))
    control = DAQ.DAQControl()
    control.daq = FakeDaq()

    control._executeCommand(['on', 2.5])
    assert control.currentVoltage == 2.5
    assert control.daq.feedback[-1] == (
        ('dac0', (0, 2.5)),
        ('dac1', (1, 2.5)),
    )

    assert control.safe_off()
    assert control.currentVoltage == 0
    assert control.daq.feedback[-1] == (
        ('dac0', (0, 0.0)),
        ('dac1', (1, 0.0)),
    )


def test_reversal_detector_handles_short_and_stationary_trails():
    detector = DAQ.ReversalDetector()
    detector.trailLimit = 100
    detector.animalLength_mm = 2.5
    detector.velocityHistoryPercentage = 0
    detector.reversalThresholdRadian = 90

    assert not detector.detectReversal([])
    assert not detector.detectReversal([[0, 0]])
    assert not detector.detectReversal([[0, 0], [0, 0], [0, 0]])


def test_reversal_detector_distinguishes_forward_and_reverse_motion():
    detector = DAQ.ReversalDetector()
    detector.trailLimit = 100
    detector.animalLength_mm = 2.5
    detector.velocityHistoryPercentage = 50
    detector.reversalThresholdRadian = 90

    assert not detector.detectReversal([
        [0, 0], [1, 0], [2, 0], [3, 0],
    ])
    assert detector.detectReversal([
        [0, 0], [1, 0], [2, 0], [3, 0], [2.5, 0],
    ])


class FlakyDaq(FakeDaq):
    """getFeedback fails while `failing` is set, like a USB timeout or an unplugged LabJack."""

    def __init__(self):
        super().__init__()
        self.failing = False

    def getFeedback(self, *commands):
        if self.failing:
            raise RuntimeError('LabJack USB timeout')
        super().getFeedback(*commands)


def _sequencer(script, monkeypatch):
    monkeypatch.setattr(DAQ.u3, 'DAC0_8', lambda value: ('dac0', value))
    monkeypatch.setattr(DAQ.u3, 'DAC1_8', lambda value: ('dac1', value))
    control = DAQ.DAQControl()
    control.daq = FlakyDaq()
    control.daqMode = DAQ.DAQMode.Sequencer
    control.parseTextScript(script)
    control.start([0.0, 0.0])
    return control


def test_daq_error_during_recording_is_reported_not_raised(monkeypatch):
    control = _sequencer('mode: [frame]\n2: [on, 3.0]\n4: [off]', monkeypatch)
    assert control.update_safely(frameNum=0, frameTime=0.0)
    control.daq.failing = True
    assert control.update_safely(frameNum=2, frameTime=0.1) is False   # the recording keeps going
    assert control.currentVoltage == 0                                   # and the light is treated as off
    control.daq.failing = False
    assert control.update_safely(frameNum=3, frameTime=0.2)


def test_disconnect_during_recording_does_not_break_the_next_command(monkeypatch):
    control = _sequencer('mode: [frame]\n1: [on, 2.0]', monkeypatch)
    control.close()
    assert not control.isConnected()
    control._executeCommand(['on', 2.0])        # the recording thread may still be mid-command
    assert control.currentVoltage == 2.0


def test_sequencer_survives_safe_off_from_another_thread(monkeypatch):
    import threading
    control = _sequencer('mode: [time]\n0.5: [on, 1.0]\n1.0: [off]\n1.5: [on, 1.0]', monkeypatch)
    stop = threading.Event()

    def clear_repeatedly():
        while not stop.is_set():
            control.safe_off()

    other = threading.Thread(target=clear_repeatedly)
    other.start()
    try:
        for i in range(3000):
            if not control.sequnceDictRunning:
                control.start([0.0, 0.0])
            control.updateSequencer(frameNum=i, frameTime=2.0)   # raised StopIteration before the fix
    finally:
        stop.set()
        other.join()
