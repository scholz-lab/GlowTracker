import math

import numpy as np

import AutoFocus as autofocus
from AutoFocus import AutoFocusPID, FocusEstimationMethod


def test_smoothing_weights_span_the_configured_range(monkeypatch):
    controller = AutoFocusPID(
        focusEstimationMethod=FocusEstimationMethod.VarianceOfLaplace,
        smoothingWindow=4,
        SP=1000.0,
        buffer_n=1,
    )
    controller.focusLog = [10.0, 20.0, 30.0]
    monkeypatch.setattr(autofocus, 'estimateFocus', lambda method, image: 40.0)
    controller.executePIDStep(np.zeros((2, 2), dtype=np.uint8), pos=0)
    positions = np.arange(4) / 3.0
    weights = controller.WEIHT_MIN + (
        controller.WEIGHT_MAX - controller.WEIHT_MIN
    ) * positions
    expected = np.average([10.0, 20.0, 30.0, 40.0], weights=weights)
    assert math.isclose(controller.focusLog[-1], expected)


def test_focus_log_updates_only_when_buffer_is_full(monkeypatch):
    controller = AutoFocusPID(
        focusEstimationMethod=FocusEstimationMethod.VarianceOfLaplace,
        buffer_n=3,
    )
    monkeypatch.setattr(autofocus, 'estimateFocus', lambda method, image: 12.0)
    image = np.zeros((2, 2), dtype=np.uint8)
    assert controller.executePIDStep(image, pos=0) == 0
    assert controller.executePIDStep(image, pos=0) == 0
    assert controller.focusLog == []
    controller.executePIDStep(image, pos=0)
    assert controller.focusLog == [12.0]
