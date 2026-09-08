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


def _climb(monkeypatch, sigma, driftPerBatch=0.0, z0=0.0, noise=0.0, batches=240, **kwargs):
    """Drive the real control law against a Gaussian focus-vs-Z curve.

    The image handed to executePIDStep carries the current Z, and the patched
    estimator turns it into a focus value, so the loop closes exactly the way it
    does against the camera. Returns (mean |z - zPeak| over the second half,
    final |z - zPeak|, controller). The mean is what says whether the loop is
    holding focus; a single final sample only catches a runaway, because a
    working climb is always dithering around the peak.
    """
    controller = AutoFocusPID(
        focusEstimationMethod=FocusEstimationMethod.SumOfHighDCT,
        coarseStep=0.02,
        minStepBeforeChangeDir=0,
        buffer_n=1,
        smoothingWindow=1,
        **kwargs,
    )
    rng = np.random.default_rng(1)
    state = {'z': z0, 'zPeak': 0.0}

    def focusAt(method, image):
        state['zPeak'] += driftPerBatch
        peak = 2000.0 * math.exp(-0.5 * ((state['z'] - state['zPeak']) / sigma) ** 2)
        return peak * (1.0 + rng.normal(0.0, noise)) if noise else peak

    monkeypatch.setattr(autofocus, 'estimateFocus', focusAt)
    errors = []
    for _ in range(batches):
        state['z'] += controller.executePIDStep(np.float64(state['z']), pos=state['z'])
        errors.append(abs(state['z'] - state['zPeak']))
    return float(np.mean(errors[len(errors) // 2:])), errors[-1], controller


def test_reverse_on_reacquire_stops_the_runaway(monkeypatch):
    """A focus collapse means we stepped away from the peak, so the climb has to
    turn around. Leaving the direction alone re-arms at coarseStep still pointing
    downhill, which makes the next collapse worse -- the loop never comes back.
    """
    sigma = 0.010
    _, driftedAway, oldController = _climb(monkeypatch, sigma, reverseOnReacquire=False)
    heldFocus, _, _ = _climb(monkeypatch, sigma, reverseOnReacquire=True)

    assert driftedAway > 50 * sigma
    assert oldController.direction == 1, 'old behaviour never reverses'
    assert heldFocus < 2 * sigma


def test_reverse_on_reacquire_can_be_switched_off(monkeypatch):
    """The flag has to be honoured in both directions so the two control laws can
    be compared on the same rig.
    """
    for reverse, expected in ((True, -1), (False, 1)):
        controller = AutoFocusPID(
            focusEstimationMethod=FocusEstimationMethod.SumOfHighDCT,
            buffer_n=1,
            reverseOnReacquire=reverse,
        )
        focusValues = iter([1000.0, 100.0])
        monkeypatch.setattr(autofocus, 'estimateFocus', lambda method, image: next(focusValues))
        image = np.zeros((2, 2), dtype=np.uint8)
        controller.executePIDStep(image, pos=0.0)
        controller.executePIDStep(image, pos=0.0)
        assert controller.step == controller.coarseStep
        assert controller.direction == expected


def test_step_growth_lets_the_climb_follow_a_drifting_sample(monkeypatch):
    """With growth disabled the step only ever shrinks, so the loop cannot keep up
    with a sample moving in Z.
    """
    sigma = 0.050
    # Drift faster than the step floor, so a climb that can only shrink is
    # capped below the speed it needs to keep up.
    drift = 4.0 * 0.002
    withoutGrowth, _, _ = _climb(monkeypatch, sigma, driftPerBatch=drift, stepGrowth=1.0)
    withGrowth, _, _ = _climb(monkeypatch, sigma, driftPerBatch=drift, stepGrowth=1.3)
    assert withGrowth < withoutGrowth


def test_step_growth_is_capped_at_the_coarse_step(monkeypatch):
    controller = AutoFocusPID(
        focusEstimationMethod=FocusEstimationMethod.SumOfHighDCT,
        coarseStep=0.02,
        buffer_n=1,
        stepGrowth=1.3,
    )
    controller.step = 0.004
    focusValues = iter([100.0] + [100.0 * 1.5 ** i for i in range(1, 20)])
    monkeypatch.setattr(autofocus, 'estimateFocus', lambda method, image: next(focusValues))
    image = np.zeros((2, 2), dtype=np.uint8)
    for _ in range(20):
        controller.executePIDStep(image, pos=0.0)
    assert controller.step == controller.coarseStep


def test_hold_at_min_step_parks_the_climb_only_when_enabled(monkeypatch):
    """Parking at the floor is only correct for a stationary sample, so it has to
    be opt-in; the default keeps dithering so slow drift is still corrected.

    Growth is disabled here to isolate the gate: with growth on, any batch that
    does not lose focus lifts the step back off the floor, so parking only ever
    lasts a single batch.
    """
    for hold, expectedMoves in ((True, 0.0), (False, None)):
        controller = AutoFocusPID(
            focusEstimationMethod=FocusEstimationMethod.SumOfHighDCT,
            coarseStep=0.02,
            minStepDist=0.002,
            buffer_n=1,
            stepGrowth=1.0,
            holdAtMinStep=hold,
        )
        controller.step = controller.minStepDist
        controller.focusLog = [100.0]
        controller.bestFocus = 100.0
        monkeypatch.setattr(autofocus, 'estimateFocus', lambda method, image: 100.0)
        move = controller.executePIDStep(np.zeros((2, 2), dtype=np.uint8), pos=0.0)
        if expectedMoves is None:
            assert abs(move) == controller.minStepDist
        else:
            assert move == expectedMoves


def test_defaults_are_the_corrected_control_law():
    controller = AutoFocusPID(focusEstimationMethod=FocusEstimationMethod.SumOfHighDCT)
    assert controller.reverseOnReacquire is True
    assert controller.stepGrowth > 1.0
    assert controller.holdAtMinStep is False


def test_dimming_rebases_focus_without_reversing_or_taking_a_coarse_z_step(monkeypatch):
    controller = AutoFocusPID(buffer_n=2, smoothingWindow=4)
    image = np.zeros((2, 2), dtype=np.uint8)
    monkeypatch.setattr(autofocus, 'estimateFocus', lambda *args: 2000.0)
    for _ in range(5):
        controller.executePIDStep(image, 140)
    controller.step = controller.minStepDist
    controller.direction = -1
    history = list(controller.focusLog)
    controller.resetBrightnessReference()
    monkeypatch.setattr(autofocus, 'estimateFocus', lambda *args: 14.0)
    assert controller.executePIDStep(image, 140) == 0
    assert controller.executePIDStep(image, 140) == 0  # learn new baseline at the same Z
    assert controller.focusLog == history + [14.0]
    assert controller.bestFocus == 14.0
    assert controller.direction == -1
    assert controller.step == controller.minStepDist
    controller.executePIDStep(image, 140)
    assert controller.executePIDStep(image, 140) < 0  # same controller continues focusing
    assert controller.step < controller.coarseStep
