import numpy as np

import image_utils
import pytest

from image_utils import (
    effective_max_brightness,
    normalize_image,
    prepare_texture_data,
)


def test_integer_images_are_normalized_by_dtype_range():
    np.testing.assert_allclose(
        normalize_image(np.array([0, 255], dtype=np.uint8)),
        [0.0, 1.0],
    )
    np.testing.assert_allclose(
        normalize_image(np.array([0, 65535], dtype=np.uint16)),
        [0.0, 1.0],
    )


def test_legacy_8_bit_max_expands_for_wider_unsigned_images():
    image = np.zeros((2, 2), dtype=np.uint16)
    assert effective_max_brightness(image, 255) == 65535
    assert effective_max_brightness(image, 4095) == 4095


def test_texture_data_preserves_camera_integer_depth():
    image8 = np.zeros((2, 2), dtype=np.uint8)
    image16 = np.zeros((2, 2), dtype=np.uint16)
    prepared8, format8 = prepare_texture_data(image8)
    prepared16, format16 = prepare_texture_data(image16)
    assert prepared8.dtype == np.uint8
    assert format8 == 'ubyte'
    assert prepared16.dtype == np.uint16
    assert format16 == 'ushort'


def test_unsupported_texture_dtype_is_safely_converted():
    image = np.array([[0.0, 2.0]], dtype=np.float64)
    prepared, buffer_format = prepare_texture_data(image)
    assert prepared.dtype == np.uint8
    assert prepared.tolist() == [[0, 255]]
    assert buffer_format == 'ubyte'


def _old_live_analysis(image):
    """What computeLiveAnalysisValues did before (SciPy skew, full-image min/max/mean)."""
    from scipy.stats import skew
    sample = image[::4, ::4]
    return {'min': np.min(image), 'max': np.max(image), 'mean': np.mean(image), 'median': np.median(sample),
            'skewness': skew(sample, axis=None, nan_policy='omit'),
            'p5': np.percentile(sample, 5), 'p95': np.percentile(sample, 95)}


@pytest.mark.parametrize('dtype,shape', [(np.uint8, (400, 400)), (np.uint16, (512, 640)), (np.float32, (300, 200))])
def test_brightness_stats_match_the_previous_computation(dtype, shape):
    from image_utils import brightness_stats
    rng = np.random.default_rng(1)
    image = (rng.gamma(2.0, 8.0, shape)).astype(dtype)
    image[100:110, 50:57] = 200                     # a small bright spot, like the worm
    new, old = brightness_stats(image), _old_live_analysis(image)
    for key in old:
        assert np.isclose(new[key], float(old[key]), rtol=1e-4, atol=1e-4), key


def test_skewness_edge_cases_match_scipy():
    import warnings
    from scipy.stats import skew
    from image_utils import sample_skewness
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        assert np.isnan(sample_skewness(np.full((8, 8), 5, np.uint8))) and np.isnan(skew(np.full(64, 5.0)))
    values = np.array([1.0, 2.0, np.nan, 10.0, 3.0])
    assert np.isclose(sample_skewness(values), skew(values, nan_policy='omit'))


@pytest.fixture
def twelve_bit():
    image_utils.set_significant_bits(12)
    yield
    image_utils.set_significant_bits(16)


def test_12_bit_pixels_in_16_bit_arrays_use_the_full_scale(twelve_bit):
    image = np.array([0, 2048, 4095], dtype=np.uint16)
    assert normalize_image(image)[2] == 1.0
    assert normalize_image(image)[1] == pytest.approx(0.5, abs=0.001)
    assert normalize_image(np.array([255], dtype=np.uint8))[0] == 1.0       # 8 bit stays as it is
    prepared, _ = prepare_texture_data(image)
    assert prepared[2] > 65000                                              # white on screen
    assert image[2] == 4095                                                 # the data is not touched


@pytest.mark.parametrize('dark_bg', [True, False])
def test_tracking_finds_the_animal_the_same_in_12_bit(twelve_bit, dark_bg):
    import cv2
    import Microscope_macros as macro
    background, animal = (0.08, 0.6) if dark_bg else (0.7, 0.25)
    frame = np.full((800, 800), background, np.float32)
    mask = np.zeros((800, 800), np.uint8)
    cv2.ellipse(mask, (460, 370), (90, 14), 30, 0, 360, 1, -1)
    frame[mask > 0] = animal
    frame = np.clip(cv2.GaussianBlur(frame, (0, 0), 3) + np.random.default_rng(1).normal(0, 0.01, frame.shape), 0, 1)
    y, x = macro.extractWormsCMS(np.rint(frame * 4095).astype(np.uint16), capture_radius=400, bin_factor=4,
                                 dark_bg=dark_bg)[:2]
    assert (y, x) == pytest.approx((-32, 58), abs=3)
