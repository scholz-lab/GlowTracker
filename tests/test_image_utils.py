import numpy as np
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
