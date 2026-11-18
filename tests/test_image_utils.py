import numpy as np

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
