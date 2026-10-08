import cv2
import numpy as np


def normalize_image(image):
    source = np.asarray(image)
    if source.size == 0:
        raise ValueError('image must not be empty')

    if np.issubdtype(source.dtype, np.bool_):
        return source.astype(np.float32)

    values = source.astype(np.float32)
    if np.issubdtype(source.dtype, np.integer):
        limits = np.iinfo(source.dtype)
        scale = float(limits.max - limits.min)
        if scale == 0:
            return np.zeros(source.shape, dtype=np.float32)
        values = (values - limits.min) / scale
    else:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError('image contains no finite pixels')
        minimum = float(np.min(finite))
        maximum = float(np.max(finite))
        if minimum < 0 or maximum > 1:
            scale = maximum - minimum
            if scale == 0:
                values = np.zeros(source.shape, dtype=np.float32)
            else:
                values = (values - minimum) / scale

    return np.clip(np.nan_to_num(values), 0.0, 1.0)


def effective_max_brightness(image, configured_max):
    if configured_max is None:
        return np.inf
    source = np.asarray(image)
    if configured_max == 255 \
            and np.issubdtype(source.dtype, np.unsignedinteger) \
            and source.dtype.itemsize > 1:
        return np.iinfo(source.dtype).max
    return configured_max


def texture_buffer_format(image):
    dtype = np.asarray(image).dtype
    if dtype == np.dtype(np.uint8):
        return 'ubyte'
    if dtype == np.dtype(np.uint16):
        return 'ushort'
    if dtype == np.dtype(np.float32):
        return 'float'
    raise TypeError(f'unsupported texture dtype: {dtype}')


def prepare_texture_data(image):
    source = np.asarray(image)
    try:
        buffer_format = texture_buffer_format(source)
        return np.ascontiguousarray(source), buffer_format
    except TypeError:
        converted = np.rint(normalize_image(source) * 255).astype(np.uint8)
        return np.ascontiguousarray(converted), 'ubyte'


def sample_skewness(values) -> float:
    """Population skewness (the same value as scipy.stats.skew with its defaults), several times
    faster on large images. NaNs are ignored; a flat image gives NaN, as SciPy does."""
    x = np.asarray(values).ravel()
    if x.dtype.kind == 'f':
        x = x[~np.isnan(x)]
    if x.size == 0:
        return float('nan')
    x = x.astype(np.float64 if x.dtype == np.float64 else np.float32, copy=False)
    d = x - x.mean(dtype=np.float64)
    m2 = float(np.dot(d, d)) / d.size
    if m2 <= 0:
        return float('nan')
    m3 = float(np.dot(d * d, d)) / d.size
    return m3 / m2 ** 1.5


def brightness_stats(image, step: int = 4) -> dict:
    """Live-analysis values of one frame. Min, max and mean use every pixel (OpenCV, so cheap even
    at 2048 x 2048), so a small bright animal is never missed. Median, 5th/95th percentile and
    skewness use every `step`-th pixel in each direction, as before."""
    image = np.asarray(image)
    if image.ndim == 2 and image.dtype in (np.uint8, np.uint16, np.int16, np.float32, np.float64):
        lo, hi, _, _ = cv2.minMaxLoc(image)
        mean = cv2.mean(image)[0]
    else:
        lo, hi, mean = image.min(), image.max(), image.mean()
    sample = image[::step, ::step]
    p5, median, p95 = np.percentile(sample, [5, 50, 95])
    return {'min': float(lo), 'max': float(hi), 'mean': float(mean), 'median': float(median),
            'skewness': sample_skewness(sample), 'p5': float(p5), 'p95': float(p95)}
