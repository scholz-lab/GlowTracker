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
