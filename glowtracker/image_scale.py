"""Settings in pixels, as they apply to the image the camera sends now.

Everything in the settings is in full-sensor pixels: the calibrated pixel size, the capture radius,
the tracking ROI. With camera binning an image pixel covers several sensor pixels, so these
are converted here, in one place, and a change of binning needs no new calibration. Without
binning they are the settings as they are. No Kivy import.
"""
from __future__ import annotations


def binning(app) -> int:
    """How many sensor pixels one image pixel covers in each direction."""
    return max(1, int(getattr(app, 'cameraBinning', 1) or 1))


def pixel_size(app) -> float:
    return app.config.getfloat('Camera', 'pixelsize') * binning(app)


def capture_radius(app) -> int:
    return max(1, round(app.config.getint('Tracking', 'capture_radius') / binning(app)))


def roi(app) -> tuple[int, int]:
    factor = binning(app)
    return (max(1, round(app.config.getint('Tracking', 'roi_x') / factor)),
            max(1, round(app.config.getint('Tracking', 'roi_y') / factor)))


def area(app) -> int:
    return max(1, round(app.config.getint('Tracking', 'area') / binning(app) ** 2))


def tracking_bin_factor(app) -> int:
    """Tracking shrinks the image before detection. Camera binning has done part of that."""
    return max(1, round(app.config.getint('Tracking', 'binning') / binning(app)))


def dual_color_translation(app) -> tuple[float, float]:
    factor = binning(app)
    return (app.config.getfloat('DualColor', 'translation_x') / factor,
            app.config.getfloat('DualColor', 'translation_y') / factor)
