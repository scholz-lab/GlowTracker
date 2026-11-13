"""Settings in pixels, as they apply to the image the camera sends now.

Everything in the settings is in full-sensor pixels: the calibrated pixel size, the capture radius,
the tracking ROI. With camera binning an image pixel covers several sensor pixels, so these
are converted here, in one place, and a change of binning needs no new calibration. Without
binning they are the settings as they are. No Kivy import.
"""
from __future__ import annotations

import math


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


def offset_from_center(spot, width: int, height: int, dual_color: bool = False, main_side: str = 'Right') -> tuple:
    """An image spot's (x, y) distance from where the stage is centred, in image pixels. With dual
    colour that is the middle of the main half."""
    middle = width / 2
    if dual_color:
        middle = width * 3 / 4 if main_side == 'Right' else width / 4
    return spot[0] - middle, spot[1] - height / 2


def measure_text(app, a, b) -> str:
    """The distance between two image spots, for showing on the image."""
    pixels = math.hypot(b[0] - a[0], b[1] - a[1])
    if app.config.getfloat('Camera', 'pixelsize') == 1.0:
        return f'{pixels:.1f} px (not calibrated)'
    in_mm = app.config.get('Calibration', 'step_units') == 'mm'
    micrometres = pixels * pixel_size(app) * (1000.0 if in_mm else 1.0)
    length = f'{micrometres / 1000:.3f} mm' if micrometres >= 1000 else f'{micrometres:.1f} µm'
    return f'{length} ({pixels:.1f} px)'


def dual_color_translation(app) -> tuple[float, float]:
    factor = binning(app)
    return (app.config.getfloat('DualColor', 'translation_x') / factor,
            app.config.getfloat('DualColor', 'translation_y') / factor)
