"""Camera binning: settings stay in sensor pixels, stage moves stay the same in millimetres."""
import types

import cv2
import numpy as np
import pytest

import image_scale
import Microscope_macros as macro


class Config:
    values = {('Camera', 'pixelsize'): 2.5, ('Tracking', 'capture_radius'): 400, ('Tracking', 'roi_x'): 1800,
              ('Tracking', 'roi_y'): 1200, ('Tracking', 'area'): 400, ('Tracking', 'binning'): 4,
              ('DualColor', 'translation_x'): 10.0, ('DualColor', 'translation_y'): -6.0}

    def getint(self, section, key):
        return int(self.values[section, key])

    def getfloat(self, section, key):
        return float(self.values[section, key])


def app(binning=None):
    fake = types.SimpleNamespace(config=Config())
    if binning is not None:
        fake.cameraBinning = binning
    return fake


@pytest.mark.parametrize('fake', [app(), app(1)])
def test_without_binning_the_settings_are_used_as_they_are(fake):
    assert image_scale.pixel_size(fake) == 2.5
    assert image_scale.capture_radius(fake) == 400
    assert image_scale.roi(fake) == (1800, 1200)
    assert image_scale.area(fake) == 400
    assert image_scale.tracking_bin_factor(fake) == 4
    assert image_scale.dual_color_translation(fake) == (10.0, -6.0)


def test_binning_scales_everything_in_pixels():
    fake = app(2)
    assert image_scale.pixel_size(fake) == 5.0
    assert image_scale.capture_radius(fake) == 200
    assert image_scale.roi(fake) == (900, 600)
    assert image_scale.area(fake) == 100                # an area goes with the square
    assert image_scale.tracking_bin_factor(fake) == 2
    assert image_scale.dual_color_translation(fake) == (5.0, -3.0)


def test_nothing_reaches_zero():
    fake = app(4)
    assert image_scale.tracking_bin_factor(fake) == 1
    fake.config = Config()
    fake.config.values = {**Config.values, ('Tracking', 'area'): 3, ('Tracking', 'capture_radius'): 1}
    assert image_scale.area(fake) == 1 and image_scale.capture_radius(fake) == 1


@pytest.mark.parametrize('factor', [2, 3, 4])
@pytest.mark.parametrize('dark_bg', [True, False])
def test_the_stage_moves_the_same_distance_with_binning(factor, dark_bg):
    """The same animal, seen unbinned and binned, must send the stage to the same place."""
    background, animal = (0.08, 0.6) if dark_bg else (0.7, 0.25)
    sensor = np.full((1200, 1200), background, np.float32)
    mask = np.zeros(sensor.shape, np.uint8)
    cv2.ellipse(mask, (700, 540), (130, 22), 30, 0, 360, 1, -1)
    sensor[mask > 0] = animal
    sensor = cv2.GaussianBlur(sensor, (0, 0), 3)
    binned = sensor.reshape(1200 // factor, factor, 1200 // factor, factor).mean(axis=(1, 3))
    moves = []
    for fake, image in ((app(1), sensor), (app(factor), binned)):
        frame = np.clip(image + np.random.default_rng(1).normal(0, 0.01, image.shape), 0, 1)
        y, x = macro.extractWormsCMS(np.rint(frame * 255).astype(np.uint8), image_scale.capture_radius(fake),
                                     bin_factor=image_scale.tracking_bin_factor(fake), dark_bg=dark_bg)[:2]
        matrix, _ = macro.CameraAndStageCalibrator.genImageToStageMatrix(-1.5789, 1, image_scale.pixel_size(fake))
        moves.append(macro.getStageDistances(np.array([-y, x]), matrix))
    # within what one binned pixel, seen through the tracking's own shrinking, can move it
    assert moves[1] == pytest.approx(moves[0], abs=2.5 * 4 * factor)
    assert np.hypot(*moves[0]) > 100                    # and it really is a move
