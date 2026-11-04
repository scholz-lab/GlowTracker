"""Camera presets: the user's own imaging settings, kept as files."""
import json

import pytest

import camera_presets as presets

BRIGHT = {'ExposureTime': 2000.0, 'Gain': 0.0, 'PixelFormat': 'Mono8'}
TRACKING = {'dark_bg': False, 'threshold': 40}


def test_none_are_predefined(tmp_path):
    assert presets.list_presets(str(tmp_path)) == []


def test_saved_presets_come_back_by_name(tmp_path):
    root = str(tmp_path)
    presets.save(root, 'Fluorescence GCaMP', {'ExposureTime': 30000.0}, {'dark_bg': True, 'threshold': 30})
    presets.save(root, 'brightfield', BRIGHT, TRACKING)
    assert [p['name'] for p in presets.list_presets(root)] == ['brightfield', 'Fluorescence GCaMP']
    assert presets.find(root, 'brightfield')['camera'] == BRIGHT
    assert (tmp_path / 'camera_presets' / 'fluorescence-gcamp.json').exists()
    assert presets.exists(root, 'Fluorescence GCaMP') and not presets.exists(root, 'other')
    presets.delete(root, 'brightfield')
    assert presets.find(root, 'brightfield') is None


def test_a_broken_file_does_not_hide_the_others(tmp_path):
    root = str(tmp_path)
    presets.save(root, 'good', BRIGHT, TRACKING)
    (tmp_path / 'camera_presets' / 'broken.json').write_text('{nope')
    (tmp_path / 'camera_presets' / 'other.json').write_text(json.dumps(['a list']))
    assert [p['name'] for p in presets.list_presets(root)] == ['good']


def test_a_preset_matches_within_what_the_camera_rounds():
    preset = {'name': 'x', 'camera': BRIGHT, 'tracking': TRACKING}
    assert presets.matches(preset, {'ExposureTime': 2003.0, 'Gain': 0.0, 'PixelFormat': 'Mono8'}, TRACKING)
    assert not presets.matches(preset, {'ExposureTime': 3000.0, 'Gain': 0.0, 'PixelFormat': 'Mono8'}, TRACKING)
    assert not presets.matches(preset, {'ExposureTime': 2000.0, 'Gain': 0.0, 'PixelFormat': 'Mono12'}, TRACKING)
    assert not presets.matches(preset, {'ExposureTime': 2000.0, 'Gain': 0.0, 'PixelFormat': 'Mono8'},
                               {'dark_bg': True, 'threshold': 40})
    assert not presets.matches(preset, {'Gain': 0.0, 'PixelFormat': 'Mono8'}, TRACKING)


def test_a_slider_row_survives_limits_that_move_its_slider():
    """The real camera's exposure starts at 8 us: 0.9 on the log scale, so setting the limits moves
    the slider before the row is finished. The emulator's start at 1 us (0) and hid this."""
    from kivy.lang import Builder
    from camera_panel import SettingRow
    rule = """
<SettingRow>:
    Slider:
        id: slider
        min: root.low
        max: root.high
        on_value: root.sliderMoved()
    TextInput:
        id: field
"""
    Builder.load_string(rule, filename='test_setting_row')
    try:
        row = SettingRow(None, 'ExposureTime', 'Exposure', True, (8.0, 10000000.0))
        row.show(30000.0)
        assert row.ids.field.text == '30000'
        assert 10 ** row.ids.slider.value == pytest.approx(30000.0)
    finally:
        Builder.unload_file('test_setting_row')
