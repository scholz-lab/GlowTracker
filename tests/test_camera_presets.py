"""Camera presets: the user's own imaging settings, kept as files."""
import json

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
