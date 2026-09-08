from itertools import islice

import pytest

from plate_plan import create_run_directory, create_visit_directory, parse_setting, validate_plate, visits


def plate(name='A', enabled=True):
    return {'id': name, 'name': name, 'enabled': enabled,
            'center': [40, 100], 'radius': 10, 'settings': {}}


def test_visits_skip_disabled_plates_and_repeat_in_order():
    plan = [plate('A'), plate('B', False), plate('C')]
    assert [(cycle, p['name']) for cycle, p in visits(plan)] == [(1, 'A'), (1, 'C')]
    assert [(cycle, p['name']) for cycle, p in islice(visits(plan, True), 5)] == [
        (1, 'A'), (1, 'C'), (2, 'A'), (2, 'C'), (3, 'A')]
    assert plan[0]['settings'] == {}


@pytest.mark.parametrize('key,text', [
    ('scan_gain', 'nan'), ('track_interval', '0'), ('search_seconds', ''),
    ('scan_z_frames', '2.5'), ('scan_overlap_w', '100'), ('scan_exposure', '-1'),
])
def test_invalid_edits_are_rejected(key, text):
    with pytest.raises(ValueError):
        parse_setting(key, text)


def test_old_plate_presets_receive_defaults_and_no_enabled_plate_is_rejected():
    settings = validate_plate(plate())['settings']
    assert settings['scan_gain'] == 30
    assert settings['track_gain'] == 22
    assert settings['search_passes'] == 1
    with pytest.raises(ValueError, match='Enable'):
        list(visits([plate(enabled=False)]))


def test_recordings_have_unique_run_plate_visit_directories(tmp_path):
    first = create_run_directory(tmp_path, '../run / name')
    second = create_run_directory(tmp_path, '../run / name')
    assert first != second
    assert first.parent == second.parent == tmp_path
    p = plate('A')
    p['name'] = '../plate / name'
    one = create_visit_directory(first, p, 1)
    two = create_visit_directory(first, p, 2)
    assert one != two
    assert one.parent.parent == first
    with pytest.raises(FileExistsError):
        create_visit_directory(first, p, 1)
