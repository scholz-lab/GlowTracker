import json

import pytest

import scan_presets


def plate(name='Plate 1', center=(40, 40), radius=27, **extra):
    return {'id': 'p1', 'name': name, 'center': list(center), 'radius': radius,
            'settings': {'track_interval': 60}, 'enabled': True, 'status': 'Tracked', **extra}


def test_save_and_load_round_trip(tmp_path):
    user = tmp_path / 'runs.json'
    scan_presets.save(str(user), ' My run ', [plate(), plate('Plate 2', (110, 40))], True)
    assert scan_presets.names(str(user), str(tmp_path / 'none.json')) == ['My run']
    loaded = scan_presets.get(str(user), 'My run', str(tmp_path / 'none.json'))
    assert [p['name'] for p in loaded['plates']] == ['Plate 1', 'Plate 2']
    assert loaded['repeat_run'] is True
    assert all(p['status'] == 'Ready' for p in loaded['plates'])       # statuses are not kept
    assert loaded['plates'][0]['settings']['track_interval'] == 60
    assert loaded['plates'][0]['settings']['scan_exposure'] == 100000  # defaults filled in


def test_user_preset_replaces_bundled_one_with_the_same_name(tmp_path):
    bundled = tmp_path / 'bundled.json'
    bundled.write_text(json.dumps({'Demo': {'plates': [plate('Bundled')]}, 'Other': {'plates': [plate()]}}))
    user = tmp_path / 'user.json'
    scan_presets.save(str(user), 'Demo', [plate('Mine')], False)
    assert scan_presets.names(str(user), str(bundled)) == ['Demo', 'Other']
    assert scan_presets.get(str(user), 'Demo', str(bundled))['plates'][0]['name'] == 'Mine'
    assert scan_presets.summary(str(user), 'Other', str(bundled)) == '1 plate · built in'


@pytest.mark.parametrize('name, plates, message', [
    ('  ', [plate()], 'name'),
    ('Run', [], 'Add a plate'),
    ('Run', [plate(radius=0)], 'radius'),
])
def test_save_refuses_bad_runs(tmp_path, name, plates, message):
    with pytest.raises(ValueError, match=message):
        scan_presets.save(str(tmp_path / 'runs.json'), name, plates, False)
    assert not (tmp_path / 'runs.json').exists()


def test_missing_and_broken_files(tmp_path):
    broken = tmp_path / 'broken.json'
    broken.write_text('{not json')
    assert scan_presets.names(str(broken), str(tmp_path / 'none.json')) == []
    with pytest.raises(KeyError):
        scan_presets.get(str(broken), 'Nope', str(tmp_path / 'none.json'))


def test_bundled_demo_runs_are_valid():
    for name in scan_presets.names('/nonexistent/runs.json'):
        assert scan_presets.get('/nonexistent/runs.json', name)['plates']
