"""Right-click menus: markers, measuring and the geometry behind them."""
import json
import types

import pytest

import image_scale
import stage_markers as markers
from Stage_minimap import StageMinimap


def test_markers_are_kept_in_a_file(tmp_path):
    root = str(tmp_path / 'data')
    assert markers.load(root) == []
    markers.add(root, ' worm 2 ', 12.34567, 80.0)
    assert markers.load(root) == [{'name': 'worm 2', 'x': 12.3457, 'y': 80.0}]
    assert markers.next_name(markers.load(root)) == 'Marker 2'
    markers.add(root, 'other', 50, 50)
    markers.remove(root, markers.load(root)[0])
    assert [m['name'] for m in markers.load(root)] == ['other']


@pytest.mark.parametrize('text', ['{nope', json.dumps({'a': 1}), json.dumps([1, {'name': 'x'}])])
def test_a_broken_marker_file_counts_as_none(tmp_path, text):
    (tmp_path / markers.FILE).write_text(text)
    assert markers.load(str(tmp_path)) == []


def test_only_a_near_marker_is_picked():
    near, far = {'name': 'a', 'x': 10.0, 'y': 10.0}, {'name': 'b', 'x': 40.0, 'y': 40.0}
    assert markers.nearest([far, near], 10.4, 10.0, 1.0) is near
    assert markers.nearest([far, near], 20.0, 20.0, 1.0) is None
    assert markers.nearest([], 0, 0, 1.0) is None


def test_offsets_are_from_where_the_stage_centres():
    assert image_scale.offset_from_center((600, 300), 1000, 800) == (100, -100)
    assert image_scale.offset_from_center((800, 400), 1000, 800, True, 'Right') == (50, 0)
    assert image_scale.offset_from_center((300, 400), 1000, 800, True, 'Left') == (50, 0)


def measuring_app(pixelsize, binning=1, units='um'):
    class Config:
        def getfloat(self, section, key):
            return pixelsize

        def get(self, section, key):
            return units
    return types.SimpleNamespace(config=Config(), cameraBinning=binning)


def test_a_measurement_is_in_real_units():
    assert image_scale.measure_text(measuring_app(2.5), (0, 0), (30, 40)) == '125.0 µm (50.0 px)'
    assert image_scale.measure_text(measuring_app(2.5, binning=2), (0, 0), (30, 40)) == '250.0 µm (50.0 px)'
    assert image_scale.measure_text(measuring_app(0.0025, units='mm'), (0, 0), (30, 40)) == '125.0 µm (50.0 px)'
    assert image_scale.measure_text(measuring_app(25.0), (0, 0), (30, 40)) == '1.250 mm (50.0 px)'


def test_an_uncalibrated_measurement_does_not_pretend():
    assert image_scale.measure_text(measuring_app(1.0), (0, 0), (30, 40)) == '50.0 px (not calibrated)'


@pytest.mark.parametrize('zoom', [0, 2])
def test_the_map_reads_positions_back(zoom):
    minimap = StageMinimap(size=(230, 181), pos=(40, 20))
    minimap._last_position_mm = [70.0, 90.0]
    minimap._view_center = [70.0, 90.0]
    minimap.zoom_level = zoom
    px, py = minimap.mm_to_px(60.0, 95.0)
    assert minimap.px_to_mm(px, py) == pytest.approx((60.0, 95.0))
    assert minimap.reaches(60, 95) and not minimap.reaches(-1, 95) and not minimap.reaches(60, 160)


def test_a_shown_area_stays_while_the_stage_moves_away():
    minimap = StageMinimap(size=(230, 181))
    minimap.position_mm = [10.0, 10.0]
    minimap.show_area(100.0, 100.0)
    assert minimap.zoom_level == 1 and minimap.pinned
    minimap.position_mm = [20.0, 20.0]
    assert minimap._view_center == [100.0, 100.0]
    minimap.follow_stage()
    assert not minimap.pinned and minimap._view_center is None or minimap._view_center != [100.0, 100.0]
