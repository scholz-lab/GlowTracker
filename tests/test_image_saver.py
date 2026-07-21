import ast
from pathlib import Path
from queue import Queue
from threading import Event

import numpy as np

import image_saver


class CoordinateFile:
    def __init__(self):
        self.rows = []

    def write(self, row):
        self.rows.append(row)


class BlockingCoordinateFile:
    def __init__(self):
        self.close_started = Event()
        self.release_close = Event()

    def close(self):
        self.close_started.set()
        self.release_close.wait()


def test_coordinate_close_returns_after_timeout():
    coordinate_file = BlockingCoordinateFile()
    closed, error, thread = image_saver.close_file_with_timeout(
        coordinate_file, 0.01
    )
    try:
        assert coordinate_file.close_started.is_set()
        assert not closed
        assert error is None
        assert thread.is_alive()
    finally:
        coordinate_file.release_close.set()
        thread.join(1)


def test_coordinate_close_reports_errors():
    class FailingCoordinateFile:
        def close(self):
            raise OSError('flush failed')

    closed, error, thread = image_saver.close_file_with_timeout(
        FailingCoordinateFile(), 1
    )
    assert not closed
    assert isinstance(error, OSError)
    assert not thread.is_alive()


def test_split_recording_uses_channel_suffixes(monkeypatch, tmp_path):
    written = []

    def write_tiff(path, image):
        written.append((path, image.copy()))
        Path(path).write_bytes(b'tiff')

    monkeypatch.setattr(image_saver.tifffile, 'imwrite', write_tiff)
    image_queue = Queue()
    image_queue.put({'img': np.zeros((2, 2)), 'idx': 3, 'channel': 0})
    image_queue.put({'img': np.ones((2, 2)), 'idx': 4, 'channel': 1})
    image_queue.put({'img': np.full((2, 2), 2), 'idx': 4, 'channel': 2})
    status_queue = Queue()
    stopped = Event()
    failed = Event()
    stopped.set()

    image_saver.save_worker(
        image_queue,
        tmp_path,
        'basler_{}.tiff',
        stopped,
        status_queue,
        failed,
    )

    assert [Path(path).name for path, _ in written] == [
        'basler_3.part.tiff',
        'basler_4-main.part.tiff',
        'basler_4-minor.part.tiff',
    ]
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        'basler_3.tiff',
        'basler_4-main.tiff',
        'basler_4-minor.tiff',
    ]
    assert [status_queue.get_nowait()[:3] for _ in range(3)] == [
        ('saved', 3, 0),
        ('saved', 4, 1),
        ('saved', 4, 2),
    ]
    assert not failed.is_set()


def test_write_failure_is_reported_and_stops_worker(monkeypatch, tmp_path):
    attempts = []

    def fail_write(path, image):
        attempts.append(Path(path).name)
        raise OSError('disk full')

    monkeypatch.setattr(image_saver.tifffile, 'imwrite', fail_write)
    image_queue = Queue()
    image_queue.put({'img': np.zeros((2, 2)), 'idx': 8, 'channel': 0})
    image_queue.put({'img': np.zeros((2, 2)), 'idx': 9, 'channel': 0})
    status_queue = Queue()
    stopped = Event()
    failed = Event()

    image_saver.save_worker(
        image_queue,
        tmp_path,
        'basler_{}.tiff',
        stopped,
        status_queue,
        failed,
    )

    assert attempts == ['basler_8.part.tiff']
    assert failed.is_set()
    status = status_queue.get_nowait()
    assert status[:3] == ('failed', 8, 0)
    assert 'OSError: disk full' in status[3]
    assert list(tmp_path.iterdir()) == []


def test_publish_failure_is_reported_and_removes_partial_file(
        monkeypatch, tmp_path):
    def write_tiff(path, image):
        Path(path).write_bytes(b'partial')

    def fail_replace(source, destination):
        raise OSError('rename failed')

    monkeypatch.setattr(image_saver.tifffile, 'imwrite', write_tiff)
    monkeypatch.setattr(image_saver.os, 'replace', fail_replace)
    image_queue = Queue()
    image_queue.put({'img': np.zeros((2, 2)), 'idx': 10, 'channel': 0})
    status_queue = Queue()
    stopped = Event()
    failed = Event()

    image_saver.save_worker(
        image_queue,
        tmp_path,
        'basler_{}.tiff',
        stopped,
        status_queue,
        failed,
    )

    assert failed.is_set()
    status = status_queue.get_nowait()
    assert status[:3] == ('failed', 10, 0)
    assert 'OSError: rename failed' in status[3]
    assert list(tmp_path.iterdir()) == []


def test_coordinates_wait_for_all_channels_and_preserve_frame_order():
    coordinate_file = CoordinateFile()
    acknowledgements = image_saver.SaveAcknowledgements(coordinate_file)
    acknowledgements.add(0, 'frame 0\n', (1, 2))
    acknowledgements.add(1, 'frame 1\n', (0,))

    acknowledgements.saved(1, 0)
    acknowledgements.saved(0, 1)
    assert coordinate_file.rows == []

    acknowledgements.saved(0, 2)
    assert coordinate_file.rows == ['frame 0\n', 'frame 1\n']
    assert acknowledgements.saved_frames == 2
    assert acknowledgements.failed_frames == 0


def test_failed_frame_has_no_coordinates_but_later_saved_frame_does():
    coordinate_file = CoordinateFile()
    acknowledgements = image_saver.SaveAcknowledgements(coordinate_file)
    acknowledgements.add(0, 'frame 0\n', (0,))
    acknowledgements.add(1, 'frame 1\n', (0,))

    acknowledgements.saved(1, 0)
    acknowledgements.failed(0)

    assert coordinate_file.rows == ['frame 1\n']
    assert acknowledgements.saved_frames == 1
    assert acknowledgements.failed_frames == 1
    assert acknowledgements.pending_count == 0


def test_recording_coordinates_include_voltage_only_through_acknowledgements():
    source = (
        Path(__file__).resolve().parents[1]
        / 'glowtracker'
        / 'GlowTracker.py'
    ).read_text()
    tree = ast.parse(source)
    record_button = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == 'RecordButton'
    )
    receive = next(
        node for node in record_button.body
        if isinstance(node, ast.FunctionDef)
        and node.name == 'receiveImageCallback'
    )
    called_attributes = {
        node.func.attr
        for node in ast.walk(receive)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    referenced_attributes = {
        node.attr for node in ast.walk(receive)
        if isinstance(node, ast.Attribute)
    }

    assert 'currentVoltage' in referenced_attributes
    assert 'add' in called_attributes
    assert 'write' not in called_attributes
    assert 'percentile_95 daqVol' in source
