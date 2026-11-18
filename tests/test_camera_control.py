import numpy as np
from pypylon import pylon

from Basler_control import Camera, CameraGrabParameters, readPFSFile


class GrabResult:
    def __init__(self, image, succeeded=True, timestamp=2500000):
        self.image = image
        self.succeeded = succeeded
        self.TimeStamp = timestamp
        self.released = False

    @property
    def Array(self):
        return self.image

    def GrabSucceeded(self):
        return self.succeeded

    def Release(self):
        self.released = True
        self.image.fill(0)


class CameraTransport:
    def __init__(self, result, grabbing=True):
        self.result = result
        self.grabbing = grabbing
        self.requests = []

    def IsGrabbing(self):
        return self.grabbing

    def RetrieveResult(self, timeout, timeout_handling):
        self.requests.append((timeout, timeout_handling))
        return self.result


def test_camera_wrapper_uses_real_pypylon_base_class():
    assert issubclass(Camera, pylon.InstantCamera)
    parameters = CameraGrabParameters(
        bufferSize=4,
        grabStrategy=pylon.GrabStrategy_OneByOne,
    )
    assert parameters.grabStrategy == pylon.GrabStrategy_OneByOne


def test_retrieved_frame_owns_its_memory_after_result_release():
    source = np.arange(12, dtype=np.uint16).reshape(3, 4)
    expected = source.copy()
    result = GrabResult(source)
    transport = CameraTransport(result)
    success, image, timestamp, retrieved_at = Camera.retrieveGrabbingResult(
        transport
    )
    assert success
    np.testing.assert_array_equal(image, expected)
    assert not np.shares_memory(image, source)
    assert result.released
    assert timestamp == 2.5
    assert retrieved_at is not None
    assert transport.requests == [(1000, pylon.TimeoutHandling_Return)]


def test_unsuccessful_grab_is_released_without_returning_an_image():
    result = GrabResult(np.ones((2, 2), dtype=np.uint8), succeeded=False)
    transport = CameraTransport(result)
    success, image, timestamp, retrieved_at = Camera.retrieveGrabbingResult(
        transport
    )
    assert not success
    assert image is None
    assert timestamp is None
    assert retrieved_at is None
    assert result.released


def test_retrieve_is_idle_when_camera_is_not_grabbing():
    result = GrabResult(np.ones((2, 2), dtype=np.uint8))
    transport = CameraTransport(result, grabbing=False)
    assert Camera.retrieveGrabbingResult(transport) == (
        False, None, None, None
    )
    assert transport.requests == []
    assert not result.released


def test_pfs_reader_extracts_feature_values(tmp_path):
    path = tmp_path / 'camera.pfs'
    path.write_text(
        '# camera settings\n'
        'Width\tInteger\t1024\n'
        'PixelFormat\tEnumeration\tMono16\n'
    )
    assert readPFSFile(path) == {
        'Width': '1024',
        'PixelFormat': 'Mono16',
    }
