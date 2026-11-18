from __future__ import annotations

import math

import numpy as np
import pytest

import Microscope_macros as macro
from Microscope_macros import (
    CameraAndStageCalibrator,
    computeAngleBetweenTwo2DVecs,
    createRigidTransformationMat,
    createScaleAndRotationMatrix,
    createTranslationMatrix,
    create_mask,
    cropCenterImage,
    extractWormsCMS,
    find_CMS,
    getStageDistances,
    rotatePointAboutOrig,
    swapMatXYOrder,
)


# ---------------------------------------------------------------------------
# Synthetic image helpers
# ---------------------------------------------------------------------------


def _blob_image(
    shape: tuple[int, int],
    center: tuple[int, int],
    radius: int,
    fg: int,
    bg: int,
    dtype=np.uint8,
) -> np.ndarray:
    """Filled disk at ``center`` of intensity ``fg`` on a ``bg`` background."""
    h, w = shape
    img = np.full(shape, bg, dtype=dtype)
    y, x = np.ogrid[:h, :w]
    mask = (y - center[0]) ** 2 + (x - center[1]) ** 2 <= radius**2
    img[mask] = fg
    return img


# ---------------------------------------------------------------------------
# cropCenterImage
# ---------------------------------------------------------------------------


class TestCropCenterImage:
    def test_crop_returns_requested_shape(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        out = cropCenterImage(img, cropWidth=40, cropHeight=40)
        assert out.shape == (40, 40)

    def test_crop_preserves_center_pixel(self):
        img = np.zeros((100, 100), dtype=np.uint8)
        img[50, 50] = 200
        out = cropCenterImage(img, 20, 20)
        # The center pixel of the cropped image should be the original center.
        assert out[10, 10] == 200

    def test_crop_zero_returns_full_image(self):
        img = np.arange(9).reshape(3, 3).astype(np.uint8)
        out = cropCenterImage(img, 0, 0)
        np.testing.assert_array_equal(out, img)

    def test_crop_larger_than_image_is_clamped(self):
        img = np.zeros((10, 10), dtype=np.uint8)
        out = cropCenterImage(img, 1000, 1000)
        assert out.shape == (10, 10)

    def test_crop_returns_copy_not_view(self):
        img = np.zeros((20, 20), dtype=np.uint8)
        out = cropCenterImage(img, 10, 10)
        out[0, 0] = 255
        assert img[5, 5] == 0  # would be the same pixel if it were a view


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


class TestGeometry:
    def test_swapMatXYOrder_on_2x2(self):
        m = np.array([[1, 2], [3, 4]], dtype=np.float32)
        swapped = swapMatXYOrder(m)
        np.testing.assert_allclose(swapped, [[4, 3], [2, 1]])

    def test_swapMatXYOrder_does_not_mutate_input(self):
        m = np.array([[1, 2], [3, 4]], dtype=np.float32)
        original = m.copy()
        _ = swapMatXYOrder(m)
        np.testing.assert_array_equal(m, original)

    def test_createTranslationMatrix_shape_and_values(self):
        t = createTranslationMatrix(5.0, -3.0)
        assert t.shape == (3, 3)
        np.testing.assert_allclose(t[:, 2], [5.0, -3.0, 1.0])
        np.testing.assert_allclose(t[:2, :2], np.eye(2))

    def test_createScaleAndRotationMatrix_identity(self):
        m = createScaleAndRotationMatrix(scale=1.0, rotation=0.0,
                                         center_rot_x=0.0, center_rot_y=0.0)
        np.testing.assert_allclose(m, np.eye(3), atol=1e-6)

    def test_createScaleAndRotationMatrix_preserves_center_of_rotation(self):
        cx, cy = 7.0, -4.0
        m = createScaleAndRotationMatrix(scale=1.0, rotation=1.1,
                                         center_rot_x=cx, center_rot_y=cy)
        # Rotating the centre point about itself must return the centre.
        p = np.array([cx, cy, 1.0])
        out = m @ p
        np.testing.assert_allclose(out[:2], [cx, cy], atol=1e-5)

    def test_createRigidTransformationMat_pure_translation(self):
        m = createRigidTransformationMat(1.0, 2.0, rotation=0.0)
        np.testing.assert_allclose(m[:, 2], [1.0, 2.0, 1.0])
        np.testing.assert_allclose(m[:2, :2], np.eye(2), atol=1e-6)

    def test_rotatePointAboutOrig_quarter_turn(self):
        out = rotatePointAboutOrig(np.array([1.0, 0.0]), math.pi / 2)
        np.testing.assert_allclose(out, [0.0, 1.0], atol=1e-6)

    def test_rotatePointAboutOrig_half_turn(self):
        out = rotatePointAboutOrig(np.array([1.0, 0.0]), math.pi)
        np.testing.assert_allclose(out, [-1.0, 0.0], atol=1e-6)

    def test_computeAngleBetweenTwo2DVecs_orthogonal(self):
        theta = computeAngleBetweenTwo2DVecs(
            np.array([1.0, 0.0]), np.array([0.0, 1.0])
        )
        assert math.isclose(theta, math.pi / 2, abs_tol=1e-6)

    def test_computeAngleBetweenTwo2DVecs_same_direction(self):
        theta = computeAngleBetweenTwo2DVecs(
            np.array([3.0, 0.0]), np.array([1.0, 0.0])
        )
        assert math.isclose(theta, 0.0, abs_tol=1e-6)

    def test_computeAngleBetweenTwo2DVecs_signed(self):
        # [1,0] -> [0,-1] is -90 deg (cross product negative)
        theta = computeAngleBetweenTwo2DVecs(
            np.array([1.0, 0.0]), np.array([0.0, -1.0])
        )
        assert math.isclose(theta, -math.pi / 2, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# genImageToStageMatrix
# ---------------------------------------------------------------------------


class TestGenImageToStageMatrix:
    def test_identity_case(self):
        # No rotation, +Z normal, pixelSize = 1 -> should be identity apart
        # from the y,x swap.
        scaled, rot_only = CameraAndStageCalibrator.genImageToStageMatrix(
            rotation=0.0, imageNormalDir=+1, pixelSize=1.0
        )
        # Applied to a [dy, dx] pixel vector it should yield the matching
        # [dy, dx] stage vector for pure identity.
        delta = np.array([3.0, -5.0])  # (y, x)
        stage = scaled @ delta
        np.testing.assert_allclose(stage, delta, atol=1e-5)

    def test_pixel_size_scales_output(self):
        pixel_size = 0.1  # um per pixel, say
        scaled, _ = CameraAndStageCalibrator.genImageToStageMatrix(
            rotation=0.0, imageNormalDir=+1, pixelSize=pixel_size
        )
        delta_pixels = np.array([100.0, 100.0])
        stage = scaled @ delta_pixels
        np.testing.assert_allclose(
            np.linalg.norm(stage),
            pixel_size * np.linalg.norm(delta_pixels),
            rtol=1e-5,
        )

    def test_rot_only_matrix_is_orthonormal(self):
        _, rot_only = CameraAndStageCalibrator.genImageToStageMatrix(
            rotation=0.7, imageNormalDir=+1, pixelSize=3.14
        )
        # Rotation-only matrix must preserve vector length.
        v = np.array([4.0, -3.0])
        assert math.isclose(
            np.linalg.norm(rot_only @ v), np.linalg.norm(v), abs_tol=1e-5
        )


# ---------------------------------------------------------------------------
# getStageDistances (thin wrapper around matmul, but callers depend on it)
# ---------------------------------------------------------------------------


class TestGetStageDistances:
    def test_identity_matrix_passthrough(self):
        out = getStageDistances(np.array([2.0, 3.0]), np.eye(2))
        np.testing.assert_allclose(out, [2.0, 3.0])

    def test_respects_rotation(self):
        # 90deg rotation in y,x order: (y, x) -> (x, -y)
        rot = np.array([[0.0, 1.0], [-1.0, 0.0]])
        out = getStageDistances(np.array([1.0, 0.0]), rot)
        np.testing.assert_allclose(out, [0.0, -1.0], atol=1e-6)


# ---------------------------------------------------------------------------
# create_mask
# ---------------------------------------------------------------------------


class TestCreateMask:
    def test_dark_bg_bright_blob_produces_nonempty_mask(self):
        img = _blob_image((200, 200), center=(100, 100), radius=30,
                          fg=230, bg=10)
        mask, resize_factor, _ = create_mask(img, dark_bg=True, bin_factor=4)
        assert mask.ndim == 2
        assert mask.max() == 255
        # Blob should dominate the foreground.
        assert mask.sum() > 0
        assert resize_factor == 0.25

    def test_bright_bg_dark_blob_produces_nonempty_mask(self):
        img = _blob_image((200, 200), center=(100, 100), radius=30,
                          fg=20, bg=240)
        mask, _, _ = create_mask(img, dark_bg=False, bin_factor=4)
        assert mask.max() == 255
        assert mask.sum() > 0

    def test_intermediate_images_returned_when_display(self):
        img = _blob_image((120, 120), center=(60, 60), radius=10,
                          fg=230, bg=10)
        _, _, intermediates = create_mask(img, dark_bg=True,
                                          display=True, bin_factor=4)
        assert isinstance(intermediates, list) and len(intermediates) >= 2

    def test_uint16_and_uint8_images_produce_equivalent_masks(self):
        y, x = np.ogrid[:200, :200]
        disk = ((y - 100) ** 2 + (x - 100) ** 2) <= 25**2
        image8 = np.full((200, 200), 3, dtype=np.uint8)
        image8[disk] = 230
        image16 = np.full((200, 200), 200, dtype=np.uint16)
        image16[disk] = 60000
        mask8, _, _ = create_mask(image8, dark_bg=True, bin_factor=4)
        mask16, _, _ = create_mask(image16, dark_bg=True, bin_factor=4)
        assert mask8.astype(bool).mean() < 0.3
        assert abs(
            mask16.astype(bool).mean() - mask8.astype(bool).mean()
        ) < 0.15


# ---------------------------------------------------------------------------
# find_CMS
# ---------------------------------------------------------------------------


class TestFindCMS:
    def _mask_with_disks(self, shape, centers, radius):
        m = np.zeros(shape, dtype=np.uint8)
        y, x = np.ogrid[: shape[0], : shape[1]]
        for (cy, cx) in centers:
            m[((y - cy) ** 2 + (x - cx) ** 2) <= radius**2] = 255
        return m

    def test_single_blob_near_center(self):
        mask = self._mask_with_disks((200, 200), [(100, 100)], radius=10)
        x, y = find_CMS(mask)
        assert math.isclose(x, 100.0, abs_tol=1.0)
        assert math.isclose(y, 100.0, abs_tol=1.0)

    def test_two_blobs_picks_the_one_closest_to_image_center(self):
        # Current (documented) behaviour: pick the blob closest to image
        # centre, not to the previous centroid. A far and a near blob both
        # big enough to survive the top-K filter.
        mask = self._mask_with_disks(
            (200, 200), [(100, 110), (30, 30)], radius=15
        )
        x, y = find_CMS(mask)
        assert math.isclose(y, 100.0, abs_tol=2.0)
        assert math.isclose(x, 110.0, abs_tol=2.0)

    def test_empty_mask_raises_value_error(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError):
            find_CMS(mask)

    def test_fully_saturated_mask_raises_value_error(self):
        mask = np.full((100, 100), 255, dtype=np.uint8)
        with pytest.raises(ValueError):
            find_CMS(mask)


# ---------------------------------------------------------------------------
# extractWormsCMS
# ---------------------------------------------------------------------------


class TestExtractWormsCMS:
    def test_centered_blob_returns_small_offset(self):
        img = _blob_image((400, 400), center=(200, 200), radius=20,
                          fg=230, bg=10)
        dy, dx, mask = extractWormsCMS(
            img, capture_radius=150, bin_factor=4, dark_bg=True
        )
        # The returned (dy, dx) is in original-image pixels of offset from
        # the centre of the capture crop. For a centred blob it should be
        # ~zero.
        assert abs(dy) <= 2
        assert abs(dx) <= 2
        assert mask.ndim == 2

    def test_offset_blob_returns_correct_sign(self):
        # Blob moved 50 px down and 30 px to the right of image centre.
        img = _blob_image((400, 400), center=(250, 230), radius=20,
                          fg=230, bg=10)
        dy, dx, _ = extractWormsCMS(
            img, capture_radius=150, bin_factor=4, dark_bg=True
        )
        # Same sign as the displacement, magnitude of the right order.
        assert dy > 10
        assert dx > 10

    def test_display_mode_returns_diagnostics(self):
        image = _blob_image(
            (400, 400), center=(200, 200), radius=20, fg=255, bg=0
        )
        try:
            dy, dx, intermediates, mask = extractWormsCMS(
                image,
                capture_radius=100,
                bin_factor=4,
                dark_bg=True,
                display=True,
            )
        finally:
            macro.plt.close('all')
        assert abs(dy) <= 2
        assert abs(dx) <= 2
        assert len(intermediates) >= 2
        assert mask.ndim == 2
