from typing import List
import cv2
import numpy as np
from enum import Enum


class FocusEstimationMethod(Enum):
    # Variance of Laplace
    VarianceOfLaplace = "VarianceOfLaplace"
    # Tenengrad
    Tenengrad = "Tenengrad"
    # Brenner's
    Brenners = "Brenners"
    # Energy of Laplacian
    EnergyOfLaplacian = "EnergyOfLaplacian"
    # Modified Laplacian
    ModifiedLaplace = "ModifiedLaplace"
    # Sum of High-Frequency DCT Coefficient
    SumOfHighDCT = "SumOfHighDCT"


def estimateFocus(
    focusEstimationMethod: FocusEstimationMethod, image: np.ndarray
) -> float:
    """Estimate focus of an image.

    Args:
        image (np.ndarray): gray-scale image
        mode (int, optional): Focus mode. Defaults to 5 which is Sum of High-Frequency DCT Coefficients.

    Returns:
        float: estimated focus
    """
    estimatedFocus: float = 0

    if focusEstimationMethod == FocusEstimationMethod.VarianceOfLaplace:
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        estimatedFocus = laplacian.var()

    elif focusEstimationMethod == FocusEstimationMethod.Tenengrad:
        gx = cv2.Sobel(image, cv2.CV_64F, 1, 0)
        gy = cv2.Sobel(image, cv2.CV_64F, 0, 1)
        gradient_magnitude = gx**2 + gy**2
        estimatedFocus = np.mean(gradient_magnitude)

    elif focusEstimationMethod == FocusEstimationMethod.Brenners:
        shifted = np.roll(image, -2, axis=1)
        diff = (image - shifted) ** 2
        estimatedFocus = np.sum(diff)

    elif focusEstimationMethod == FocusEstimationMethod.EnergyOfLaplacian:
        lap = cv2.Laplacian(image, cv2.CV_64F)
        estimatedFocus = np.sum(np.abs(lap))

    elif focusEstimationMethod == FocusEstimationMethod.ModifiedLaplace:
        mlap = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
        estimatedFocus = np.sum(np.abs(mlap))

    elif focusEstimationMethod == FocusEstimationMethod.SumOfHighDCT:
        resized = cv2.resize(image, (32, 32))  # Small for fast DCT
        dct = cv2.dct(np.float32(resized))
        hf_coeffs = dct[8:, 8:]  # Keep only high-freq block
        estimatedFocus = np.sum(np.abs(hf_coeffs))

    return estimatedFocus


class AutoFocusPID:
    """Hill-climbing autofocus controller.

    Class name is legacy. PID is the wrong shape of controller for a unimodal
    focus signal: the sign of (SP - focus) doesn't tell you which way to move.
    This controller samples focus before/after each step, keeps direction when
    focus improves, and flips + halves the step when focus drops. Step size
    grows on success and shrinks on overshoot, so it converges fast far from
    the peak and is stable near it.
    """

    def __init__(
        self,
        focusEstimationMethod: FocusEstimationMethod = FocusEstimationMethod.SumOfHighDCT,
        minStepDist: float = 0.0001,
        maxStepDist: float | None = None,
        smoothingWindow: int = 1,
        noiseDeadband: float = 0.02,
        KP: float = 0.0, KI: float = 0.0, KD: float = 0.0, SP: float = 0.0,
        integralLifeTime: int = 0, minStepBeforeChangeDir: int = 0,
        acceptableErrorPercentage: float = 0.0,
    ) -> None:
        self.focusEstimationMethod = focusEstimationMethod
        self.minStep: float = abs(minStepDist) if minStepDist else 1e-4
        self.maxStep: float = maxStepDist if maxStepDist is not None else 10.0 * self.minStep
        self.smoothingWindow: int = max(1, smoothingWindow)
        self.noiseDeadband: float = max(0.0, noiseDeadband)

        self.posLog: List[float] = []
        self.focusLog: List[float] = []

        self.direction: int = 1
        self.step: float = self.minStep

    def _smoothedFocus(self) -> float:
        if self.smoothingWindow <= 1:
            return self.focusLog[-1]
        return float(np.mean(self.focusLog[-self.smoothingWindow:]))

    def executePIDStep(self, image: np.ndarray, pos: float) -> float:
        """Compute the next relative z move (mm) given the current image and stage pos."""
        rawFocus = estimateFocus(self.focusEstimationMethod, image)
        self.focusLog.append(rawFocus)
        self.posLog.append(pos)

        if len(self.focusLog) == 1:
            return self.step * self.direction

        N = self.smoothingWindow
        currFocus = self._smoothedFocus()
        if len(self.focusLog) > N:
            prevFocus = float(np.mean(self.focusLog[-N - 1:-1]))
        else:
            prevFocus = self.focusLog[-2]

        delta = currFocus - prevFocus
        ref = max(abs(currFocus), abs(prevFocus), 1e-9)

        if abs(delta) / ref < self.noiseDeadband:
            return 0.0

        if delta > 0:
            self.step = min(self.step * 1.5, self.maxStep)
        else:
            self.direction *= -1
            self.step = max(self.step * 0.5, self.minStep)

        return self.step * self.direction
