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
    """Periodic 3-point bracket autofocus.

    Class name is legacy. Each AF event probes z+δ and z-δ, picks whichever of
    {z-δ, z, z+δ} has the highest focus metric (subject to an improvement
    margin), and moves there. Stateless per event: no PID memory, no direction
    history. Cumulative travel across the session is capped for safety.

    Per-event motion is bounded to ≤ 4·δ in total: +δ, then -2δ, then at most
    +2δ to settle on the best position. So a single event can never wander far
    regardless of focus measurement noise.
    """

    SAFETY_MAX_PROBE_MM: float = 0.02
    SAFETY_MAX_TOTAL_TRAVEL_MM: float = 2.0

    def __init__(
        self,
        focusEstimationMethod: FocusEstimationMethod = FocusEstimationMethod.SumOfHighDCT,
        probeDelta: float = 0.002,
        improvementMargin: float = 0.02,
        minStepDist: float = 0.0, smoothingWindow: int = 1, noiseDeadband: float = 0.0,
        maxStepDist: float = 0.0, KP: float = 0.0, KI: float = 0.0, KD: float = 0.0,
        SP: float = 0.0, integralLifeTime: int = 0,
        minStepBeforeChangeDir: int = 0, acceptableErrorPercentage: float = 0.0,
    ) -> None:
        self.focusEstimationMethod = focusEstimationMethod
        rawDelta = abs(probeDelta) if probeDelta else 0.002
        self.probeDelta: float = min(rawDelta, self.SAFETY_MAX_PROBE_MM)
        self.improvementMargin: float = max(0.0, improvementMargin)

        self.totalTravel: float = 0.0
        self.aborted: bool = False
        self.focusLog: List[float] = []
        self.posLog: List[float] = []

    def estimate(self, image: np.ndarray) -> float:
        return estimateFocus(self.focusEstimationMethod, image)

    def commitMove(self, distance: float) -> bool:
        """Track cumulative absolute travel. Returns False once the cap is hit."""
        self.totalTravel += abs(distance)
        if self.totalTravel > self.SAFETY_MAX_TOTAL_TRAVEL_MM:
            self.aborted = True
            return False
        return True

    def pickBest(self, F_minus: float, F_zero: float, F_plus: float) -> int:
        """Return -1, 0, or +1 indicating which of {z-δ, z, z+δ} has the best focus.

        A probe wins only if it beats F_zero by at least improvementMargin
        (relative). Otherwise we hold position. This is the natural deadband:
        no spurious moves when the current z is already close to peak.
        """
        threshold = F_zero * (1.0 + self.improvementMargin)
        if F_plus >= F_minus and F_plus > threshold:
            return +1
        if F_minus > F_plus and F_minus > threshold:
            return -1
        return 0
