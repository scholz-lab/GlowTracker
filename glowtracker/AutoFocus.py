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
    margin against BOTH F_zero AND the opposing probe), and moves there.
    Stateless per event.

    Safety: tracks signed net displacement from the session start. Probes that
    return to start contribute 0 to the budget; only real drift consumes it.
    """

    SAFETY_MAX_PROBE_MM: float = 0.04
    SAFETY_MAX_NET_DISPLACEMENT_MM: float = 2.0

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

        self.netDisplacement: float = 0.0
        self.aborted: bool = False
        self.focusLog: List[float] = []
        self.posLog: List[float] = []

    def estimate(self, image: np.ndarray) -> float:
        return estimateFocus(self.focusEstimationMethod, image)

    def commitMove(self, distance: float) -> bool:
        """Track signed net displacement. Returns False once |net| exceeds the cap."""
        self.netDisplacement += distance
        if abs(self.netDisplacement) > self.SAFETY_MAX_NET_DISPLACEMENT_MM:
            self.aborted = True
            return False
        return True

    def pickBest(self, F_minus: float, F_zero: float, F_plus: float) -> int:
        """Return -1, 0, or +1 indicating which of {z-δ, z, z+δ} has the best focus.

        A probe wins only if it beats BOTH F_zero AND the opposing probe by
        improvementMargin (relative). This requires evidence of a real
        gradient: random noise can only push one comparison over the margin
        at a time, not both. If neither probe meets both criteria, hold.
        """
        margin = 1.0 + self.improvementMargin
        if F_plus > F_zero * margin and F_plus > F_minus * margin:
            return +1
        if F_minus > F_zero * margin and F_minus > F_plus * margin:
            return -1
        return 0
