from typing import List
import cv2
import numpy as np
from enum import Enum

class FocusMode(Enum):
    # Variance of Laplace
    VarianceOfLaplace = 0
    # Tenengrad
    Tenengrad = 1
    # Brenner's
    Brenners = 2
    # Energy of Laplacian
    EnergyOfLaplacian = 3
    # Modified Laplacian
    ModifiedLaplace = 4
    # Sum of High-Frequency DCT Coefficient
    SumOfHighDCT = 5


class AutoFocusPID:

    def __init__(self, KP: float = 0.5, KI: float = 0.01, KD: float = 0.1, errorThreshold: float = 0.001) -> None:

        self.KP = KP
        self.KI = KI
        self.KD = KD
        self.errorThreashold: float = errorThreshold
        
        self.integral: float = 0

        self.focusLog: List[float] = []
        self.errorLog: List[float] = []
        self.posLog: List[float] = []

        self.direction: int = 1


    def estimateFocus(self, image: np.ndarray, mode: FocusMode = FocusMode.SumOfHighDCT) -> float:
        """Estimate focus of an image.

        Args:
            image (np.ndarray): gray-scale image
            mode (int, optional): Focus mode. Defaults to 5 which is Sum of High-Frequency DCT Coefficients.

        Returns:
            float: estimated focus
        """
        estimatedFocus = 0

        if mode == FocusMode.VarianceOfLaplace:

            laplacian = cv2.Laplacian(image, cv2.CV_64F)
            estimatedFocus = laplacian.var()

        elif mode == FocusMode.Tenengrad:
            gx = cv2.Sobel(image, cv2.CV_64F, 1, 0)
            gy = cv2.Sobel(image, cv2.CV_64F, 0, 1)
            gradient_magnitude = gx**2 + gy**2
            estimatedFocus = np.mean(gradient_magnitude)

        elif mode == FocusMode.Brenners:
            shifted = np.roll(image, -2, axis=1)
            diff = (image - shifted)**2
            estimatedFocus = np.sum(diff)
        
        elif mode == FocusMode.EnergyOfLaplacian:
            lap = cv2.Laplacian(image, cv2.CV_64F)
            estimatedFocus = np.sum(np.abs(lap))
        
        elif mode == FocusMode.ModifiedLaplace:
            mlap = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
            estimatedFocus = np.sum(np.abs(mlap))
        
        elif mode == FocusMode.SumOfHighDCT:
            resized = cv2.resize(image, (32, 32))  # Small for fast DCT
            dct = cv2.dct(np.float32(resized))
            hf_coeffs = dct[8:, 8:]  # Keep only high-freq block
            estimatedFocus = np.sum(np.abs(hf_coeffs))

        return estimatedFocus
        

    def executePIDStep(self, image: np.ndarray, currentPos: float, dt: float = 1, mode: FocusMode = FocusMode.SumOfHighDCT) -> float:
        """Perform one PID control step based on current image and lens position."""
        
        focus_measure = self.estimateFocus(image, mode)

        # If this is the first time executing
        prevFocus: float = focus_measure
        prevError: float = 0

        if len(self.focusLog) > 0:
            prevFocus = self.focusLog[-1]
            prevError = self.errorLog[-1]
            
        # Compute error as a simple gradient.
        error = (focus_measure - prevFocus) / dt

        # # Auto direction reversal
        # if error < 0:
        #     self.direction *= -1
        #     error = abs(error)  # Flip the gradient to remain positive


        # PID calculations
        self.integral += error

        derivative = (error - prevError)

        pid_output = (self.KP * error) + (self.KI * self.integral) + (self.KD * derivative)

        # Update lens position based on PID output
        newPos = currentPos + pid_output

        print(f'\t{focus_measure:.4f}, {error:.4f}, {self.KP * error:.4f}, {self.KI * self.integral:.4f}, {self.KD * derivative:.4f}')


        # Record
        self.focusLog.append(focus_measure)
        self.errorLog.append(error)
        self.posLog.append(newPos)

        return newPos 


    def isStable(self, recentSteps: int = 3) -> bool:
        """Check if all recent-step errors are less than threshold"""

        if len(self.errorLog) < recentSteps:
            # Not enough history
            return False
        
        recentErrors = self.errorLog[-recentSteps:]

        return all(abs(e) < self.errorThreashold for e in recentErrors)


if __name__ == '__main__':

    autoFocusPID = AutoFocusPID(Kp=0.5, Ki=0.01, Kd=0.1)

    # Get current stage-z pos
    pos_z = 0.0

    MAX_STEP = 10

    while not autoFocusPID.is_focus_stable() \
        and len(autoFocusPID.focusLog) < MAX_STEP:

        # Take an image at the current position
        image = np.array([], np.float32)
        
        new_pos_z = autoFocusPID.executePIDStep(image, pos_z, dt= 1)

        # Move lens to new lens_position
        stage.move_absolute(new_pos_z)
