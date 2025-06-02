from typing import List
import cv2
import numpy as np
from enum import Enum
import math

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

class Autofocus:

    def __init__(self, initial_step=0.1, min_step=0.001, max_iterations=20):
        
        self.initial_step = initial_step
        self.min_step = min_step
        self.max_iterations = max_iterations
        self.reset()

    def __init__(self, step_size=1.0, min_step=0.1, trend_window=5, damping=0.9):
        self.step_size = step_size
        self.min_step = min_step
        self.damping = damping
        self.direction = 1  # +1 = forward, -1 = backward
        self.trend_window = trend_window
        self.trend_buffer = collections.deque(maxlen=trend_window)
        self.last_focus = None
        self.best_focus = None
        self.best_position = None


    def reset(self):

        self.current_step = self.initial_step
        self.direction = 1  # Start moving forward
        self.iterations = 0
        self.best_focus = None
        self.best_pos = None
        self.last_pos = None
        self.last_focus = None
        self.done = False


    def estimateFocus(self, image):
        resized = cv2.resize(image, (32, 32))  # Small for fast DCT
        dct = cv2.dct(np.float32(resized))
        hf_coeffs = dct[8:, 8:]  # Keep only high-freq block
        estimatedFocus = np.sum(np.abs(hf_coeffs))
        return estimatedFocus

    def execute_one_step(self, pos, image):
        
        # Estimate focus
        focus_value = self.estimateFocus(image)
        
        # Update best focus
        if self.best_focus is None or \
            focus_value > self.best_focus:
            
            self.best_focus = focus_value
            self.best_pos = pos

        if self.last_focus is None:
            # Nudge a little in random direction
            pass

        else:
            # Check avg focus trend if it has been increasin or decreasing
            trend = focus_value - self.last_focus
            self.trend_buffer.append(trend)

            avg_trend = sum(self.trend_buffer) / len(self.trend_buffer)

            if avg_trend < 0:
                # Focus is degrading → reverse direction
                self.direction *= -1
                self.step_size *= self.damping

                # if self.step_size < self.min_step:
                #     self.step_size = self.min_step

        self.last_focus = focus_value

        return pos + self.direction * self.step_size

        if self.last_focus is not None:
            if focus_value < self.last_focus:
                # We got worse → reverse and reduce step
                self.direction *= -1
                self.current_step *= 0.5

        if self.current_step < self.min_step or self.iterations >= self.max_iterations:
            self.done = True
            return None

        # Update state
        self.last_pos = pos
        self.last_focus = focus_value
        self.iterations += 1

        # Return next lens position
        next_pos = pos + self.direction * self.current_step
        return next_pos

    def should_continue(self):
        return not self.done

    def get_result(self):
        return self.best_pos, self.best_focus


class AutoFocusPID:

    def __init__(self, KP: float = 0.5, KI: float = 0.01, KD: float = 0.1, SP: float = 1000, minStepDist: float = 0.001, focusClosenessThreshold: float = 100, integralLifeTime: int = 20 ) -> None:

        self.KP = KP
        self.KI = KI
        self.KD = KD

        self.SP: float = SP
        self.minStepDist: float = minStepDist
        self.focusClosenessThreshold: float = focusClosenessThreshold
        
        self.integralLifeTime: int = integralLifeTime
        
        self.posLog: List[float] = []
        self.focusLog: List[float] = []
        self.errorLog: List[float] = []

        self.direction: int = 1


    def estimateFocus(self, image: np.ndarray, mode: FocusMode = FocusMode.SumOfHighDCT) -> float:
        """Estimate focus of an image.

        Args:
            image (np.ndarray): gray-scale image
            mode (int, optional): Focus mode. Defaults to 5 which is Sum of High-Frequency DCT Coefficients.

        Returns:
            float: estimated focus
        """
        estimatedFocus: float = 0

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
        

    def executePIDStep(self, image: np.ndarray, pos: float, estimateFocusMode: FocusMode = FocusMode.SumOfHighDCT) -> float:
        """Perform one PID control step based on current image and lens position."""
        
        # Estimate focus the image at current position
        PV = self.estimateFocus(image, estimateFocusMode)
        # Compute error
        err = self.SP - PV
        U: float = 0.0

        if len(self.focusLog) == 0:
            # If this is the first time executing, simply move by a minimum distance
            U = self.minStepDist * self.direction

        else:
            prevErr = self.errorLog[-1]
            # Here we assume t to be a discrete time of this function is call. Thus simplify the formula.
            derivative = (err - prevErr)

            # If the difference between current and previous error is still high enough
            #   and the difference between current focus and target focus (SP) is still high
            if abs(derivative) > self.focusClosenessThreshold \
                and abs(err) > self.focusClosenessThreshold:
                
                # PID calculations
                self.integral = np.sum(self.errorLog[-self.integralLifeTime:])

                U = (self.KP * err) + (self.KI * self.integral) + (self.KD * derivative)

                # Decide direction. If the error is increasing then we should flip direction.
                # Maybe we should average from some range to avoid noise?
                if derivative > 0:
                    self. direction = self.direction * -1

                U = U * self.direction
                
                print(f'\t{PV:.4f}, {err:.4f}, {self.KP * err:.4f}, {self.KI * self.integral:.4f}, {self.KD * derivative:.4f}, {U:.4f}, {pos:.4f}')


        # Record
        self.focusLog.append(PV)
        self.errorLog.append(err)
        self.posLog.append(pos)

        newPos = pos + U

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
