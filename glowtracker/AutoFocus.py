from typing import List
import cv2
import numpy as np

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


    def estimateFocus(self, image: np.ndarray) -> float:
        """Compute focus measure using Variance of Laplacian."""
        
        laplacian = cv2.Laplacian(image, cv2.CV_64F)

        return laplacian.var()
        

    def executePIDStep(self, image: np.ndarray, currentPos: float, dt: float = 1) -> float:
        """Perform one PID control step based on current image and lens position."""
        
        focus_measure = self.estimateFocus(image)

        # If this is the first time executing
        prevFocus: float = focus_measure
        prevError: float = 0

        if len(self.focusLog) > 0:
            prevFocus = self.focusLog[-1]
            prevError = self.errorLog[-1]
            
        # Compute error as a simple gradient.
        # TODO: Change step size to time base?
        error = (focus_measure - prevFocus) / dt

        # PID calculations
        self.integral += error

        derivative = (error - prevError)

        pid_output = (self.KP * error) + (self.KI * self.integral) + (self.KD * derivative)

        # Update lens position based on PID output
        newPos = currentPos + pid_output

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

    while not autoFocusPID.is_focus_stable():

        # Take an image at the current position
        image = np.array([], np.float32)
        
        new_pos_z = autoFocusPID.executePIDStep(image, pos_z, dt= 1)

        # Move lens to new lens_position
        stage.move_absolute(new_pos_z)
