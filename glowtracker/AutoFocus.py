from typing import List
import cv2
import numpy as np
from enum import Enum

class FocusEstimationMethod(Enum):
    # Variance of Laplace
    VarianceOfLaplace = 'VarianceOfLaplace'
    # Tenengrad
    Tenengrad = 'Tenengrad'
    # Brenner's
    Brenners = 'Brenners'
    # Energy of Laplacian
    EnergyOfLaplacian = 'EnergyOfLaplacian'
    # Modified Laplacian
    ModifiedLaplace = 'ModifiedLaplace'
    # Sum of High-Frequency DCT Coefficient
    SumOfHighDCT = 'SumOfHighDCT'


def estimateFocus(focusEstimationMethod: FocusEstimationMethod, image: np.ndarray) -> float:
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
        diff = (image - shifted)**2
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

    def __init__(
        self, 
        KP: float = 0.5, 
        KI: float = 0.01,
        KD: float = 0.1,
        SP: float = 1000,
        focusEstimationMethod: FocusEstimationMethod = FocusEstimationMethod.SumOfHighDCT,
        minStepDist: float = 0.0001,
        integralLifeTime: int = 0,
        smoothingWindow: int = 1,
        minStepBeforeChangeDir: int = 0,
        acceptableErrorPercentage: float = 0.05
    ) -> None:
        """Initialize attributes

        Args:
            KP (float, optional): Proportional constant. Defaults to 0.5.
            KI (float, optional): Integral constant. Defaults to 0.01.
            KD (float, optional): Differential constant. Defaults to 0.1.
            SP (float, optional): Setpoint. Defaults to 1000.
            focusEstimationMethod (FocusEstimationMethod, optional): Which kind of focus estimation mode to use. Defaults to focusEstimationMethod.SumOfHighDCT
            minStepDist (float, optional): Minimum move distance. Defaults to 0.0001.
            integralLifeTime (int, optional): How far back (iteration step) do we count the values into the integral. Defaults to 0 means no limit.
            smoothingWindow (int, optional): How far back (iteration step) do we smooth the PV over. Defaults to 1, which means no smoothing.
            minStepBeforeChangeDir (int, optional): Minimum number of steps to perform before allowing changing of direction. Defaults to 0, which means can change direction at any time.
            acceptableErrorPercentage (float, optional): Acceptable error ratio between PV / SP . Defaults to 0.05 which means PV is acceptable within 0.95 <= PV / SP <= 1.05 range
        """

        self.KP = KP
        self.KI = KI
        self.KD = KD
        self.SP: float = SP
        self.focusEstimationMethod = focusEstimationMethod
        self.minStepDist: float = minStepDist
        self.integralLifeTime: int = integralLifeTime
        self.smoothingWindow: int = smoothingWindow
        # Blending weight for PV smoothing
        self.WEIGHT_MAX = 9
        self.WEIHT_MIN = 1
        self.minStepBeforeChangeDir: int = minStepBeforeChangeDir
        self.acceptableErrorPercentage: float = acceptableErrorPercentage
        
        
        self.posLog: List[float] = []
        self.focusLog: List[float] = []
        self.errorLog: List[float] = []

        self.direction: int = 1
        self.directionResetCounter = 0


    def executePIDStep(self, image: np.ndarray, pos: float) -> float:
        """Perform one PID control step based on current image and lens position.

        Args:
            image (np.ndarray): gray-scaled image
            pos (float): stage z-axis position

        Returns:
            relPosZ (float): estimated **relative** z-axis position to move to
        """

        # Estimate focus the image at current position
        PV = estimateFocus(self.focusEstimationMethod, image)

        # Apply a linear, weighted average to PV with emphasis on recent data
        focuses = [PV]
        if self.smoothingWindow > 1:
            focuses = self.focusLog[-(self.smoothingWindow - 1):] + focuses
        focuses = np.array(focuses)

        # Compute linear weight
        t = np.array([1])

        if (len(focuses) > 1):
            t = np.arange(len(focuses)) / float( min(1, len(focuses) - 1) )
        
        weights = self.WEIHT_MIN + (self.WEIGHT_MAX - self.WEIHT_MIN) * t

        PV = sum(focuses * weights) / sum(weights)

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

            # If the PV is not close enough to the SP (percentage-wise), then execute 
            errorRatio = abs( PV / self.SP - 1.0)
            if errorRatio > self.acceptableErrorPercentage:
                
                # PID calculations
                if self.integralLifeTime > 0:
                    self.integral = np.sum(self.errorLog[-self.integralLifeTime:])
                else:
                    self.integral = np.sum(self.errorLog)

                U = (self.KP * err) + (self.KI * self.integral) + (self.KD * derivative)

                # Decide direction. If the error is increasing then we should flip direction.
                if self.directionResetCounter > self.minStepBeforeChangeDir:
                    
                    # Compute derivative of past error up to histLength
                    pastErrs = list(zip( self.errorLog[1:], self.errorLog ))[-(self.minStepBeforeChangeDir + 1):]
                    diffs = list( map( lambda x: x[0] - x[1], pastErrs ) )

                    # The averaing error is increasing
                    if sum(diffs) > 0:
                        
                        self.direction = self.direction * -1
                        self.directionResetCounter = 0
                
                self.directionResetCounter += 1

                U = U * self.direction
                

        # Record
        self.focusLog.append(PV)
        self.errorLog.append(err)
        self.posLog.append(pos)

        return U

