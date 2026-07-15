from __future__ import annotations
import LabJackPython
import u3
import re
from enum import Enum
from collections import OrderedDict
from copy import deepcopy
from typing import List
from Microscope_macros import Vertex2D, Exterior
import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from dataclasses import dataclass
from Microscope_macros import computeAngleBetweenTwo2DVecs

class DAQMode(Enum):
    Off = 'Off'
    Sequencer = 'Sequencer'
    StageProgram = 'StageProgram'
    Reversal = 'Reversal'


class SequencerMode(Enum):
    Frame = 'Frame'
    Time = 'Time'


class StageProgramMode(Enum):
    FourPoint = 'FourPoint'
    Gaussian = 'Gaussian'


class DAQControl():

    @classmethod
    def createAndConnectDaq(cls) -> DAQControl | None:
        """Create a DAQControl class object. Return the object if the connection to the actual DAQ
        is successful. Otherwise, return None.

        Returns:
            DAQControl (DAQControl|None): DAQControl class if successful, otherwise None.
        """
        daqControl = DAQControl()

        try:
            # Connect to DAQ
            daq = u3.U3(debug= False)

            # Instantiate DAQConatrol object
            daqControl.daq = daq
            
            print(f"Using {daqControl.daq.deviceName}, serial: {daqControl.daq.serialNumber}")
            
            # Set to factory default
            daqControl.daq.setDefaults()
            # Calibrate
            daqControl.daq.getCalibrationData()

        except Exception as e:
            print(e)
        
        finally:
            return daqControl


    def __init__(self):
        self.daq: u3.U3 | None = None
        self.sequncerDict : OrderedDict = OrderedDict()
        self.sequnceDictRunning : OrderedDict = OrderedDict()
        # self.isEnable: bool = False
        self.daqMode: DAQMode = DAQMode.Off
        self.sequencerMode: SequencerMode = SequencerMode.Frame
        self.daqStageProgram: DAQStageProgram = DAQStageProgram()
        self.reversalDetector: ReversalDetector = ReversalDetector()
        self.currentVoltage: float = 0
    

    def isConnected(self) -> bool:
        return self.daq is not None


    def close(self):
        if self.isConnected():
            # Check if Windows then call LabJackPython.Close(), else call self.daq.close()
            self.daq.close()
            self.daq = None

    
    def start(self, startRecordPosition: np.ndarray):
        """Reset internal command dict to original to prepare for running.
        """
        self.sequnceDictRunning = deepcopy(self.sequncerDict)
        self.daqStageProgram.startRecordPosition = startRecordPosition
    
    
    def reset(self):
        """Set DAQ values to factory default. Should be call after finished executing a command list.
        """
        if not self.isConnected():
            return

        # Set to factory default
        self.daq.setDefaults(SetToFactoryDefaults= True)
        
        # Manually set DAC0 to 0 (off)
        dac0Val = self.daq.voltageToDACBits(volts= 0, dacNumber= 0, is16Bits= False)
        dac0Command = u3.DAC0_8(dac0Val)
        self.daq.getFeedback(dac0Command)

        # Clean running command queue
        self.sequnceDictRunning.clear()
        
        self.daqStageProgram.startRecordPosition = np.zeros([2], np.float32)
        self.currentVoltage = 0

        
    def parseTextScript(self, text: str) -> None:

        try:
            # Remove empty lines and surrounding whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]

            # Remove trailing commas from each line
            lines = [re.sub(r',$', '', line) for line in lines]

            # Wrap into a dict literal
            preprocessdText = "{\n" + ",\n".join(lines) + "\n}"
            
            # Parse the text to be a dict object. Highlight keywords "on", "off"
            processedDict = eval(preprocessdText, {
                "on": "on", 
                "off": "off", 
                "mode": "mode",
                "frame": "frame",
                "time": "time"
            })

            # Check if empty
            if len(processedDict) == 0:
                self.sequncerDict.clear()
                return

            # Get running mode
            mode = processedDict.pop('mode')[0]
            
            # Sort and convert to OrderedDict
            self.sequncerDict = OrderedDict( {key:val for key, val in sorted(processedDict.items(), key= lambda x: x[0])} )

            if mode == 'frame':
                self.sequencerMode = SequencerMode.Frame

            elif mode == 'time':
                self.sequencerMode = SequencerMode.Time
                
            else:
                raise ValueError(f"Failed to parse DAQ script text: Invalid 'mode' argument. Options are ['frame', 'time']")

        except Exception as e:
            raise ValueError(f"Failed to parse DAQ script text: {e}")
    

    def update(self, frameNum: int = 0, frameTime: float = 0, stagePosition: List[float] = [], posHist: np.ndarray = None) -> None:
        if self.daqMode == DAQMode.Off:
            return
        
        elif self.daqMode == DAQMode.Sequencer:
            self.updateSequencer(frameNum= frameNum, frameTime= frameTime)
        
        elif self.daqMode == DAQMode.StageProgram:
            self.updateStageProgram(stagePosition)
        
        elif self.daqMode == DAQMode.Reversal:
            self.updateReversalDetection(posHist)


    def updateSequencer(self, frameNum: int = 0, frameTime: float = 0) -> None:

        # Seperate this into two cases, one for each DAQMode
        #   Also need stage position input
        if len(self.sequnceDictRunning) == 0 or not self.daqMode == DAQMode.Sequencer:
            return

        if self.sequencerMode == SequencerMode.Frame:
        
            # Get the exact frame command
            frameCommand = self.sequnceDictRunning.pop(frameNum, default= None)

            if frameCommand is not None:
                print(f"Frame {frameNum}:")
                self._executeCommand(frameCommand)
        
        elif self.sequencerMode == SequencerMode.Time:
            
            # Get the first (lowest frame time) command in queue
            commandFrameTime = next(iter(self.sequnceDictRunning))

            if frameTime >= commandFrameTime:

                # Pop the command
                # Get all the commands with time that are lower than the frame time
                commands = []
                while commandFrameTime <= frameTime and len(self.sequnceDictRunning) > 0:

                    # Pop first item (lowest frame time)
                    commandFrameTime, frameCommand = self.sequnceDictRunning.popitem(last= False)
                    commands.append([commandFrameTime, frameCommand])

                    # If the command queue is now empty then stop
                    if len(self.sequnceDictRunning) == 0:
                        break
                    
                    # Get the next one
                    commandFrameTime = next(iter(self.sequnceDictRunning))

                    # If the next commandFrameTime is already higher then break
                    if commandFrameTime > frameTime:
                        break
                
                if len(commands) > 0:
                    # Execute the last command (closest to the frame time)
                    commandFrameTime, frameCommand = commands[-1]
                    print(f"Time {frameTime:.3f} sec:")
                    self._executeCommand(frameCommand)


    def updateStageProgram(self, stagePosition: List[float]) -> None:
        #   We want to evalute this
        vol = self.daqStageProgram.getValue(stagePosition[0], stagePosition[1])
        
        if math.isclose(vol, 0):
            self._executeCommand(frameCommand= ['off'])

        else:
            self._executeCommand(frameCommand= ['on', vol])
            

    def updateReversalDetection(self, posHist: np.ndarray) -> None:
        # Convert posHist to numpy and discard the z-axis position
        #   and update unit from mm to meter.
        trail = np.array(posHist)[:, (0, 1)] * 1e3
        isReversing = self.reversalDetector.detectReversal(trail= trail)

        vol = self.reversalDetector.reversalVoltage if isReversing else self.reversalDetector.forwardVoltage

        if math.isclose(vol, 0):
            self._executeCommand(frameCommand= ['off'])

        else:
            self._executeCommand(frameCommand= ['on', vol])
    

    def _executeCommand(self, frameCommand: list) -> None:
        
        command: list = frameCommand[0]

        if command == 'on':
            if len(frameCommand) != 2:
                print("\"on\" command requires a voltage argument.")
                return
            
            vol = frameCommand[1]

            # TODO: This should be in the setting to support High-voltage DAQ
            # Clip to 0, 4.95
            vol = max( min( vol, 4.95 ), 0 )

            print(f"Light on {vol} vol")

            # Send command to DAQ at DAC0
            dac0Val = self.daq.voltageToDACBits(volts= vol, dacNumber= 0, is16Bits= False)
            dac0Command = u3.DAC0_8(dac0Val)
            self.daq.getFeedback(dac0Command)
            self.currentVoltage = vol
        
        elif command == 'off':
            
            print(f"Light off")

            # Send command to DAQ at DAC0
            dac0Val = self.daq.voltageToDACBits(volts= 0, dacNumber= 0, is16Bits= False)
            dac0Command = u3.DAC0_8(dac0Val)
            self.daq.getFeedback(dac0Command)
            self.currentVoltage = 0


@dataclass
class GaussianParams():
    amplitude: float = 0
    x_mean: float = 0
    x_sigma: float = 0
    y_mean: float = 0
    y_sigma: float = 0


class DAQStageProgram():

    def __init__(self):
        self.mode: StageProgramMode = StageProgramMode.Gaussian
        self.quadVertex: List[Vertex2D] = []
        self.exterior = Exterior.Zero
        self.exteriorConstant: float = 0
        self.isFourPointRelative = False
        self.gaussianParams = GaussianParams()
        self.isGaussianRelative = False
        self.startRecordPosition = np.zeros([2], np.float32)
    
    
    def update(
            self, 
            mode: StageProgramMode | None = None, 
            quadVertex: List[Vertex2D] | None = None, 
            exterior: Exterior | None = None, 
            exteriorConstant: float | None = None, 
            isFourPointRelative: bool | None = None,
            gaussianParams: GaussianParams | None = None, 
            isGaussianRelative: bool | None = None
        ) -> None:
        """Parse variables and process them.

        Args:
            mode (StageProgramMode): StageProgram mode
            quadVertex (List[Vertex2D]): A list of four Vertex2D
            exterior (Exterior): Exterior mode
            exteriorConstant (float): Constant exterior value in case Exterior mode is Constant.
        """

        if mode:
            self.mode = mode
        
        if quadVertex:
            self.quadVertex = quadVertex

            # Sort points in anti-clockwise order starting from btmLeft: btmLeft, btmRight, topRight, topRight
            #   Compute center
            center = np.zeros([2], np.float32)
            for vert in quadVertex:
                center = center + vert.point
            center = center / 4

            #   Sort
            self.quadVertex.sort(key= lambda vertex: math.atan2(vertex.point[1] - center[1], vertex.point[0] - center[0]))
        
        if exterior:
            self.exterior = exterior
        
        if exteriorConstant:
            self.exteriorConstant = exteriorConstant
        
        if isFourPointRelative is not None:
            self.isFourPointRelative = isFourPointRelative
        
        if gaussianParams:
            self.gaussianParams = gaussianParams
        
        if isGaussianRelative is not None: 
            self.isGaussianRelative = isGaussianRelative
    
    
    def getValue(self, x: float, y: float) -> float:
        """Get an interpolated signal value at a given stage position.

        Args:
            x (float): stage x-position
            y (float): stage y-position

        Returns:
            float: bilinear-interpolated voltage
        """

        val = 0

        if self.mode == StageProgramMode.FourPoint:

            currentPosition = np.array([x, y], np.float32)
            if self.isFourPointRelative:
                currentPosition = currentPosition - self.startRecordPosition
                self.quadVertex[0].point
            
            val = Vertex2D.bilerp(
                self.quadVertex[0], 
                self.quadVertex[1], 
                self.quadVertex[2], 
                self.quadVertex[3], 
                currentPosition, 
                self.exterior, 
                self.exteriorConstant
            )

        
        elif self.mode == StageProgramMode.Gaussian:

            if not (math.isclose(self.gaussianParams.x_sigma, 0.0) or math.isclose(self.gaussianParams.y_sigma, 0.0)):

                currentPosition = np.array([x, y], np.float32)

                if self.isGaussianRelative:
                    currentPosition = currentPosition - self.startRecordPosition

                gMeans = np.array([self.gaussianParams.x_mean, self.gaussianParams.y_mean], np.float32)

                distance = currentPosition - gMeans

                # Compute normalized gaussian distribution
                val = (
                    self.gaussianParams.amplitude
                    * np.exp(
                        -((distance[0])**2 / (2 * (self.gaussianParams.x_sigma**2)))
                        -((distance[1])**2 / (2 * (self.gaussianParams.y_sigma**2)))
                    )
                )
                
        # TODO: This should be in the setting to support High-voltage DAQ
        # Clamp between 0, 5 vol
        val = min(max(0, val), 5)

        return val


    def generateValueMapPlot(self)-> np.ndarray:
        """Generate a heat-map plot of possible values on the stage area.

        Returns:
            np.ndarray: RGB image of the plot
        """
        stageRange = [160, 160]

        isRelativeToStart = (self.mode == StageProgramMode.FourPoint and self.isFourPointRelative) \
                            or (self.mode == StageProgramMode.Gaussian and self.isGaussianRelative)
        
        # Allocate value map
        valMapShape = [stageRange[0] + 1, stageRange[1] + 1, 1]
        if isRelativeToStart:
            valMapShape = [stageRange[0]*2 + 1, stageRange[1]*2 + 1, 1]
        
        valMap = np.zeros(valMapShape)

        # Compute value map
        for j in range(valMap.shape[0]):

            y = j
            if isRelativeToStart:
                y = y - stageRange[0]
                
            for i in range(valMap.shape[1]):

                x = i
                if isRelativeToStart:
                    x = x - stageRange[1]

                valMap[j, i] = self.getValue(x, y)
        
        # Create the plot
        plt.ioff()
        fig = plt.figure(figsize=(6, 6))
        
        # Plot map
        extent = None
        if isRelativeToStart:
            extent = (-stageRange[0], stageRange[0], stageRange[1], -stageRange[1])

        im = plt.imshow(valMap, cmap= 'magma', extent= extent)
        plt.colorbar(im)
        
        # Plot landmarks
        def drawPointWithAnnotation(point: List[float], color: str, name: str) -> None:
            plt.scatter(point[0], point[1], c= color)
            plt.annotate(name, (point[0], point[1]), textcoords= 'offset points', xytext= (10,10), ha= 'center', fontsize= 12, color= 'green')
        
        if self.mode == StageProgramMode.FourPoint:
            drawPointWithAnnotation(self.quadVertex[0].point, 'r', self.quadVertex[0].name)
            drawPointWithAnnotation(self.quadVertex[1].point, 'r', self.quadVertex[1].name)
            drawPointWithAnnotation(self.quadVertex[2].point, 'r', self.quadVertex[2].name)
            drawPointWithAnnotation(self.quadVertex[3].point, 'r', self.quadVertex[3].name)
        
        elif self.mode == StageProgramMode.Gaussian:
            drawPointWithAnnotation([self.gaussianParams.x_mean, self.gaussianParams.y_mean], 'r', 'Mean')
        
        if isRelativeToStart:
            drawPointWithAnnotation([0, 0], 'r', 'Start Pos')

        # Set plot limits and labels
        if isRelativeToStart:

            # Find min, max
            btmLeft = np.zeros([2], np.float32)
            topRight = np.zeros([2], np.float32)

            if self.mode == StageProgramMode.FourPoint:
                
                for vertex in self.quadVertex:
                    btmLeft = np.where(btmLeft > vertex.point, vertex.point, btmLeft)
                    topRight = np.where(topRight < vertex.point, vertex.point, topRight)


            elif self.mode == StageProgramMode.Gaussian:

                span = 2

                btmLeft[0] = -self.gaussianParams.x_sigma * span + self.gaussianParams.x_mean
                btmLeft[1] = -self.gaussianParams.y_sigma * span + self.gaussianParams.y_mean
                topRight[0] = self.gaussianParams.x_sigma * span + self.gaussianParams.x_mean
                topRight[1] = self.gaussianParams.y_sigma * span + self.gaussianParams.y_mean

                # Also check bound with origin
                btmLeft = np.where(btmLeft > np.zeros([2]), np.zeros([2]), btmLeft)
                topRight = np.where(topRight < np.zeros([2]), np.zeros([2]), topRight)
                

            # Add padding
            btmLeft = btmLeft - 10
            topRight = topRight + 10
            
            plt.xlim(btmLeft[0], topRight[0])
            plt.ylim(btmLeft[1], topRight[1])
            
        else:
            plt.xlim(0, stageRange[0])
            plt.ylim(0, stageRange[1])

        plt.xlabel('Stage X (mm)')
        plt.ylabel('Stage Y (mm)')

        title = "Stage position to Voltage map"
        if isRelativeToStart:
            title = "Relative stage-position to Voltage map"

        plt.title(title)

        # Add grid and legend
        plt.grid(True)

        # Render the plot to a numpy array
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        imageArr = np.frombuffer(canvas.tostring_argb(), dtype='uint8').reshape(int(height), int(width), 4)
        # Remove alpha channel at the front
        imageArr = imageArr[:,:,1:4]

        # Finally close the figure
        plt.close(fig= fig)
        plt.ion()

        return imageArr


class ReversalDetector():
    
    def __init__(self):
        self.isReversing: bool = False
        self.animalLength_mm: float = 0
        self.trailLimit: int = 0
        self.velocityHistoryPercentage: float = 0
        self.reversalThresholdRadian: float = 0
        self.reversalVoltage: float = 0
        self.forwardVoltage: float = 0

    
    def detectReversal(self, trail: np.ndarray) -> bool:
        
        # Get last M (trial limit) vertices and 
        #   apply transformation to each row vertex
        croppedTrail = trail[-self.trailLimit::, :]

        # Greedy sums up until equal or exceed animal's length
        #   Get a reversed view: from bottom (most recent/head) to top (first point in the history)
        revTrail = croppedTrail[::-1]
        sumLength = 0
        
        tailIndex = 0
        
        for i in range(1, len(revTrail)):
            length = np.linalg.norm(revTrail[i-1] - revTrail[i])
            sumLength = sumLength + length
            tailIndex = i

            if sumLength >= self.animalLength_mm:
                break
        
        # Copy points from head to tail
        # We now have bodyVert: Bx2 (B:= body length), rows of point from head to tail
        bodyVert = revTrail[0:tailIndex+1:1]

        # Atleast two vertices
        if len(bodyVert) > 1:

            # 
            # Estimate velocity
            # 
            numHistVert = round(len(bodyVert) * self.velocityHistoryPercentage / 100)
            # Slice from head to numHistVert
            histVert = bodyVert[0:numHistVert]

            # Compute derivative between each pair of vertex. Assume equal delta time.
            velocities = histVert[0:-1] - histVert[1:]

            # Uniform weighted average
            velocity = np.sum(velocities, axis= 0) / len(velocities)

            # Check if the velocity is angling more than the reversal threshold with the the tailToHead body.
            #   If yes, reversal -> red color.
            #   If not, non-reversal -> green color.

            vecTailToHead = bodyVert[0] - bodyVert[-1]
            angle_radian = computeAngleBetweenTwo2DVecs(vecTailToHead, velocity)
            angle_degree = angle_radian * 180 / math.pi

            if angle_degree > self.reversalThresholdRadian or angle_degree < -self.reversalThresholdRadian:
                self.isReversing = True

            else:
                self.isReversing = False
        
        return self.isReversing
