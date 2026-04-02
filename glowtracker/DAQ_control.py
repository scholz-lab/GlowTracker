from __future__ import annotations
import LabJackPython
import u3
import re
from enum import StrEnum
from collections import OrderedDict
from copy import deepcopy
from typing import List
from Microscope_macros import Vertex2D, Exterior
import numpy as np
import math
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from dataclasses import dataclass


class LEDsMode(StrEnum):
    Off = 'Off'
    Sequencer = 'Sequencer'
    StageProgram = 'StageProgram'


class SequencerMode(StrEnum):
    Frame = 'Frame'
    Time = 'Time'


class StageProgramMode(StrEnum):
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
        self.ledsSequnceDict : OrderedDict = OrderedDict()
        self.ledsSequnceDictRunning : OrderedDict = OrderedDict()
        # self.isEnable: bool = False
        self.ledsMode: LEDsMode = LEDsMode.Off
        self.sequencerMode: SequencerMode = SequencerMode.Frame
        self.daqStageProgram: DAQStageProgram = DAQStageProgram()
    

    def isConnected(self) -> bool:
        return self.daq is not None


    def close(self):
        if self.isConnected():
            # Check if Windows then call LabJackPython.Close(), else call self.daq.close()
            self.daq.close()
            self.daq = None

    
    def start(self):
        """Reset internal command dict to original to prepare for running.
        """
        self.ledsSequnceDictRunning = deepcopy(self.ledsSequnceDict)
    
    
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
        self.ledsSequnceDictRunning.clear()

        
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
                self.ledsSequnceDict.clear()
                return

            # Get running mode
            mode = processedDict.pop('mode')[0]
            
            # Sort and convert to OrderedDict
            self.ledsSequnceDict = OrderedDict( {key:val for key, val in sorted(processedDict.items(), key= lambda x: x[0])} )

            if mode == 'frame':
                self.sequencerMode = SequencerMode.Frame

            elif mode == 'time':
                self.sequencerMode = SequencerMode.Time
                
            else:
                raise ValueError(f"Failed to parse LED script text: Invalid 'mode' argument. Options are ['frame', 'time']")

        except Exception as e:
            raise ValueError(f"Failed to parse LED script text: {e}")
    

    def update(self, frameNum: int = 0, frameTime: float = 0, stagePosition: List[float] = []) -> None:
        if self.ledsMode == LEDsMode.Off:
            return
        
        elif self.ledsMode == LEDsMode.Sequencer:
            self.updateSequencer(frameNum= frameNum, frameTime= frameTime)
        
        elif self.ledsMode == LEDsMode.StageProgram:
            self.updateStageProgram(stagePosition)


    def updateSequencer(self, frameNum: int = 0, frameTime: float = 0) -> None:

        # Seperate this into two cases, one for each LEDsMode
        #   Also need stage position input
        if len(self.ledsSequnceDictRunning) == 0 or not self.ledsMode == LEDsMode.Sequencer:
            return

        if self.sequencerMode == SequencerMode.Frame:
        
            # Get the exact frame command
            frameCommand = self.ledsSequnceDictRunning.pop(frameNum, default= None)

            if frameCommand is not None:
                print(f"Frame {frameNum}:")
                self._executeCommand(frameCommand)
        
        elif self.sequencerMode == SequencerMode.Time:
            
            # Get the first (lowest frame time) command in queue
            commandFrameTime = next(iter(self.ledsSequnceDictRunning))

            if frameTime >= commandFrameTime:

                # Pop the command
                # Get all the commands with time that are lower than the frame time
                commands = []
                while commandFrameTime <= frameTime and len(self.ledsSequnceDictRunning) > 0:

                    # Pop first item (lowest frame time)
                    commandFrameTime, frameCommand = self.ledsSequnceDictRunning.popitem(last= False)
                    commands.append([commandFrameTime, frameCommand])

                    # If the command queue is now empty then stop
                    if len(self.ledsSequnceDictRunning) == 0:
                        break
                    
                    # Get the next one
                    commandFrameTime = next(iter(self.ledsSequnceDictRunning))

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
        
        self._executeCommand(frameCommand= ['on', vol])
    

    def _executeCommand(self, frameCommand: list) -> None:
        
        command: list = frameCommand[0]

        if command == 'on':
            if len(frameCommand) != 2:
                print("\"on\" command requires a voltage argument.")
                return
            
            vol = frameCommand[1]

            # Clip to 0, 4.95
            vol = max( min( vol, 4.95 ), 0 )

            print(f"Light on {vol} vol")

            # Send command to DAQ at DAC0
            dac0Val = self.daq.voltageToDACBits(volts= vol, dacNumber= 0, is16Bits= False)
            dac0Command = u3.DAC0_8(dac0Val)
            self.daq.getFeedback(dac0Command)
            
        
        elif command == 'off':
            
            print(f"Light off")

            # Send command to DAQ at DAC0
            dac0Val = self.daq.voltageToDACBits(volts= 0, dacNumber= 0, is16Bits= False)
            dac0Command = u3.DAC0_8(dac0Val)
            self.daq.getFeedback(dac0Command)


@dataclass
class GaussianParams():
    amplitude: float = 0
    x_mean: float = 0
    x_sigma: float = 0
    y_mean: float = 0
    y_sigma: float = 0


class DAQStageProgram():

    def __init__(self):
        self.mode: StageProgramMode = StageProgramMode.FourPoint
        self.quadVertex: List[Vertex2D] = []
        self.exterior = Exterior.Zero
        self.exteriorConstant: float = 0
        self.gaussianParams = GaussianParams()
    
    
    def update(self, mode: StageProgramMode = None, quadVertex: List[Vertex2D] = None, exterior: Exterior = None, exteriorConstant: float = None, gaussianParams: GaussianParams = None) -> None:
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
        
        if gaussianParams:
            self.gaussianParams = gaussianParams
    
    
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

            val = Vertex2D.bilerp(self.quadVertex[0], self.quadVertex[1], self.quadVertex[2], self.quadVertex[3], np.array([x, y], np.float32), self.exterior, self.exteriorConstant)

        
        elif self.mode == StageProgramMode.Gaussian:

            if not (math.isclose(self.gaussianParams.x_sigma, 0.0) or math.isclose(self.gaussianParams.y_sigma, 0.0)):

                # Compute normalized gaussian distribution
                val = (
                    self.gaussianParams.amplitude
                    * np.exp(
                        -((x - self.gaussianParams.x_mean)**2 / (2 * (self.gaussianParams.x_sigma**2)))
                        -((y - self.gaussianParams.y_mean)**2 / (2 * (self.gaussianParams.y_sigma**2)))
                    )
                )
                
        # Clamp between 0, 5 vol
        val = min(max(0, val), 5)

        return val


    def generateValueMapPlot(self)-> np.ndarray:
        """Generate a heat-map plot of possible values on the stage area.

        Returns:
            np.ndarray: RGB image of the plot
        """
        
        # Compute value map
        valMap = np.zeros([160 + 1, 160 + 1, 1], np.float32)
        for y in range(valMap.shape[0]):
            for x in range(valMap.shape[1]):
                valMap[y, x] = self.getValue(x, y)
            
        
        # Create the plot
        plt.ioff()
        fig = plt.figure(figsize=(6, 6))
        
        # Plot map
        im = plt.imshow(valMap, cmap= 'magma')
        plt.colorbar(im)
        
        # Plot 4 points
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

        # Set plot limits and labels
        plt.xlim(0, 160)
        plt.ylim(0, 160)
        plt.xlabel('Stage X (mm)')
        plt.ylabel('Stage Y (mm)')
        plt.title("Stage position to Voltage map")

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
