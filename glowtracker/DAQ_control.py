from __future__ import annotations
import LabJackPython
import u3
import re
from enum import StrEnum
from collections import OrderedDict
from copy import deepcopy

class TriggerUpdateMode(StrEnum):
    Frame = 'Frame'
    Time = 'Time'


class DAQControl():

    @classmethod
    def createAndConnectDaq(cls) -> DAQControl | None:
        """Create a DAQControl class object. Return the object if the connection to the actual DAQ
        is successful. Otherwise, return None.

        Returns:
            DAQControl (DAQControl|None): DAQControl class if successful, otherwise None.
        """
        try:
            # Connect to DAQ
            daq = u3.U3(debug= False)

        except Exception as e:
            print(e)
            return None

        else:
            # Instantiate DAQConatrol object
            daqControl = DAQControl()
            daqControl.daq = daq
            
            print(f"Using {daqControl.daq.deviceName}, serial: {daqControl.daq.serialNumber}")
            
            # Set to factory default
            daqControl.daq.setDefaults()
            # Calibrate
            daqControl.daq.getCalibrationData()

            return daqControl


    def __init__(self):
        self.daq: u3.U3 | None = None
        self.ledsSequnceDict : OrderedDict = OrderedDict()
        self.ledsSequnceDictRunning : OrderedDict = OrderedDict()
        self.isEnable: bool = False
        self.mode: TriggerUpdateMode = TriggerUpdateMode.Frame

    
    def close(self):
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
            processedDict = eval(preprocessdText, globals= {
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
                self.mode = TriggerUpdateMode.Frame

            elif mode == 'time':
                self.mode = TriggerUpdateMode.Time
                
            else:
                raise ValueError(f"Failed to parse LED script text: Invalid 'mode' argument. Options are ['frame', 'time']")

        except Exception as e:
            raise ValueError(f"Failed to parse LED script text: {e}")
    

    def triggerCommand(self, frameNum: int = 0, frameTime: float = 0) -> None:
        
        if len(self.ledsSequnceDictRunning) == 0 or not self.isEnable:
            return

        if self.mode == TriggerUpdateMode.Frame:
        
            # Get the exact frame command
            frameCommand = self.ledsSequnceDictRunning.pop(frameNum, default= None)

            if frameCommand is not None:
                print(f"Frame {frameNum}:")
                self._executeCommand(frameCommand)
        
        elif self.mode == TriggerUpdateMode.Time:
            
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
