from __future__ import annotations
import LabJackPython
import u3
import re

class DAQControl():

    HIGH = 1
    LOW = 0

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
        self.ledsSequnceDict : dict | None = None
        self.isEnable: bool = False

    
    def close(self):
        # Check if Windows then call LabJackPython.Close(), else call self.daq.close()
        self.daq.close()
        self.daq = None

    
    def reset(self):
        # Set to factory default
        self.daq.setDefaults()

        
    def parseTextScript(self, text: str) -> None:

        try:
            # Remove empty lines and surrounding whitespace
            lines = [line.strip() for line in text.splitlines() if line.strip()]

            # Remove trailing commas from each line
            lines = [re.sub(r',$', '', line) for line in lines]

            # Wrap into a dict literal
            preprocessdText = "{\n" + ",\n".join(lines) + "\n}"
            
            # Parse the text to be a dict object. Highlight keywords "on", "off"
            self.ledsSequnceDict = eval(preprocessdText, globals= {"on": "on", "off": "off"})
        
        except Exception as e:
            raise ValueError(f"Failed to parse LED script text: {e}") from None
    

    def triggerCommand(self, frameNum: int) -> None:
        
        if self.ledsSequnceDict is None or not self.isEnable:
            return
        
        frameCommand = self.ledsSequnceDict.get(frameNum)

        if frameCommand is None:
            return

        command: list = frameCommand[0]

        if command == 'on':
            if len(frameCommand) != 2:
                print("\"on\" command requires a voltage argument.")
                return
            
            vol = frameCommand[1]

            # Clip to 0, 4.95
            vol = max( min( vol, 4.95 ), 0 )

            print(f"Turn on the light! {vol}")

            # Send command to DAQ at DAC0
            dac0Val = self.daq.voltageToDACBits(volts= vol, dacNumber= 0, is16Bits= False)
            dac0Command = u3.DAC0_8(dac0Val)
            self.daq.getFeedback(dac0Command)
            
        
        elif command == 'off':
            
            print(f"Turn off the light!")

            # Send command to DAQ at DAC0
            dac0Val = self.daq.voltageToDACBits(volts= 0, dacNumber= 0, is16Bits= False)
            dac0Command = u3.DAC0_8(dac0Val)
            self.daq.getFeedback(dac0Command)
        
