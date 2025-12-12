import u3
from pypylon import pylon
import numpy as np
import time
import matplotlib.pyplot as plt
from threading import Thread

HIGH = 1
LOW = 0

def setupLabJackU3() -> u3.U3:
    # Connect to device
    device = u3.U3()
    device.debug = False
    # Set to factory default
    device.setDefaults()
    # Calibrate
    device.getCalibrationData()
    # Set FIO0, FIO2 to digital 
    device.configDigital(u3.FIO0, u3.FIO2)

    return device


def setupCamera() -> pylon.InstantCamera:
    # Init camera
    camera = pylon.InstantCamera( pylon.TlFactory.GetInstance().CreateFirstDevice() )

    camera.Open()

    # Specify FPS and Exposure 
    #   FPS
    FPS = 30
    camera.AcquisitionFrameRateEnable.Value = True
    camera.AcquisitionFrameRate.Value = float(FPS)

    #   Exposure
    exposureTime = 30000 # in micro second unit
    camera.ExposureTime.Value = exposureTime

    return camera


def blinkLED(daq: u3.U3, duration: float):

    daq.configDigital(u3.FIO0)
    daq.setDOState(ioNum= u3.FIO0, state= HIGH)
    time.sleep(duration)
    daq.setDOState(ioNum= u3.FIO0, state= LOW)


def sineWaveLed(daq: u3.U3, mean: float, amplitude: float, freq: float, duration: float, resolution: float):

    # From testing
    resolution = max(0.0003, resolution)

    # Generate signal
    data = np.arange(int(duration / resolution)) % (freq / resolution)
    data = (data / np.max(data)) * 2 * np.pi
    data =  np.sin(data) * amplitude + mean

    # Cap signal within safe 0-5 V
    data = np.clip(data, 0, 5)

    plt.figure()
    plt.plot(data)
    plt.show()

    cumOpTime = 0
    elapseTimeBegin = time.perf_counter()
    
    for value in data:

        startTime = time.perf_counter()
        
        dac0Val = daq.voltageToDACBits(volts= value, dacNumber= 0, is16Bits= False)
        dac0Command = u3.DAC0_8(dac0Val)
        daq.getFeedback(dac0Command)
        
        # Wait
        endTime = time.perf_counter()
        opTime = endTime - startTime
        cumOpTime = cumOpTime + opTime
        waitTime = resolution - opTime
        if waitTime > 0:
            time.sleep(waitTime)

    cumOpTime = cumOpTime / (len(data))
    print(f"Avg OP time {cumOpTime}")        

    elapseTimeEnd = time.perf_counter()
    print(f"Elapse Time {elapseTimeEnd - elapseTimeBegin}")
    print(f"Avg update time {(elapseTimeEnd - elapseTimeBegin)/len(data)}")

    # Set back to 0
    dac0Val = daq.voltageToDACBits(volts= 0, dacNumber= 0, is16Bits= False)
    dac0Command = u3.DAC0_8(dac0Val)
    daq.getFeedback(dac0Command)


class TriggeredImage(pylon.ImageEventHandler):

    def __init__(self):
        super().__init__()

        self.grab_times = []

        self.images = []
        self.retrieveTimestamps = []
        self.timestamps = []
        
        
    def OnImageGrabbed(self, camera, grabResult: pylon.GrabResult):

        if grabResult.GrabSucceeded():
            print("Got an image!")

            self.grab_times.append(grabResult.TimeStamp)
            
            self.images.append(grabResult.Array)

            self.retrieveTimestamps.append(time.perf_counter())
            
            conversion_factor = 1e6  # for conversion in ms
            timestamp = round(grabResult.TimeStamp/conversion_factor, 1)
            self.timestamps.append(timestamp)

            grabResult.Release()

        else:
            print("Grab didn't succeed")
    

    def OnImagesSkipped(self, camera, countOfSkippedImages):
        print(countOfSkippedImages, " images have been skipped.")


class ThreadWithReturn(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, verbose=None):
        # Initializing the Thread class
        super().__init__(group, target, name, args, kwargs)
        self._return = None

    # Overriding the Thread.run function
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self):
        super().join()
        return self._return


def triggerSingleCameraAcquisition(daq: u3.U3, camera: pylon.InstantCamera):

    # Prep control signal
    daq.configDigital(u3.FIO0)
    daq.setDOState(ioNum= u3.FIO0, state= LOW)

    # 
    # Set camera to take trigger signal
    # 
    camera.TriggerMode.Value = "On"

    #   Set Line1 to be input
    camera.LineSelector.Value = "Line1"
    camera.LineMode.Value = "Input"
    
    #   Set trigger source to Line1
    camera.TriggerSource.Value = "Line1"
    camera.TriggerSelector.Value = "FrameStart"
    camera.TriggerActivation.Value = "RisingEdge"

    # Start camera aquisition one frame in a thread
    def grabOneFrame(camera: pylon.InstantCamera) -> pylon.GrabResult:

        print("Start grabbing")
        grabResult = camera.GrabOne(pylon.waitForever)
        print("End grabbing")

        return grabResult
    
    cameraThread = ThreadWithReturn(target=grabOneFrame, args=(camera,))
    cameraThread.start()

    print("Wait for trigger ready")
    camera.WaitForFrameTriggerReady(1)

    print("Begin send signal")
    # Send trigger signal
    daq.setDOState(ioNum= u3.FIO0, state= HIGH)
    print("End send signal")
    
    # Stop the thread and get the result back
    grabResult: pylon.GrabResult = cameraThread.join()
    print("Thread is joined")

    if grabResult and grabResult.GrabSucceeded():

        print("Successful!")

        img = grabResult.Array
        retrieveTimestamp = time.perf_counter()
        conversion_factor = 1e6  # for conversion in ms
        timestamp = round(grabResult.TimeStamp/conversion_factor, 1)
        grabResult.Release()

        plt.figure()
        plt.imshow(img)
        plt.show()

    else:
        print("Unsuccessful")

    # Reset signal back to low
    daq.setDOState(ioNum= u3.FIO0, state= LOW)


if __name__ == '__main__':

    daq = setupLabJackU3()
    camera = setupCamera()

    # Try blinking LED 3 times
    # blinkLED(daq, 0.5)
    # time.sleep(0.5)
    # blinkLED(daq, 0.5)
    # time.sleep(0.5)
    # blinkLED(daq, 0.5)
    
    # Sine wave LED
    # sineWaveLed(daq= daq, mean= 1.5, amplitude= 1.5,  freq= 1, duration= 5, resolution= 0.00001)

    # Trigger acquisition
    triggerSingleCameraAcquisition(daq, camera)
    



    


    