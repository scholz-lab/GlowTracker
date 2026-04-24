from pypylon import pylon, genicam
import numpy as np
import time
import matplotlib.pyplot as plt

    
if __name__ == '__main__':

    # Init camera
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
    camera.Open()

    # enable all chunks
    camera.ChunkModeActive.Value = True

    for cf in camera.ChunkSelector.Symbolics:
        camera.ChunkSelector.Value = cf
        camera.ChunkEnable.Value = True
    
    # Specify FPS and Exposure 
    #   FPS
    FPS = 30
    camera.AcquisitionFrameRateEnable.Value = True
    camera.AcquisitionFrameRate.Value = float(FPS)

    #   Exposure
    exposureTime = 30000 # in micro second unit
    camera.ExposureTime.Value = exposureTime

    print("Preped camera")

    lines = ["Line1", "Line2", "Line3", "Line4"]
    for line in lines:
        camera.LineSelector.Value = line
        # camera.LineMode.Value = "Input"
        lineMode = camera.LineMode.GetValue();
        lineStatus = camera.LineStatus.GetValue()
        print(line, lineMode, lineStatus)

    camera.StartGrabbingMax(100)

    io_res = []
    while camera.IsGrabbing():
        with camera.RetrieveResult(1000) as res:
            time_stamp = res.TimeStamp
            io_res.append((time_stamp, res.ChunkLineStatusAll.Value))


    camera.StopGrabbing()


    # list of timestamp + io status
    print(io_res[:10])

    # simple logic analyzer :-)

    # convert to numpy array
    io_array = np.array(io_res)
    # extract first column timestamps
    x_vals = io_array[:,0]
    #  start with first timestamp as '0'
    x_vals -= x_vals[0]

    # extract second column io values
    y_vals = io_array[:,1]
    # for each bit plot the graph
    for bit in range(8):
        
        logic_level = ((y_vals & (1<<bit)) != 0)*0.8 +bit
        # plot in seconds
        plt.plot(x_vals / 1e9, logic_level, label = bit)
        
    plt.xlabel("time [s]")
    plt.ylabel("IO_LINE [#]")
    plt.legend()
    plt.show()
    x = 2

