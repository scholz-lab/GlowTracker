from pypylon import pylon, genicam
import numpy as np
import time

def grabOne(camera: pylon.InstantCamera):
    grabResult = camera.GrabOne(1000)
    img = grabResult.Array
    return img


def singleGrab(camera: pylon.InstantCamera):
    return
    camera.StartGrabbingMax(1)
    grabResult: pylon.GrabResult = camera.RetrieveResult(1000, pylon.TimeoutHandling_Return)
    camera.StopGrabbing()
    return grabResult.Array

# 
#   All of these function performs the same time.
#   They have different names for being identified by the profiler
# 
def grabStreamOneByOne_1_buffer(camera: pylon.InstantCamera):
    grabResult = camera.RetrieveResult(1000, pylon.TimeoutHandling_Return)


def grabStreamOneByOne_10_buffer(camera: pylon.InstantCamera):
    grabResult = camera.RetrieveResult(1000, pylon.TimeoutHandling_Return)


def grabStreamLatestImageOnly(camera: pylon.InstantCamera):
    grabResult = camera.RetrieveResult(1000, pylon.TimeoutHandling_Return)


def grabStreamLatestImages_1_buffer(camera: pylon.InstantCamera):
    grabResult = camera.RetrieveResult(1000, pylon.TimeoutHandling_Return)


def grabStreamLatestImages_10_buffer(camera: pylon.InstantCamera):
    grabResult = camera.RetrieveResult(1000, pylon.TimeoutHandling_Return)


    
if __name__ == '__main__':

    # Init camera

    camera = pylon.InstantCamera(
        pylon.TlFactory.GetInstance().CreateFirstDevice())

    camera.Open()

    # enable all chunks
    camera.ChunkModeActive = True

    for cf in camera.ChunkSelector.Symbolics:
        camera.ChunkSelector = cf
        camera.ChunkEnable = True
    
    FPS = 60
    SPF = 1/FPS * 1e6 # in micro second unit
    camera.ExposureTime.SetValue( SPF )

    N = 100

    # 
    # Time Grab One perf
    # 
    grabOneAvgTime = 0
    for i in range(N):
        
        beginTime = time.perf_counter()

        grabOne(camera= camera)

        endTime = time.perf_counter()

        grabOneAvgTime += endTime - beginTime
    
    grabOneAvgTime /= N

    print(f'GrabOne AvgTime: {grabOneAvgTime}')

    # 
    # Time Single Grab perf
    # 

    # Setup
    camera.MaxNumBuffer = 1

    singleGrabAvgTime = 0
    for i in range(N):

        beginTime = time.perf_counter()

        singleGrab(camera= camera)

        endTime = time.perf_counter()

        singleGrabAvgTime += endTime - beginTime
    
    singleGrabAvgTime /= N
    
    print(f'Single Grab Avg Time: {singleGrabAvgTime}')

    # 
    # Time Grab Stream: One-by-One 1 frame buffer
    # 
    camera.MaxNumBuffer = 1
    camera.StartGrabbing(pylon.GrabStrategy_OneByOne)

    grabStreamOnebyOne_1_buffer_AvgTime = 0
    for i in range(N):

        beginTime = time.perf_counter()

        grabStreamOneByOne_1_buffer(camera= camera)

        endTime = time.perf_counter()

        grabStreamOnebyOne_1_buffer_AvgTime += endTime - beginTime
    
    camera.StopGrabbing()

    grabStreamOnebyOne_1_buffer_AvgTime /= N
    
    print(f'Grab Stream: One-by-One 1 buffer Avg Time: {grabStreamOnebyOne_1_buffer_AvgTime}')

    # 
    # Time Grab Stream: One-by-One 10 frame buffer
    # 
    camera.MaxNumBuffer = 10
    camera.StartGrabbing(pylon.GrabStrategy_OneByOne)

    grabStreamOnebyOne_10_buffer_AvgTime = 0
    for i in range(N):

        beginTime = time.perf_counter()

        grabStreamOneByOne_10_buffer(camera= camera)

        endTime = time.perf_counter()

        grabStreamOnebyOne_10_buffer_AvgTime += endTime - beginTime
    
    camera.StopGrabbing()
    
    grabStreamOnebyOne_10_buffer_AvgTime /= N
    
    print(f'Grab Stream: One-by-One 10 buffers Avg Time: {grabStreamOnebyOne_10_buffer_AvgTime}')


    # 
    # Time Grab Stream: Latest Image Only 10 frame buffer
    # 
    camera.MaxNumBuffer = 1
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    grabStreamLatestImageOnly_AvgTime = 0
    for i in range(N):

        beginTime = time.perf_counter()

        grabStreamLatestImageOnly(camera= camera)

        endTime = time.perf_counter()

        grabStreamLatestImageOnly_AvgTime += endTime - beginTime
    
    camera.StopGrabbing()
    
    grabStreamLatestImageOnly_AvgTime /= N
    
    print(f'Grab Stream: Latest Image Only Avg Time: {grabStreamLatestImageOnly_AvgTime}')

    # 
    # Time Grab Stream: One-by-One 1 frame buffer
    # 
    camera.MaxNumBuffer = 1
    camera.StartGrabbing(pylon.GrabStrategy_LatestImages)

    grabStreamLatestImages_1_buffer_AvgTime = 0
    for i in range(N):

        beginTime = time.perf_counter()

        grabStreamLatestImages_1_buffer(camera= camera)

        endTime = time.perf_counter()

        grabStreamLatestImages_1_buffer_AvgTime += endTime - beginTime
    
    camera.StopGrabbing()

    grabStreamLatestImages_1_buffer_AvgTime /= N
    
    print(f'Grab Stream: Latest Images 1 buffer Avg Time: {grabStreamLatestImages_1_buffer_AvgTime}')

    # 
    # Time Grab Stream: One-by-One 10 frame buffer
    # 
    camera.MaxNumBuffer = 10
    camera.StartGrabbing(pylon.GrabStrategy_LatestImages)

    grabStreamLatestImages_10_buffer_AvgTime = 0
    for i in range(N):

        beginTime = time.perf_counter()

        grabStreamLatestImages_10_buffer(camera= camera)

        endTime = time.perf_counter()

        grabStreamLatestImages_10_buffer_AvgTime += endTime - beginTime
    
    camera.StopGrabbing()

    grabStreamLatestImages_10_buffer_AvgTime /= N
    
    print(f'Grab Stream: Latest Images 10 buffers Avg Time: {grabStreamLatestImages_10_buffer_AvgTime}')


    # Close
    camera.Close()