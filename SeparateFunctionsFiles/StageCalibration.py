import math
import sys
import numpy as np
import os 
from pypylon import genicam
from skimage.filters import gaussian, threshold_otsu, threshold_li
from zaber_motion import Units
from zaber_motion.ascii import Connection
from pypylon import pylon
from tifffile import imsave
from skimage.registration import phase_cross_correlation
from skimage import io


def camera_init():
    try:
        # Create an instant camera object with the camera device found first.
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        camera.Open()
        # Print the model name of the camera.
        print("Using device ", camera.GetDeviceInfo().GetModelName())
        # Loading settings
        print("Possible place to read the file defining the camera's settings...")
        nodeFile = "FluorescenceTestWorms.pfs"
        pylon.FeaturePersistence.Load(nodeFile, camera.GetNodeMap(), True)
        return camera
    except genicam.GenericException as e:
        # Error handling.
        print("An exception occurred.")
        print(e.GetDescription())
        exit_code = 1
    sys.exit(exit_code)

def genCalibrationMatrix(shift):
    '''Calculating calibration matrix from extracted coordinates
    input: list of coordinates from images taken between stage movements
    output: calibration matrix, translating image distances to stage & correcting rotation
    '''
    xTranslation = shift[1]
    yTranslation = shift[0]
    print(shift)
    # Length translation
    xPixelSize = 500/xTranslation # units of um/px
    yPixelSize = 500/yTranslation # units of um/px
    # Rotation angle
    rotation = math.atan2(xTranslation, yTranslation) - math.pi/4 
    print(math.degrees(rotation))
    # Create calibration matrix (Rotation matrix reordered y, x)
    calibrationMatrix = np.zeros((2,2))
    calibrationMatrix[0][0] = -yPixelSize*math.sin(rotation)
    calibrationMatrix[0][1] = yPixelSize*math.cos(rotation)
    calibrationMatrix[1][0] = xPixelSize*math.cos(rotation)
    calibrationMatrix[1][1] = xPixelSize*math.sin(rotation)

    return calibrationMatrix


def getImgs(im0name = 'StartPosition.tiff', im1name = 'X_Y_AxisMotion.tiff'):
    # Hardware initialization
    connection =  Connection.open_serial_port('com3')
    connection.renumber_devices(first_address = 1)
    device_list = connection.detect_devices()
    
    device1 = device_list[0]
    device2 = device_list[1]
    axis_x = device1.get_axis(1)
    axis_y = device2.get_axis(1)

    camera = camera_init()
    
    # grabbing images before & after stage movement
    counter = 0
    while counter < 2:
        print('Grabbing image...')
        camera.StartGrabbingMax(1)      # taking image
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        img = grabResult.Array          # accessing image data  
        if counter == 0:
            imsave(im0name, img)
            axis_x.move_relative(500, Units.LENGTH_MICROMETRES, wait_until_idle=False)
            axis_y.move_relative(500, Units.LENGTH_MICROMETRES, wait_until_idle=True)
        elif counter == 1:
            imsave(im1name, img)
        counter += 1
    # return to origin
    axis_x.move_relative(-500, Units.LENGTH_MICROMETRES, wait_until_idle=False)
    axis_y.move_relative(-500, Units.LENGTH_MICROMETRES, wait_until_idle=True)
    camera.Close()
    connection.close()
    

def stageCalibration(im0name = 'StartPosition.tiff', im1name = 'X_Y_AxisMotion.tiff'):
    #getImgs()
    im0 = io.imread(os.path.join(os.getcwd(), im0name))
    im1 = io.imread(os.path.join(os.getcwd(), im1name))
    # calculate the shift using FFT correlation
    shift = phase_cross_correlation(im0, im1, upsample_factor=1, space='real', return_error=0, overlap_ratio=0.1)    
    # Generate calibration matrix
    calibrationMatrix = genCalibrationMatrix(shift)         
    print('Calibration completed.')
    return calibrationMatrix


def writeCalibration(calibrationMatrix, fname = 'calibrationMatrix.txt'):
    np.savetxt(os.path.join(os.getcwd(), fname), calibrationMatrix)


def readCalibration(fname):
    return np.loadtxt(os.path.join(os.getcwd(), fname))
    

calibmatrix = stageCalibration(im0name = 'StartPosition.tiff', im1name = 'X_Y_AxisMotion.tiff')
writeCalibration(calibmatrix, fname = 'calibrationMatrix.txt')
