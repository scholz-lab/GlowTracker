import math
import sys
import numpy as np
import csv
from pypylon import genicam
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import regionprops, label
from zaber_motion import Library, Units
from zaber_motion.ascii import Connection
from pypylon import pylon
from tifffile import TiffWriter
import StageCalibration as sc

Library.toggle_device_db_store(True)


def camera_init():
    try:
        # Create an instant camera object with the camera device found first.
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        camera.Open()
        # Print the model name of the camera.
        print("Using device ", camera.GetDeviceInfo().GetModelName())
        # Loading settings
        print("Possible place to read the file defining the camera's settings...")
        nodeFile = "FluorescenceTestWormsSmallROI.pfs"
        pylon.FeaturePersistence.Load(nodeFile, camera.GetNodeMap(), True)
        return camera
    except genicam.GenericException as e:
        # Error handling.
        print("An exception occurred.")
        print(e.GetDescription())
        exit_code = 1
    sys.exit(exit_code)


def extractWorms(img, area = 0, binfactor = 4, li_init = 10):
    '''
    use otsu threshold to obtain mask of pharynx & label them
    input: image of shape (N,M) 
    output: array of worm coordinates.
    '''
    #im = gaussian(im, 3)
    img = img[::binfactor, ::binfactor]
    mask = img>threshold_otsu(img)
    #mask = im>threshold_li(im, initial_guess = li_init)
    labeled = label(mask)
    coords = []
    for region in regionprops(labeled):
        if area <= region.area:
            y, x = region.centroid
            coords.append([y*binfactor,x*binfactor])
    return np.array(coords)


def getDistanceToCenter(coords, imshape):
    '''
    calculates distance of worms to center, keeps x/y-distances for worm closest to center
    input: list of all worm coordinates & tuple of image shape
    output: list of distances(px) of closest worm to center
    '''
    centerCoords = [px/2 for px in imshape]
    smallestDistanceToCenter = math.sqrt((imshape[0] - centerCoords[0])**2 + (imshape[1] - centerCoords[1])**2)
    for worm in coords:
        distanceToCenter = math.sqrt((centerCoords[0] - worm[0])**2 + (centerCoords[1] - worm[1])**2)
        if distanceToCenter < smallestDistanceToCenter:
            smallestDistanceToCenter = distanceToCenter    
            deltaY = worm[0] - centerCoords[0] # y coordinates are inverted relative to axis
            deltaX = centerCoords[1] - worm[1]
            deltaCoords = [deltaY, deltaX]
    return deltaCoords


def getStageDistances(deltaCoords, calibrationMatrix):
    '''
    translates distance of worm to center from px to um
    input: distance of worm to center in px
    output: distance of worm to center in um
    '''
    stageDistances = np.dot(calibrationMatrix, deltaCoords)
    return stageDistances
   

def main():
    connection =  Connection.open_serial_port('com3')
    connection.renumber_devices(first_address = 1)
    device_list = connection.detect_devices()
    
    device1 = device_list[0]
    device2 = device_list[1]
    axis_x = device1.get_axis(1)
    axis_y = device2.get_axis(1)
    
    # get calibration matrix from text file
    calibrationMatrix = sc.readCalibration(fname = 'calibrationMatrix.txt')
                
    my_tiff = TiffWriter('TestTracking', bigtiff=True, append=True)

    camera = camera_init()
    count = 0
    with open('TrackingLog.csv', 'w') as logfile:
        writer = csv.writer(logfile, lineterminator='\n')
        writer.writerow(['Iteration #', 'x Coord', 'y Coord'])
        while count < 100:
            print('Grabbing image...')
            camera.StartGrabbingMax(1)      # taking image
            grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            img = grabResult.Array          # accessing image data
            my_tiff.save(img)               # appending image to tiff file
            imshape = img.shape
            coords = extractWorms(img)      # extracting worm coordinates from image
            deltaCoords = getDistanceToCenter(coords, imshape)      # calculating distance of worm coordinates from center of image
            stageDistances = getStageDistances(deltaCoords, calibrationMatrix)  # get distances for stage axes
            # write a string into the open file
            writer.writerow([count, stageDistances[1], stageDistances[0]])
            print('Centering worm...')
            axis_x.move_relative(stageDistances[1], Units.LENGTH_MICROMETRES, wait_until_idle=False)
            axis_y.move_relative(stageDistances[0], Units.LENGTH_MICROMETRES, wait_until_idle=False)
            count += 1
        camera.Close()
        my_tiff.close()
        connection.close()
        print('Tracking finished.')
    
    
if __name__ == '__main__':
    main()
