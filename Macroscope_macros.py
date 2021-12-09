# Macroscope Stage Module
import math
import numpy as np
import os
import csv
import matplotlib.pylab as plt
from pypylon import pylon
from skimage.filters import threshold_otsu, threshold_li
from skimage.measure import regionprops, label
from skimage.feature import register_translation as phase_cross_correlation
from skimage.registration import phase_cross_correlation
from skimage import io
from tifffile import imsave, TiffWriter
from zaber_motion import Library, Units
from zaber_motion.ascii import connection
import Basler_control as basler


#Library.toggle_device_db_store(True)


#%% Stage Calibration
def genCalibrationMatrix(pixelsize, rotation):
    '''Calculating calibration matrix from pixelsize and known rotation.
    output: calibration matrix, translating image distances to stage & correcting rotation
    '''
    # Create calibration matrix (Rotation matrix reordered y, x)
    calibrationMatrix = np.zeros((2,2))
    calibrationMatrix[0][1] = math.cos(rotation)*pixelsize
    calibrationMatrix[0][0] = -math.sin(rotation)*pixelsize
    calibrationMatrix[1][1] = math.sin(rotation)*pixelsize
    calibrationMatrix[1][0] = math.cos(rotation)*pixelsize
    return calibrationMatrix


def take_calibration_images(stage, camera, stepsize, stepunits):
    """take two image between a defined move of the camera."""
    # take one image
    _, img1 = basler.single_take(camera)
    # move stage
    stage.move_rel((stepsize,stepsize,0), stepunits, wait_until_idle = True)
    # take another one
    _, img2 = basler.single_take(camera)
    # undo stage motion
    stage.move_rel((-stepsize,-stepsize,0), stepunits, wait_until_idle = True)
    return img1, img2


def getCalibrationMatrix(im1, im2, stage_step):
    """generate calibration from correlation between two images separated by known stage distance."""
    # calculate the shift using FFT correlation
    shift = phase_cross_correlation(im1, im2, upsample_factor=1, space='real', return_error=0, overlap_ratio=0.5)    
    # Generate calibration matrix
    xTranslation = shift[1]
    yTranslation = shift[0]
    print(shift, stage_step)
    # Length translation
    xPixelSize = stage_step/np.abs(xTranslation) # units of um/px
    yPixelSize = stage_step/np.abs(yTranslation) # units of um/px
    # Rotation angle - image angle versus stage motion 
    # (45 degrees - both axes move the same amount)
    rotation = math.atan2(xTranslation, yTranslation) - math.pi/4 
    return 0.5*(xPixelSize+yPixelSize), rotation
        

#%% Autofocus using z axis
def zFocus(stage, camera, stepsize, stepunits, nsteps):
    """take a series of images and move the camera, then calculate focus."""
    stack = []
    zpos = []
    stage.move_z(-0.5*nsteps*stepsize, stepunits, wait_until_idle = True)
    for i in np.arange(0, nsteps):
            ret, img = basler.single_take(camera)
            pos = stage.get_position()
            print(pos)
            if ret and len(pos)>2:
                stack.append(img)
                zpos.append(pos[2])
                print(stepunits)
                stage.move_z(stepsize, stepunits, wait_until_idle = True)
    stack = np.array(stack)
    zpos = np.array(zpos)
    # Looking for the focal plane using the frame of maximal variance
    average_frame_intensity = np.mean(stack, axis=(2, 1))
    normalized_stack = stack.astype(float)/average_frame_intensity[:, np.newaxis, np.newaxis] 
    stack_variance = np.var(normalized_stack, axis=(2, 1))
    focal_plane = np.argmax(stack_variance)
    # Moving to best position
    stage.move_abs((None,None,zpos[focal_plane]))
    # return focus values, images and best location
    return stack_variance, stack, zpos, focal_plane


# functions for live focusing
def calculate_focus(im, nbin = 1):
    """given an image array, calculate a proxy for sharpness."""
    return np.std(im[::nbin, ::nbin])/np.mean(im[::nbin, ::nbin])


def calculate_focus_move(past_motion, focus_history, min_step, focus_step_factor = 100):
    """given past values calculate the next focussing move."""
    error = (focus_history[-1]-focus_history[-2])/np.abs(focus_history[-2])#  current - previous. negative means it got worse
    print(error)
    if ~np.isfinite(error):
        return 0
    return error*np.sign(past_motion)*focus_step_factor*min_step#np.min([error*focus_step_factor*min_step, focus_step_factor*min_step])


# functions for tracking
#%% Functions used for centering stage
def extractWorms(img, area=0, bin_factor=4, li_init=10):
    '''
    use otsu threshold to obtain mask of pharynx & label them
    input: image of shape (N,M) 
    output: array of worm coordinates.
    '''
    img = img[::bin_factor, ::bin_factor]
    mask = img > threshold_otsu(img)
    labeled = label(mask)
    coords = []
    for region in regionprops(labeled):
        if region.area >=area:
            y, x = region.centroid
            coords.append([y*bin_factor,x*bin_factor])
    return np.array(coords)


def getDistanceToCenter(coords, imshape):
    '''
    calculates distance of worms to center, keeps x/y-distances for worm closest to center
    input: list of all worm coordinates & tuple of image shape
    output: list of distances(px) of closest worm to center
    '''
    #centerCoords = [px/2 for px in imshape]
    # calculate maximal distance from center in field of view
    #smallestDistanceToCenter = (imshape[0] - centerCoords[0])**2 + (imshape[1] - centerCoords[1])**2
    h,w = imshape
    current_distance = h**2+w**2
    yc, xc = 0,0
    for (y,x) in coords:
        distanceToCenter = (h//2 - y)**2 + (w//2-x)**2
        if distanceToCenter < current_distance:
            current_distance  = distanceToCenter    
            yc, xc = y,x
    # return offset from center for closest object
    return yc-h//2, xc-w//2


def getStageDistances(deltaCoords, calibrationMatrix):
    '''
    translates distance of worm to center from px to um
    input: distance of worm to center in px
    output: distance of worm to center in um
    '''
    stageDistances = np.dot(calibrationMatrix, deltaCoords)
    return stageDistances


