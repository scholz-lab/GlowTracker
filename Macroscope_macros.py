# Macroscope Stage Module
import math
import numpy as np
import os
import csv
import scipy.ndimage as ndi
import matplotlib as mpl
import matplotlib.pylab as plt
from pypylon import pylon
from skimage.filters import threshold_otsu, threshold_li, threshold_yen
from skimage.measure import regionprops, label
from skimage.transform import downscale_local_mean
#from skimage.feature import register_translation as phase_cross_correlation
from skimage.registration import phase_cross_correlation
from skimage import io
from tifffile import imsave, TiffWriter
from zaber_motion import Library, Units
from zaber_motion.ascii import connection
import Basler_control as basler
from typing import Tuple
import itk
import cv2
from skimage.exposure import match_histograms

def switchXY2x2Mat(matXY: np.ndarray) -> np.ndarray:
    """Modified a 2x2 matrix such that the the multiplication operation 
    is suitable for a 2D vector of order y,x

    Args:
        matXY (np.ndarray): A 2x2 matrix

    Returns:
        np.ndarray: A modified 2x2 matrix with flipped order x,y
    """    
    if matXY.shape != (2, 2):
        return
    
    A = matXY[0][0]
    B = matXY[0][1]
    C = matXY[1][0]
    D = matXY[1][1]

    matYX = np.array([
        [D, C],
        [B, A]
    ], matXY.dtype)

    return matYX


def genImageToStageMatrix(scale: float, rotation: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute trasnformation matrix from image space to stage space using the given rotation angle and scale. Assume only rotation and uniform scaling.

    Args:
        scale (float): width or height (meter) per pixel
        rotation (float): rotation angle between the image space and stage space

    Returns:
        imageToStageMat (np.ndarray): transformation matrix from image space to stage space
        imageToStageRotOnlyMat (np.ndarray): transformation matrix from image space to stage space without uniform scaling
    """    

    # Stage to Image. Standard 2D rotation matrix
    cosval, sinval = math.cos(rotation), math.sin(rotation)
    stageToImageMat = np.array([
        [cosval, -sinval],
        [sinval, cosval]
    ], np.float32)

    # Stage to Image
    imageToStageMat = np.linalg.inv( stageToImageMat )

    # switch x,y to y,x
    imageToStageMat = switchXY2x2Mat(imageToStageMat)

    return imageToStageMat * scale, imageToStageMat


def takeCalibrationImages(stage, camera, stepsize, stepunits):
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


def computeStageScaleAndRotation(im1: np.ndarray, im2: np.ndarray, stage_step: float) -> Tuple[float, float]:
    """Compute stage rotation and scale from the correlation between two images moved by a known stage distance.

    Args:
        im1 (np.ndarray): first image
        im2 (np.ndarray): second image, moved by stage step size
        stage_step (float): stage step size

    Returns:
        scale (float): ratio between distance in image space and in stage space
        theta (float): rotation angle from image space to stage space
    """    
    #   Calculate the shift using FFT correlation
    shift = phase_cross_correlation(im1, im2, upsample_factor=1, space='real', return_error=0, overlap_ratio=0.5)    
    
    # Vector of change in the stage space. Measured by phase cross correlation
    vec_stage_space = -np.array([shift[1], shift[0]], np.float32)
    vec_stage_space_len = np.linalg.norm(vec_stage_space)
    
    # Vector of change expected from the setting
    vec_image_space = np.array([-stage_step, -stage_step], np.float32)
    vec_image_space_len = np.linalg.norm(vec_image_space)

    # Assume both vectors have the same origin, compute rotation and scaling difference
    theta = np.arccos( np.dot(vec_stage_space, vec_image_space) / (vec_stage_space_len * vec_image_space_len) )
    scale = vec_image_space_len / vec_stage_space_len
    
    return scale, theta
        

#%% Autofocus using z axis
def zFocus(stage, camera, stepsize, stepunits, nsteps):
    """take a series of images and move the camera, then calculate focus."""
    stack = []
    zpos = []
    stage.move_z(-0.5*nsteps*stepsize, stepunits, wait_until_idle = True)
    for i in np.arange(0, nsteps):
            isSuccess, img = basler.single_take(camera)
            pos = stage.get_position()
            print(pos)
            if isSuccess and len(pos) > 2:
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
# def extractWorms(img, area=0, bin_factor=4, li_init=10, display = True):
#     '''
#     use otsu threshold to obtain mask of pharynx & label them
#     input: image of shape (N,M) 
#     output: array of worm coordinates.
#     '''
#     img = img[::bin_factor, ::bin_factor]
#     print('image shape', img.shape)
#     mask = img > threshold_otsu(img)


#     labeled = label(mask)
#     if display:
#         plt.figure()
#         plt.subplot(211)
#         plt.imshow(img)
#         plt.subplot(212)
#         plt.imshow(labeled)
        
#     coords = []
#     for region in regionprops(labeled):
#         if region.area >=area:
#             y, x = region.centroid
#             coords.append([y*bin_factor,x*bin_factor])
#     return np.array(coords)


# def getDistanceToCenter(coords, imshape):
#     '''
#     calculates distance of worms to center, keeps x/y-distances for worm closest to center
#     input: list of all worm coordinates & tuple of image shape
#     output: list of distances(px) of closest worm to center
#     '''
#     #centerCoords = [px/2 for px in imshape]
#     # calculate maximal distance from center in field of view
#     #smallestDistanceToCenter = (imshape[0] - centerCoords[0])**2 + (imshape[1] - centerCoords[1])**2
#     h,w = imshape
#     current_distance = h**2+w**2
#     yc, xc = 0,0
#     for (y,x) in coords:
#         distanceToCenter = (h//2 - y)**2 + (w//2-x)**2
#         if distanceToCenter < current_distance:
#             current_distance  = distanceToCenter    
#             yc, xc = y,x
#     # return offset from center for closest object
#     return yc-h//2, xc-w//2


def getStageDistances(deltaCoords, imageToStageMat):
    '''
    translates distance of worm to center from px to um
    input: distance of worm to center in px
    output: distance of worm to center in um
    '''
    stageDistances = np.matmul(imageToStageMat, deltaCoords)
    return stageDistances

# functions for tracking
#%% Functions used for centering stage
def extractWormsDiff(img1, img2, capture_radius = -1,  bin_factor=4, area = 0, threshold = 10, dark_bg = True, display = False):
    '''
    use image difference to detect motion of object.
    input: image of shape (N,M) 
    minimal_difference: fraction of pixel that need to have changed to consider a difference
    output: vector of maximal/minimal change indicating where stage should compensate.
    '''
    # clip image
    h,w = img1.shape
    #to region of interest
    ymin, ymax, xmin, xmax = 0,h,0,w
    if capture_radius > 0 :
        ymin, ymax, xmin, xmax = np.max([0,h//2-capture_radius]), np.min([h,h//2+capture_radius]), np.max([0,w//2-capture_radius]), np.min([w,w//2+capture_radius])
    img1_sm = img1[ymin:ymax, xmin:xmax]
    img2_sm = img2[ymin:ymax, xmin:xmax]
    
    # reduce image size
    img1_sm = downscale_local_mean(img1_sm, (bin_factor, bin_factor), cval=0, clip=True)
    img2_sm = downscale_local_mean(img2_sm, (bin_factor, bin_factor), cval=0, clip=True)
    
    # threshold
    # threshold = threshold_yen(img1_sm)
    # if dark_bg:
    #     img1_sm = img1_sm > threshold
    #     img2_sm = img2_sm > threshold
    # else:
    #     img1_sm = img1_sm < threshold
    #     img2_sm = img2_sm < threshold
   
     # generate image difference - use floats!
    diff = img1_sm.astype(float) - img2_sm.astype(float)
    
    h,w = diff.shape
    # reduced image size
    # print('image shape after binning', diff.shape)
    # return early if there was no substantial change in the image
    if np.sum(np.abs(diff)>threshold) < area/bin_factor:
        return 0,0

    if dark_bg:
        yc, xc = np.unravel_index(diff.argmin(), diff.shape)
    else:
        yc, xc = np.unravel_index(diff.argmax(), diff.shape)
    # show intermediate steps for debugging
    if display:
        plt.subplot(221)
        plt.imshow(img1_sm)
        plt.title('Previous - img1')
        plt.subplot(222)
        plt.title('Current - img2')
        plt.imshow(img2_sm)
        plt.subplot(223)
        plt.title('Difference')
        plt.imshow(diff)
        plt.plot(xc, yc, 'ro')
        plt.colorbar()
        plt.subplot(224)
        plt.title('Original')
        plt.imshow(img2)
        plt.plot( (xc*bin_factor + xmin) ,(yc*bin_factor+ymin), 'ro')
        rect = mpl.patches.Rectangle((xc+xmin, yc+ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor='w', facecolor='none')

        # Add the patch to the Axes
        plt.gca().add_patch(rect)
        plt.colorbar()


    print(yc, xc)
    return (yc-h//2)*bin_factor, (xc - w//2)*bin_factor


def extractWorms(img1, capture_radius = -1,  bin_factor=4, dark_bg = True, display = False):
    '''
    use image to detect motion of object.
    input: image of shape (N,M) 
    minimal_difference: fraction of pixel that need to have changed to consider a difference
    output: vector of maximal/minimal change indicating where stage should compensate.
    '''
    # capture radius
    # clip image
    h,w = img1.shape
    #to region of interest
    ymin, ymax, xmin, xmax = 0,h,0,w
    
    if capture_radius > 0 :
        ymin, ymax, xmin, xmax = np.max([0,h//2-capture_radius]), np.min([h,h//2+capture_radius]), np.max([0,w//2-capture_radius]), np.min([w,w//2+capture_radius])
    print(xmin, xmax, ymin, ymax)
    img1_sm = img1[ymin:ymax, xmin:xmax]
    # reduce image size
    img1_sm = downscale_local_mean(img1_sm, (bin_factor, bin_factor), cval=0, clip=True)
    h,w = img1_sm.shape
    # reduced image size
    # print('image shape after binning',h,w)
    # get cms
    h,w = img1_sm.shape
    
    # simply use max or min location
    if dark_bg:
        yc, xc = np.unravel_index(img1_sm.argmax(), img1_sm.shape)
    else:
        yc, xc = np.unravel_index(img1_sm.argmin(), img1_sm.shape)
    # show intermediate steps for debugging
    if display:
        plt.subplot(211)
        plt.imshow(img1)
        plt.title('img1 original')
        plt.plot(xc*bin_factor+xmin, yc*bin_factor+ymin, 'ro')
        plt.plot( (xc*bin_factor + xmin) ,(yc*bin_factor+ymin), 'ro')
        rect = mpl.patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        plt.gca().add_patch(rect)
        
        plt.colorbar()
        plt.subplot(212)
        plt.imshow(img1_sm)
        plt.title('img1_reduced')
        plt.plot(xc, yc, 'ro')
        plt.colorbar()

    print(yc, xc)
    return (yc-h//2)*bin_factor, (xc - w//2)*bin_factor


def extractWormsCMS(img1, capture_radius = -1,  bin_factor=4, dark_bg = True, display = False):
    '''
    use image to detect motion of object.
    input: image of shape (N,M) 
    minimal_difference: fraction of pixel that need to have changed to consider a difference
    output: vector of maximal/minimal change indicating where stage should compensate.
    '''
    # capture radius
    # clip image
    h,w = img1.shape
    #to region of interest
    ymin, ymax, xmin, xmax = 0,h,0,w
    
    if capture_radius > 0 :
        ymin, ymax, xmin, xmax = np.max([0,h//2-capture_radius]), np.min([h,h//2+capture_radius]), np.max([0,w//2-capture_radius]), np.min([w,w//2+capture_radius])
    # print(xmin, xmax, ymin, ymax)
    img1_sm = img1[ymin:ymax, xmin:xmax]
        
    # reduce image size
    img1_sm = downscale_local_mean(img1_sm, (bin_factor, bin_factor), cval=0, clip=True)

    h,w = img1_sm.shape
    # reduced image size
    # print('image shape after binning',h,w)

    # get cms
    h,w = img1_sm.shape
    ## threshold object cms
    if dark_bg:
        img1_sm = img1_sm > threshold_yen(img1_sm)
        yc, xc = ndi.center_of_mass(img1_sm)
    else:
        img1_sm = img1_sm < threshold_yen(img1_sm)
        yc, xc = ndi.center_of_mass(-img1_sm)
   
    # show intermediate steps for debugging
    if display:
        plt.subplot(211)
        plt.imshow(img1)
        plt.title('img1 original')
        plt.plot(xc*bin_factor+xmin, yc*bin_factor+ymin, 'ro')
        plt.plot( (xc*bin_factor + xmin) ,(yc*bin_factor+ymin), 'ro')
        rect = mpl.patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=1, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        plt.gca().add_patch(rect)
        
        plt.colorbar()
        plt.subplot(212)
        plt.imshow(img1_sm)
        plt.title('img1_reduced')
        plt.plot(xc, yc, 'ro')
        plt.colorbar()

    # print(yc, xc)
    return (yc-h//2)*bin_factor, (xc - w//2)*bin_factor


def cropCenterImage( image: np.ndarray, cropWidth: int, cropHeight: int) -> np.ndarray:
    """Crop the image at the center.

    Args:
        image (np.ndarray): image
        cropWidth (int): width of the cropped image
        cropHeight (int): height of the cropped image

    Returns:
        croppedImage (np.ndarray): the center cropped image
    """    
    y, x = image.shape
    startx = x//2-(cropWidth//2)
    starty = y//2-(cropHeight//2)    
    croppedImage = image[ starty:starty+cropHeight, startx:startx+cropWidth ]
    return croppedImage


def createTranslationMatrix(translation_x: float, translation_y: float) -> np.float32:
    """Create 2D translation matrix.

    Args:
        translation_x (float): translation in X axis
        translation_y (float): translation in Y axis

    Returns:
        translationMat (np.float32): A 3x3 translation matrix
    """    

    translationMat = np.float32([
        [1, 0, translation_x], 
        [0, 1, translation_y],
        [0, 0, 1]
    ], np.float32)

    return translationMat


def createScaleAndRotationMatrix(scale: float, rotation: float, center_rot_x: float, center_rot_y: float) -> np.ndarray:
    """Create 2D scale-and-rotation matrix around a specified center of rotation.

    Args:
        scale (float): scaling
        rotation (float): rotation angle in radian
        center_rot_x (float): center of rotation in X axis
        center_rot_y (float): center of rotation in Y axis

    Returns:
        matrix (np.ndarray): A 3x3 transformation matrix.
    """    
    cos = scale * math.cos(rotation)
    sin = scale * math.sin(rotation)

    matrix = np.array([
        [ cos,    -sin,      (1-cos) * center_rot_x + sin * center_rot_y ],
        [ sin,     cos,      (1-cos) * center_rot_y - sin * center_rot_x ],
        [ 0,        0,       1]
    ], np.float32)

    return matrix


def createRigidTransformationMat(translation_x: float, translation_y: float, rotation: float) -> np.ndarray:
    """Create the rigid transformation matrix. Assume center of rotation at origin.

    Args:
        translation_x (float): translation in x
        translation_y (float): translation in y
        rotation (radian) (float): rotation angle around origin

    Returns:
        mat (np.ndarray): the transformation matrix
    """    
    cos = math.cos(rotation)
    sin = math.sin(rotation)

    matrix = np.zeros([
        [cos,   -sin,   translation_x],
        [sin,   cos,    translation_y],
        [0,     0,      1],
    ], np.float32)
    
    return matrix


class DualColorImageCalibrator:
    
    mainSide: str
    dualColorImage: np.ndarray
    mainSideImage: np.ndarray
    minorSideImage: np.ndarray
    
    def processDualColorImage(self, dualColorImage: np.ndarray, mainSide: str, cropWidth: int, cropHeight: int) -> None:
        """Crop dual color image into main and minor side. Then apply filties for better visibility.

        Args:
            dualColorImage (np.ndarray): Dual color image.
            mainSide (str): The main side of the dual color.
            cropWidth (int): Crop width of the main and minor images.
            cropHeight (int): Crop height of the main and minor images.

        Returns:
            mainImg: main side image
            minorImg: minor side image
        """

        # Copy the dual color image
        self.dualColorImage = np.copy(dualColorImage)

        # Crop into main and minor side
        fullImg_h, fullImg_w = self.dualColorImage.shape
        if mainSide == 'Right':
            # Main at right side, minor at left
            self.mainSideImage = self.dualColorImage[:,fullImg_w//2:]
            self.minorSideImage = self.dualColorImage[:,:fullImg_w//2]

        elif mainSide == 'Left':
            # Main at right side, minor at left
            self.mainSideImage = self.dualColorImage[:,:fullImg_w//2]
            self.minorSideImage = self.dualColorImage[:,fullImg_w//2:]
        
        # Crop center of both images
        self.mainSideImage = cropCenterImage(self.mainSideImage, cropWidth, cropHeight)
        self.minorSideImage = cropCenterImage(self.minorSideImage, cropWidth, cropHeight)

        # Equalize Histogram
        #   create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        self.mainSideImage = clahe.apply(self.mainSideImage)
        self.minorSideImage = clahe.apply(self.minorSideImage)

        return self.mainSideImage, self.minorSideImage
    

    def calibrateMinorToMainTransformationMatrix(self) -> Tuple[float, float, float]:
        """Estimate transformation from minor side to main side. Only account for translation and rotation.

        Returns:
            translation_x (float): translation x
            translation_y (float): translation y
            rotation (float): rotation in radian
        """        

        # Create an ITK image from the numpy array
        mainSideImageITK = itk.GetImageFromArray(self.mainSideImage)
        minorSideImageITK = itk.GetImageFromArray(self.minorSideImage)
        
        # Create registration parameter object
        parameter_object = itk.ParameterObject.New()
        #   Set regid estimation parameters
        rigid_parameter_map = parameter_object.GetDefaultParameterMap('rigid')
        rigid_parameter_map['AutomaticScalesEstimation'] = ['false']
        rigid_parameter_map['WriteResultImage'] = ['false']
        rigid_parameter_map.erase('ResultImageFormat')
        rigid_parameter_map['NumberOfResolutions'] = ['7']

        parameter_object.AddParameterMap(rigid_parameter_map)

        # Execute image registration process
        result_image, result_transform_parameters = itk.elastix_registration_method(
            mainSideImageITK, minorSideImageITK,
            parameter_object= parameter_object,
            log_to_console= False
        )
        
        # Get transformation matrix result
        result_parameter_map = result_transform_parameters.GetParameterMap(0)
        rotation, translation_x, translation_y = ( float(x) for x in result_parameter_map['TransformParameters'] )

        mainToMinorTransformationMatrix = createRigidTransformationMat(translation_x, translation_y, rotation)

        # Compute the inverse transform, from minor to main
        minorToMainTransformationMatrix = np.linalg.inv( mainToMinorTransformationMatrix )

        # Extract translation and rotation parameter back
        translation_x = minorToMainTransformationMatrix[0,2]
        translation_y = minorToMainTransformationMatrix[1,2]

        cosTheta = minorToMainTransformationMatrix[0,0]
        sinTheta = minorToMainTransformationMatrix[1,0]
        theta = math.atan2( sinTheta, cosTheta )

        return translation_x, translation_y, theta
    

    @staticmethod
    def genMinorToMainMatrix(translation_x: float, translation_y: float, rotation: float, center_x: float, center_y: float):
        """Create a rigid transformation matrix (scale = 1) from minor to main side. 

        Args:
            translation_x (float): translation in X axis
            translation_y (float): translation in Y axis
            rotation (radian) (float): rotation angle in radian about center
            center_x (float): center of rotation in X
            center_y (float): center of rotation in Y

        Returns:
            transformationMatrix (np.ndarray): transformation matrix from minor to main
        """        
        # Compute the rotation matrix.
        rotationMat = createScaleAndRotationMatrix(1, rotation, center_x, center_y)

        # Compute the translation matrix
        translationMat = createTranslationMatrix(translation_x, translation_y)

        # Compute transformation matrix
        transformationMat = translationMat @ rotationMat
        
        return transformationMat

