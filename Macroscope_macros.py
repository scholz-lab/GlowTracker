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
import Zaber_control as zaber
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

    translationMat = np.array([
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

    matrix = np.array([
        [cos,   -sin,   translation_x],
        [sin,   cos,    translation_y],
        [0,     0,      1],
    ], np.float32)
    
    return matrix


def computeAngleBetweenTwo2DVecs(vec1: np.ndarray, vec2: np.ndarray) -> float:
    

    vec1normalized = vec1 / np.linalg.norm(vec1)
    vec2normalized = vec2 / np.linalg.norm(vec2)

    
    # Dot product
    cosTheta = np.dot(vec1normalized, vec2normalized)

    # Cross product
    sinTheta = np.cross(vec1normalized, vec2normalized)
    
    # Compute angle
    theta = math.atan2(sinTheta, cosTheta)

    return theta


class CameraAndStageCalibrator:

    origImage: np.ndarray
    basisXImage: np.ndarray
    basisYImage: np.ndarray
    stepsize: float
    stepunit: str

    def __init__(self) -> None:
        pass


    def takeCalibrationImage(self, camera: pylon.InstantCamera, stage: zaber.Stage, stepsize: float, stepunits: str, dualColorMode: bool = False, dualColorModeMainSide: str = 'Right') -> Tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Take images for camera&stage transformation matrix calibration.

        Args:
            camera (pylon.InstantCamera): pylon camera to take images with
            stage (zaber.Stage): zaber stage class to move the camera with
            stepsize (float): movement step size in each stage basis axes
            stepunits (str): stage movement step unit
            dualColorMode (bool): is in dual color mode. Defaults to False.
            dualColorModeMainSide (str): if in dual color mode, which side is the main side.Defaults to 'Right'.

        Returns:
            - None: if taking images is not successful
            - Tuple(basisImageOrig, basisImageX, basisYImage): if taking images is successful
        """        

        self.stepsize = stepsize
        self.stepunits = stepunits

        isSuccessImageOrig, self.basisOrigImage = basler.single_take(camera)

        # Move along stage's X axis
        stage.move_rel((stepsize, 0, 0), stepunits, wait_until_idle = True)
        isSuccessImageX, self.basisXImage = basler.single_take(camera)

        # Move along stage's Y axis
        stage.move_rel((-stepsize, 0, 0), stepunits, wait_until_idle = True)
        stage.move_rel((0, stepsize, 0), stepunits, wait_until_idle = True)
        isSuccessImageY, self.basisYImage = basler.single_take(camera)

        # Move back to origin
        stage.move_rel((-stepsize, stepsize, 0), stepunits, wait_until_idle = True)

        # If taking image is not successful then return None
        if not (isSuccessImageOrig and isSuccessImageX and isSuccessImageY):
            return None, None, None
        
        # If in dual color mode then crop only relavent region
        if dualColorMode:
            h, w = self.basisXImage.shape

            if dualColorModeMainSide == 'Left':
                self.basisOrigImage = self.basisOrigImage[:,:w//2]
                self.basisXImage = self.basisXImage[:,:w//2]
                self.basisYImage = self.basisYImage[:,:w//2]

            elif dualColorModeMainSide == 'Right':
                self.basisOrigImage = self.basisOrigImage[:,w//2:]
                self.basisXImage = self.basisXImage[:,w//2:]
                self.basisYImage = self.basisYImage[:,w//2:]
        
        return self.basisOrigImage, self.basisXImage, self.basisYImage
    

    def calibrateCameraToStageTransform(self) -> np.ndarray:

        # Estimate camera basis X 
        basisXPhaseShift = phase_cross_correlation(self.basisOrigImage, self.basisXImage, upsample_factor= 1, space= 'real', return_error= 0, overlap_ratio= 0.5)    
    
        camBasisXVec = np.array([basisXPhaseShift[1], -basisXPhaseShift[0]], np.float32)
        camBasisXLen = np.linalg.norm(camBasisXVec)

        # Estimate camera basis Y
        basisYPhaseShift = phase_cross_correlation(self.basisOrigImage, self.basisYImage, upsample_factor= 1, space= 'real', return_error= 0, overlap_ratio= 0.5)    
    
        camBasisYVec = np.array([basisYPhaseShift[1], -basisYPhaseShift[0]], np.float32)
        camBasisYLen = np.linalg.norm(camBasisYVec)


        # Compute angle between the two basis 
        angleBetweenXYBasis = computeAngleBetweenTwo2DVecs( camBasisXVec, camBasisYVec )
        signAngleBetweenXYBasis = int(np.sign(angleBetweenXYBasis))

        absAngleBetweenXYBasis = abs(angleBetweenXYBasis)

        # Compensate the angle between if it's not 90 deg by rotation basis X, Y outward/inward
        #   with relative to normal vector +Z.
        absDiffAngle = abs(math.pi/2 - absAngleBetweenXYBasis)
        diffAngleHalf = absDiffAngle / 2
        basisXCompensatedAngle = 0
        basisYCompensatedAngle = 0
        
        if absAngleBetweenXYBasis < math.pi/2:
        
            basisXCompensatedAngle = -1 * signAngleBetweenXYBasis * diffAngleHalf
            basisYCompensatedAngle = +1 * signAngleBetweenXYBasis * diffAngleHalf

        elif absAngleBetweenXYBasis > math.pi/2:

            basisXCompensatedAngle = +1 * signAngleBetweenXYBasis * diffAngleHalf
            basisYCompensatedAngle = -1 * signAngleBetweenXYBasis * diffAngleHalf
            

        # Rotate the basis by the compensation amount
        def rotateVecAboutOrig(vec: np.ndarray, rotation: float) -> np.ndarray:
            rotationMatrix = createScaleAndRotationMatrix(1, rotation, 0, 0)[:2,:2]
            return rotationMatrix @ vec
        
        newCamBasisXVec = rotateVecAboutOrig(camBasisXVec, basisXCompensatedAngle)
        newCamBasisYVec = rotateVecAboutOrig(camBasisYVec, basisYCompensatedAngle)

        import matplotlib.pyplot as plt

        plt.figure()
        origin = np.zeros(shape=(2,6), dtype= np.float32)
        X = np.array([
            self.stepsize,
            0,
            camBasisXVec[0],
            camBasisYVec[0],
            newCamBasisXVec[0],
            newCamBasisYVec[0],
        ])

        Y = np.array([
            0,
            self.stepsize,
            camBasisXVec[1],
            camBasisYVec[1],
            newCamBasisXVec[1],
            newCamBasisYVec[1],
        ])

        plt.quiver(*origin, X, Y, color= ['r', 'tab:pink', 'g', 'tab:olive', 'b', 'tab:cyan'], scale= 1.0)
        plt.show()

        camBasisXVec = newCamBasisXVec
        camBasisYVec = newCamBasisYVec

        # Compute change of basis matrix
        camToStageMat = np.array([
            [camBasisXVec[0], camBasisYVec[0]], 
            [camBasisXVec[1], camBasisYVec[1]], 
        ])

        # Compute the unscaled version
        unscaledCamBasisXVec = camBasisXVec / camBasisXLen
        unscaledCamBasisYVec = camBasisYVec / camBasisYLen

        unscaledCamToStageMat = np.array([
            [unscaledCamBasisXVec[0], unscaledCamBasisYVec[0]], 
            [unscaledCamBasisXVec[1], unscaledCamBasisYVec[1]], 
        ])

        # Compute pixelsize
        pixelSize_X = self.stepsize / camBasisXLen
        pixelSize_Y = self.stepsize / camBasisYLen
        #   Average between the two
        pixelsize = (pixelSize_X + pixelSize_Y) / 2

        return camToStageMat, unscaledCamToStageMat, pixelsize


class DualColorImageCalibrator:
    
    mainSide: str
    dualColorImage: np.ndarray
    mainSideImage: np.ndarray
    minorSideImage: np.ndarray
    
    def __init__(self) -> None:
        pass

    
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

