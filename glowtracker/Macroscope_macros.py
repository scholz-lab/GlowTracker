# 
# IO, Utils
# 
import os
from multiprocessing.pool import ThreadPool
from queue import Queue
from tifffile import imwrite
from typing import Tuple, List
from pypylon import pylon

# 
# Own classes
# 
import Basler_control as basler
import Zaber_control as zaber

# 
# Math
# 
import math
import numpy as np
import scipy.ndimage as ndi
import matplotlib as mpl
import matplotlib.pylab as plt
plt.set_loglevel('warning')
from matplotlib.backends.backend_agg import FigureCanvasAgg
from skimage.filters import threshold_otsu, threshold_li, threshold_yen
from skimage.transform import downscale_local_mean
from skimage.registration import phase_cross_correlation
import itk
import cv2


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
        yc, xc = ndi.center_of_mass(~img1_sm)
   
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


class ImageSaver:
    """An image saver static class that manages multiples small saving threads.
    """    

    def __init__(self) -> None:
        pass
    

    @staticmethod
    def startSavingImageInQueueThread(imageQueue: Queue, numMaxThreads: int | None = None) -> None:
        """An image saving thread manager that spawn a fix number of threads that iteratively consume an image from a queue and save it.

        Args:
            imageQueue (Queue): 
            numMaxThreads (int | None, optional): Maximum number of small threads. Defaults to None.
        """

        # Create a thread pool
        consumerThreadPool = ThreadPool(processes= numMaxThreads)
        
        # Start spawn image saving workers
        for i in range(consumerThreadPool._processes):
            consumerThreadPool.apply_async(
                func= ImageSaver._imageSavingThreadWorker, 
                args= (imageQueue,)
            )
        
        # Wait until all workers are done
        consumerThreadPool.close()
        consumerThreadPool.join()

        print(f'Finished saving all the images')
    

    @staticmethod
    def _imageSavingThreadWorker(imageQueue: Queue):
        """A private function of image saving worker capabilities. The worker runs indefinitely
        , comsumes the image data from the queue and save it. Terminates when the data from
        the image queue is None.

        Args:
            imageQueue (Queue): _description_
        """        
        # Run until there is no more work
        while True:

            # Wait and retrieve an item from the queue
            queueItem = imageQueue.get(block= True)

            # check for signal of no more work
            if queueItem is not None:
                
                img, imgPath, imgFileName = queueItem
                # Save the image
                imwrite(os.path.join(imgPath, imgFileName), img)
                
            else:
                # put back on the queue for other consumers
                imageQueue.put(None)
                # shutdown the thread
                break


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


def swapMatXYOrder(matrix: np.ndarray) -> np.ndarray:
    """Modified a matrix such that the the multiplication operation 
    is suitable for vectors of order (y,x,..) from (x,y,...) or vice versa.

    Args:
        matrix (np.ndarray): An NxM matrix of size atleast 2x2

    Returns:
        matrixXYSwapped: An X,Y swapped version of the matrix
    """    
    matrixXYSwapped = np.copy(matrix)

    # Swap 1st and 2nd row
    matrixXYSwapped[[0, 1], :] = matrixXYSwapped[[1, 0], :]

    # Swap 1st and 2nd column
    matrixXYSwapped[:, [0, 1]] = matrixXYSwapped[:, [1, 0]]
    
    return matrixXYSwapped


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
    """Compute angle between the two vector 

    Args:
        vec1 (np.ndarray): vector of starting angle
        vec2 (np.ndarray): vector of ending angle

    Returns:
        angle (float): angle between the two vectors in radian
    """
    vec1normalized = vec1 / np.linalg.norm(vec1)
    vec2normalized = vec2 / np.linalg.norm(vec2)

    # Sin(theta)
    cosTheta = np.dot(vec1normalized, vec2normalized)
    # Cos(theta)
    sinTheta = np.cross(vec1normalized, vec2normalized)
    # Compute angle
    theta = math.atan2(sinTheta, cosTheta)

    return theta


def rotatePointAboutOrig(point: np.ndarray, rotation: float) -> np.ndarray:
    """Rotate a point about origin with a given angle.

    Args:
        point (np.ndarray): point in x,y plane
        rotation (float): rotation angle in deg

    Returns:
        point (np.ndarray): the rotated point
    """    
    rotationMatrix = createScaleAndRotationMatrix(1, rotation, 0, 0)[:2,:2]
    return rotationMatrix @ point


class CameraAndStageCalibrator:

    # Class attributes
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
            camera (pylon.InstantCamera): pylon camera to take images with.
            stage (zaber.Stage): zaber stage class to move the camera with.
            stepsize (float): movement step size in each stage basis axes.
            stepunits (str): stage movement step unit.
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
        stage.move_rel((0, -stepsize, 0), stepunits, wait_until_idle = True)

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
    

    def calibrateCameraAndStageTransform(self) -> None | Tuple[float, int, float]:
        """Estimate the transformation from stage space to image space using phase cross correlation in X and Y bases.

        Returns:
            Calibration parameters:
                - None : if calibration is not successfull.
                - Tuple[float, float, float] : if calibration is successful
                    - rotationStageToCam (float): rotation angle from stage
                    - imageNormalDir (int): image plane normal vector's direction (+X cross +Y in image space). Use to imply the direction of Y axis in camera-stage change of basis matrix. Possible results are +1 (for +Z) and -1 (for -Z).
                    - pixelsize (float): ratio bettween unit in stage space and pixel space (e.g. mm/px).
        """        

        # Estimate camera basis X 
        basisXPhaseShift, _, _ = phase_cross_correlation(self.basisOrigImage, self.basisXImage, upsample_factor= 1, space= 'real', return_error= 0, overlap_ratio= 0.5)    
    
        camBasisXVec = np.array([basisXPhaseShift[1], -basisXPhaseShift[0]], np.float32)
        camBasisXLen = np.linalg.norm(camBasisXVec)

        # Estimate camera basis Y
        basisYPhaseShift, _, _ = phase_cross_correlation(self.basisOrigImage, self.basisYImage, upsample_factor= 1, space= 'real', return_error= 0, overlap_ratio= 0.5)    
    
        camBasisYVec = np.array([basisYPhaseShift[1], -basisYPhaseShift[0]], np.float32)
        camBasisYLen = np.linalg.norm(camBasisYVec)

        # Check if any estimated phase shift is nan or zero
        if np.any(np.isnan(np.hstack((camBasisXVec, camBasisYVec)))) \
            or np.equal(camBasisXLen, 0) or np.equal(camBasisYLen, 0):
            return None

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
        
        camBasisXVec = rotatePointAboutOrig(camBasisXVec, basisXCompensatedAngle)
        camBasisYVec = rotatePointAboutOrig(camBasisYVec, basisYCompensatedAngle)

        # Compute rotation angle from stage to camera
        normCamBasisXVec = camBasisXVec / camBasisXLen
        rotationStageToCam = computeAngleBetweenTwo2DVecs( 
            np.array([1., 0.], np.float32), 
            normCamBasisXVec
        )

        # Compute pixelsize
        pixelSize_X = self.stepsize / camBasisXLen
        pixelSize_Y = self.stepsize / camBasisYLen
        #   Average between the two
        pixelSize = (pixelSize_X + pixelSize_Y) / 2

        return (rotationStageToCam, signAngleBetweenXYBasis, pixelSize)

    
    @staticmethod
    def genImageToStageMatrix(rotation: float, imageNormalDir: int, pixelSize: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute trasnformation matrix from image space to stage space using the given rotation angle, sign of cross product between Camera Space X,Y basis, and pixelsize. Assume only rotation and uniform scaling.

        Args:
            rotationStageToCam (float): rotation angle from stage
            imageNormalDir (int): image plane normal vector's direction (+X cross +Y in image space). Use to imply the direction of Y axis in camera-stage change of basis matrix. Possible results are +1 (for +Z) and -1 (for -Z).
            pixelsize (float): ratio bettween unit in stage space and pixel space (e.g. mm/px).

        Returns:
            - imageToStageMat (np.ndarray): transformation matrix from image space to stage space
            - imageToStageRotOnlyMat (np.ndarray): transformation matrix from image space to stage space without uniform scaling
        """    
        # Stage to Image. Standard 2D rotation matrix
        cosval, sinval = math.cos(rotation), math.sin(rotation)

        camBasisXVec = np.array([cosval, sinval])
        camBasisYVec = rotatePointAboutOrig(camBasisXVec, math.pi/2 * imageNormalDir)

        stageToImageMat = np.array([
            [camBasisXVec[0], camBasisYVec[0]],
            [camBasisXVec[1], camBasisYVec[1]]
        ], np.float32)

        # Stage to Image
        imageToStageMat = np.linalg.inv( stageToImageMat )

        # switch x,y to y,x
        imageToStageMat = swapMatXYOrder(imageToStageMat)

        return pixelSize * imageToStageMat, imageToStageMat


class DualColorImageCalibrator:
    
    # Class attributes
    mainSide: str
    dualColorImage: np.ndarray
    mainSideImage: np.ndarray
    minorSideImage: np.ndarray
    
    def __init__(self) -> None:
        pass

    
    def processDualColorImage(self, dualColorImage: np.ndarray, mainSide: str) -> None:
        """Crop dual color image into main and minor side and apply histrogram equalization 
        for better visibility.

        Args:
            dualColorImage (np.ndarray): Dual color image.
            mainSide (str): The main side of the dual color.

        Returns:
            mainImg: main side image
            minorImg: minor side image
        """

        # Copy the dual color image
        self.dualColorImage = np.copy(dualColorImage)

        # Crop into main and minor side
        fullImg_w = self.dualColorImage.shape[1]
        if mainSide == 'Right':
            # Main at right side, minor at left
            self.mainSideImage = self.dualColorImage[:,fullImg_w//2:]
            self.minorSideImage = self.dualColorImage[:,:fullImg_w//2]

        elif mainSide == 'Left':
            # Main at right side, minor at left
            self.mainSideImage = self.dualColorImage[:,:fullImg_w//2]
            self.minorSideImage = self.dualColorImage[:,fullImg_w//2:]
        
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


def renderChangeOfBasisImage(stageToImageMat: np.ndarray) -> np.ndarray:
    """A utility function for plotting a 2D Change of Basis matrix and saving into an image data.

    Args:
        stageToImageMat (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """    
    # Define the coordinates for the vectors
    stageX = [1, 0]  # Vector from origin to (1,0)
    stageY = [0, 1]  # Vector from origin to (0,1)
    imageX = [stageToImageMat[0, 0], stageToImageMat[1, 0]]
    imageY = [stageToImageMat[0, 1], stageToImageMat[1, 1]]

    # Create the plot
    fig = plt.figure(figsize=(6, 6))
    
    def drawVectorFromOrigWithAnnotation(point: List[float], color: str, name: str, linestyle: str) -> None:
        plt.quiver(*[0, 0], *point, color= color, scale= 1, scale_units= 'xy', angles= 'xy', label= name, linestyle= linestyle, linewidth= 1, facecolor= color)
        plt.annotate(name, (point[0], point[1]), textcoords= 'offset points', xytext= (10,10), \
                        ha= 'center', fontsize= 12, color= 'black')
    
    drawVectorFromOrigWithAnnotation(stageX, 'r', 'Stage +X', linestyle= 'solid')
    drawVectorFromOrigWithAnnotation(stageY, 'r', 'Stage +Y', linestyle= 'solid')
    drawVectorFromOrigWithAnnotation(imageX, 'g', 'Image +X', linestyle= 'dashed')
    drawVectorFromOrigWithAnnotation(imageY, 'g', 'Image +Y', linestyle= 'dashed')

    # Set plot limits and labels
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.xlabel('X')
    plt.ylabel('Y')

    # Add grid and legend
    plt.grid(True)
    plt.legend()

    # Render the plot to a numpy array
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    width, height = fig.get_size_inches() * fig.get_dpi()
    image_array = np.frombuffer(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)

    return image_array