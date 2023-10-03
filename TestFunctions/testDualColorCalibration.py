import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from skimage.registration import phase_cross_correlation
from skimage.exposure import match_histograms
from skimage.transform import SimilarityTransform

import cv2

import itk

# Provide the test image path and ground truth transformation here
DUAL_COLOR_IMG_PATH = ''

GROUND_TRUTH_TRANSLATION_X = -59
GROUND_TRUTH_TRANSLATION_Y = -71
GROUND_TRUTH_ROTATION_DEG = 4.5
GROUND_TRUTH_ROTATION_RADIAN = GROUND_TRUTH_ROTATION_DEG * math.pi / 180.0
GROUND_TRUTH_SCALE = 1.0


def load_tiff_and_convert_to_gray_and_float(input_path):
    # Load the TIFF image
    image = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    if image is None:
        print("Error: Unable to load the TIFF image.")
        return None

    # Convert to floating-point type
    float_image = image.astype(np.float32)

    # Normalize from [0, 255] to [0, 1]
    float_image /= 255

    return float_image


def cropCenterImage( img: np.ndarray, cropx: int, cropy: int) -> np.ndarray:
    y, x = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)    
    return img[ starty:starty+cropy, startx:startx+cropx ]


def normalizeMinMaxToZeroOne( img: np.ndarray ) -> None:
    imgMin, imgMax = img.min(), img.max()
    img = (img - imgMin)/(imgMax - imgMin)
    return img


def equalizeHistogram( img: np.ndarray, isLocalMode: bool= False ) -> np.ndarray:

    # Convert float to 8 bit uint
    img_8uint = (img * 255).astype(np.uint8)

    if not isLocalMode:
        img_equalized_8uint = cv2.equalizeHist(img_8uint)

    else:
        # create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_equalized_8uint = clahe.apply(img_8uint)

    # Convert back to float
    return (img_equalized_8uint / 255).astype(np.float32)


def translationAndRotationToMat3x3(translation_x, translation_y, rotation):
    # Create the similarity transformation matrix
    matrix = np.zeros((3, 3))
    matrix[0, 0] = np.cos(rotation)
    matrix[0, 1] = -np.sin(rotation)
    matrix[0, 2] = translation_x
    matrix[1, 0] = np.sin(rotation)
    matrix[1, 1] = np.cos(rotation)
    matrix[1, 2] = translation_y
    matrix[2, 2] = 1
    return matrix


def applyTransformationSRTToImage(img: np.ndarray, scale: float, rotation: float, translation_x: float, translation_y: float) -> np.ndarray:
    """Apply transformation SRT to an image

    Args:
        img (np.ndarray): image
        scale (float): scale
        rotation (float): rotation in degree (0, 360)
        translation_x (float): translation in x
        translation_y (float): translation in y

    Returns:
        np.ndarray: The transformed image
    """    
    
    # Compute the rotation matrix. The cv2.getRotationMatrix2D returns 2x3 mat so we will need to fill in the last row
    rotationMat = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), rotation, scale)
    rotationMat = np.vstack([ rotationMat, np.array([0,0,1], np.float32) ])

    # Compute the translation matrix
    translationMat = np.float32([
        [1, 0, translation_x], 
        [0, 1, translation_y],
        [0, 0, 1]
    ])

    # Compute transformation matrix
    transformationMat = translationMat @ rotationMat

    # Apply transformation
    #   Use warpAffine() here instead of warpPerspective for a little faster computation.
    #   It also takes 2x3 mat, so we cut the last row out accordingly.
    translated_image = cv2.warpAffine(img, transformationMat[:2,:], (img.shape[1], img.shape[0]))

    return translated_image


def siftFeatureMatching(srcImg: np.ndarray, targetImg: np.ndarray) -> None:

    # Convert from float to uint8
    srcImg_8bit = (srcImg * 255).astype(np.uint8)
    targetImg_8bit = (targetImg * 255).astype(np.uint8)

    sift = cv2.xfeatures2d.SIFT_create()

    srcKeypoints, srcDescriptors = sift.detectAndCompute(srcImg_8bit, None)
    targetKeypoints, targetDescriptors = sift.detectAndCompute(targetImg_8bit, None)

    # feature matching
    # NORM_HAMMING
    bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck= True)

    matches = bf.match(srcDescriptors, targetDescriptors)
    matches = sorted( matches, key= lambda x: x.distance )

    featureMatchingImage = cv2.drawMatches(srcImg_8bit, srcKeypoints, targetImg_8bit, targetKeypoints, matches[:50], targetImg_8bit, flags= 2)

    plt.figure()
    plt.title('SIFT match')
    plt.imshow(featureMatchingImage)

    # Pack keypoints
    srcPoints = []
    targetPoints = []

    for match in matches:
        srcPoints.append( srcKeypoints[match.queryIdx].pt )
        targetPoints.append( targetKeypoints[match.trainIdx].pt )

    return srcPoints, targetPoints


def estimate_similarity_transform(src_pts, dst_pts):
    # Estimate the translation by finding the mean shift
    src_mean = np.mean(src_pts, axis=0)
    dst_mean = np.mean(dst_pts, axis=0)
    translation = dst_mean - src_mean

    # # Estimate the rotation using the Kabsch algorithm
    # H = np.dot((src_pts - src_mean).T, (dst_pts - dst_mean))
    # U, _, Vt = np.linalg.svd(H)
    # rotation = np.dot(U, Vt).T

    # # Create the similarity transformation matrix
    # similarity_transform = np.zeros((3, 3))
    # similarity_transform[:2, :2] = rotation
    # similarity_transform[:2, 2] = translation
    # similarity_transform[2, 2] = 1.0

    # Estimate the rotation using the difference in angles
    src_angle = np.arctan2(src_pts[:, 1] - src_mean[1], src_pts[:, 0] - src_mean[0])
    dst_angle = np.arctan2(dst_pts[:, 1] - dst_mean[1], dst_pts[:, 0] - dst_mean[0])
    rotation_angle = np.mean(dst_angle - src_angle)

    # Create the similarity transformation matrix
    similarity_transform = translationAndRotationToMat3x3(translation[0], translation[1], rotation_angle)

    return similarity_transform


def ransac_image_registration(src_pts, dst_pts, num_iterations=1000, threshold=5.0):
    best_model = None
    max_inliers = 0

    src_pts_np = np.array(src_pts)
    dst_pts_np = np.array(dst_pts)

    for _ in range(num_iterations):
        # Randomly select 2 correspondences
        random_indices = np.random.choice(len(src_pts_np), 20, replace=False)
        src_sample = src_pts_np[random_indices]
        dst_sample = dst_pts_np[random_indices]

        # Compute the similarity transformation
        model = estimate_similarity_transform(src_sample, dst_sample)

        # Transform all source points using the model
        transformed_pts = np.dot(model[:2, :2], src_pts_np.T).T + model[:2, 2]

        # Calculate the inliers (points close enough to the transformed points)
        distances = np.linalg.norm(transformed_pts - dst_pts_np, axis=1)
        inliers = np.count_nonzero(distances < threshold)

        # Update the best model if this iteration produced more inliers
        if inliers > max_inliers:
            best_model = model
            max_inliers = inliers

    return best_model


def dualColorImageProcessing(dualColorImg: np.ndarray, mainSide: str = 'Right', cropWidth: int = 500, cropHeight: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """Crop dual color image into main and minor side. Then apply filties for better visibility.

    Args:
        dualColorImg (np.ndarray): Dual color image
        mainSide (str, optional): The main side of the dual color. Options are 'Left', and 'Right'. Defaults to 'Right'.
        cropWidth (int, optional): Crop width of the main and minor images. Defaults to 500.
        cropHeight (int, optional): Crop height of the main and minor images. Defaults to 500.

    Returns:
        mainImg: main side image
        minorImg: minor side image
    """
    mainImg, minorImg = None, None

    if mainSide == 'Right':
        # Main at right side, minor at left
        fullImg_h, fullImg_w = dualColorImg.shape
        mainImg = dualColorImg[:fullImg_h//2,fullImg_w//2:]
        minorImg = dualColorImg[:fullImg_h//2,:fullImg_w//2]

    elif mainSide == 'Left':
        # Main at right side, minor at left
        fullImg_h, fullImg_w = dualColorImg.shape
        mainImg = dualColorImg[:fullImg_h//2,:fullImg_w//2]
        minorImg = dualColorImg[:fullImg_h//2,fullImg_w//2:]

    # Crop center of both images
    mainImg = cropCenterImage(mainImg, cropWidth, cropHeight)
    minorImg = cropCenterImage(minorImg, cropWidth, cropHeight)

    # Equalize Histogram
    mainImg = equalizeHistogram(mainImg, isLocalMode= True)
    minorImg = equalizeHistogram(minorImg, isLocalMode= True)

    # Normalize both to full range 0, 1
    mainImg = normalizeMinMaxToZeroOne(mainImg)
    minorImg = normalizeMinMaxToZeroOne(minorImg)

    # Invert minor to same color as main
    minorImg = 1.0 - minorImg

    # Match histogram from minorImage to majorImage
    minorImg = match_histograms(minorImg, mainImg)

    return mainImg, minorImg


def estimateMainToMinorTransformation(mainImg: np.ndarray, minorImg: np.ndarray, mode: str = 'SIFT') -> Tuple[float, float, float]:
    """Estimate transformation from minor side to main side. Only account for translation and rotation.

    Args:
        mainImg (np.ndarray): main side image.
        minorImg (np.ndarray): minor side iamge.
        mode (str): choose the estimation techniques. Options: 'PHASE_CROSS', 'SIFT'. Defaults to 'SIFT'

    Returns:
        minorToMainTransformationMatrix: a 3x3 transformation matrix from minor to main.
    """    

    minorToMainTransformationMatrix: np.ndarray = np.zeros((3,3), np.float32)

    if mode == 'SIFT':
        # 
        # SIFT feature
        # 
        srcPoints, targetPoints = siftFeatureMatching(minorImg, mainImg)

        # Cutoff
        srcPoints = srcPoints[:80]
        targetPoints = targetPoints[:80]

        minorToMainTransformationMatrix = ransac_image_registration(src_pts= srcPoints, dst_pts= targetPoints)
    
    elif mode == 'PHASE_CROSS':

        shift = phase_cross_correlation(mainImg, minorImg, upsample_factor=1, space='real', return_error=0, overlap_ratio=0.5)

        translation_x, translation_y = shift[0], shift[1]
        # res = find_rotation(mainImg, minorImg)

        # Estimate similarity transform (includes rotation and scaling)
        tform = SimilarityTransform(translation=(translation_x, translation_y))
        tform.estimate(np.array(mainImg), np.array(minorImg))

        # Extract rotation and scaling from the transformation matrix
        rotation = np.arctan2(tform.params[0, 1], tform.params[0, 0])

        minorToMainTransformationMatrix = translationAndRotationToMat3x3(translation_x, translation_y, rotation)
    
    elif mode == 'elastix':

        # Create an ITK image from the numpy array and then write it out
        mainImg_copy = (mainImg * 255).astype(np.uint8)
        minorImg_copy = (minorImg * 255).astype(np.uint8)
        
        mainImg_itk = itk.GetImageFromArray(mainImg_copy)
        minorImg_itk = itk.GetImageFromArray(minorImg_copy)
        
        # Set parameters
        parameter_object = itk.ParameterObject.New()
        
        #   Set regid estimation parameters
        rigid_parameter_map = parameter_object.GetDefaultParameterMap('rigid')
        rigid_parameter_map['AutomaticScalesEstimation'] = ['false']
        rigid_parameter_map['WriteResultImage'] = ['false']
        rigid_parameter_map.erase('ResultImageFormat')
        rigid_parameter_map['NumberOfResolutions'] = ['7']

        parameter_object.AddParameterMap(rigid_parameter_map)

        # Call registration function
        result_image, result_transform_parameters = itk.elastix_registration_method(
            mainImg_itk, minorImg_itk,
            parameter_object= parameter_object,
            log_to_console= False)
        
        # Get transformation matrix result
        result_parameter_map = result_transform_parameters.GetParameterMap(0)
        rotation, translation_x, translation_y = ( float(x) for x in result_parameter_map['TransformParameters'] )

        mainToMinorTransformationMatrix = translationAndRotationToMat3x3(translation_x, translation_y, rotation)

        minorToMainTransformationMatrix = np.linalg.inv( mainToMinorTransformationMatrix )


    # Print the rotation and scaling
    np.set_printoptions(precision= 3, suppress= True)
    print('Estimation:')
    print(minorToMainTransformationMatrix)

    groundTruthTransformation = translationAndRotationToMat3x3(GROUND_TRUTH_TRANSLATION_X, GROUND_TRUTH_TRANSLATION_Y, GROUND_TRUTH_ROTATION_RADIAN)
    print('Ground Truth:')
    print(groundTruthTransformation)

    print(f'Diff with ground truth:')
    print(groundTruthTransformation - minorToMainTransformationMatrix)

    return minorToMainTransformationMatrix


if __name__ == '__main__':

    # Load image
    dualColorImg = load_tiff_and_convert_to_gray_and_float(DUAL_COLOR_IMG_PATH)

    # Main and Minor image processing
    cropWidth, cropHeight = 700, 700
    mainImg, minorImg = dualColorImageProcessing(dualColorImg, 'Right', cropWidth, cropHeight)
    
    # Transformation Estimation
    minorToMainTransformatinonMatrix = estimateMainToMinorTransformation(mainImg, minorImg, mode= 'elastix')

    # Apply the estimated transform
    tx = minorToMainTransformatinonMatrix[0,2]
    ty = minorToMainTransformatinonMatrix[1,2]
    cosTheta = minorToMainTransformatinonMatrix[0,0]
    theta = math.acos(cosTheta)
    transformedMinorImage = applyTransformationSRTToImage(minorImg, 1, theta * 180 / math.pi, tx, ty)

    # Combined main and the transformed minor image
    combinedImage = np.zeros(shape= (cropHeight, cropWidth, 3), dtype= np.float32)
    combinedImage[:,:,0] = mainImg
    combinedImage[:,:,1] = transformedMinorImage

    # 
    # Ground truth
    # 
    GTTransformedMinorImage = applyTransformationSRTToImage(minorImg, GROUND_TRUTH_SCALE, GROUND_TRUTH_ROTATION_DEG, GROUND_TRUTH_TRANSLATION_X, GROUND_TRUTH_TRANSLATION_Y)

    # Create a ground truth combined image
    GTCombinedImage = np.zeros(shape= (cropHeight, cropWidth, 3), dtype= np.float32)
    GTCombinedImage[:,:,0] = mainImg
    GTCombinedImage[:,:,1] = GTTransformedMinorImage

    # 
    # Plot
    # 

    plt.figure()
    plt.imshow(dualColorImg, vmin= 0, vmax= 1, cmap= 'gray')
    
    subplotRows = 2
    subplotColumns = 4
    subfig = plt.figure( figsize= (11, 6) )
    ax1 = plt.subplot(subplotRows, subplotColumns, 1)
    ax1.imshow(mainImg, vmin= 0, vmax= 1, cmap= 'gray')
    ax1.set_title('main')

    ax2 = plt.subplot(subplotRows, subplotColumns, 2)
    ax2.imshow(minorImg, vmin= 0, vmax= 1, cmap= 'gray')
    ax2.set_title('minor')

    ax3 = plt.subplot(subplotRows, subplotColumns, 3)
    ax3.imshow(transformedMinorImage, vmin= 0, vmax= 1, cmap= 'gray')
    ax3.set_title('transformed minor')

    ax4 = plt.subplot(subplotRows, subplotColumns, 4)
    ax4.imshow(GTTransformedMinorImage, vmin= 0, vmax= 1, cmap= 'gray')
    ax4.set_title('GT transformed minor')

    ax5 = plt.subplot(subplotRows, subplotColumns, 5)
    mainImgHist = cv2.calcHist(mainImg, channels= [0], mask= None, histSize= [100], ranges= [0, 1])
    ax5.plot(mainImgHist)
    ax5.set_title('main histogram')

    ax6 = plt.subplot(subplotRows, subplotColumns, 6)
    # Histograms
    minorImgHist = cv2.calcHist(minorImg, channels= [0], mask= None, histSize= [100], ranges= [0, 1])
    ax6.plot(minorImgHist)
    ax6.set_title('minor histogram')

    ax7 = plt.subplot(subplotRows, subplotColumns, 7)
    ax7.imshow(combinedImage, vmin= 0, vmax= 1)
    ax7.set_title('combined')

    ax8 = plt.subplot(subplotRows, subplotColumns, 8)
    ax8.imshow(GTCombinedImage, vmin= 0, vmax= 1)
    ax8.set_title('GT combined')

    plt.show()