#!/usr/bin/env python3
import datetime
import os
from pickle import TRUE
import time
from skimage.io import imread
import numpy as np
import matplotlib.pylab as plt
import sys
import inspect

# Append parent directory
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

from Microscope_macros import extractWorms, extractWormsDiff, extractWormsCMS

single = True
mode = 'CMS'
# test a single image pair to optimize parameters
if single:
    if mode=='Diff':
        img1 = imread('/media/nif9202/Monika/KITP/Larvae/Larvae_testTracking/2022-06-15-22-51-41-basler_0.tiff')
        img2 = imread('/media/nif9202/Monika/KITP/Larvae/Larvae_testTracking/2022-06-15-22-51-41-basler_10.tiff')
        
        t0 = time.time()
        c = extractWormsDiff(img1, img2, capture_radius = 300, bin_factor=10, area = 200, threshold = 12, dark_bg = True, display=True)
        t1 = time.time()
        plt.show()

        print(f'duration: {t1-t0}, {c}')
    elif mode=='CMS':
        img1 = imread(r'C:\Users\shomali\Desktop\recording_3\2023-11-21-16-17-49-basler_1.tiff')
        
        t0 = time.time()
        cy, cx, intermediate_images, annotated_mask = extractWormsCMS(img1, capture_radius = 1000, bin_factor=5, dark_bg = True, display=TRUE)
        t1 = time.time()
        
        plt.subplot(233)
        plt.imshow(intermediate_images[0], cmap='gray')
        plt.title('Resized image')

        plt.subplot(234)
        plt.imshow(intermediate_images[1], cmap='gray')
        plt.title('Thresholded image')

        plt.subplot(235)
        plt.imshow(intermediate_images[2], cmap='gray')
        plt.title('Eroded & dilated image')

        plt.subplot(236)
        plt.imshow(annotated_mask, cmap='gray')
        plt.title('Distances')

        plt.tight_layout()
        plt.show()

        print(f'duration: {t1-t0}, {(cx, cy)}')

       
    else:
        img1 = imread('/media/nif9202/Monika/KITP/Larvae/Larvae_testTracking/2022-06-15-22-51-41-basler_0.tiff')
        #img2 = imread('/media/nif9202/Monika/KITP/Larvae/Larvae_testTracking/2022-06-15-22-51-41-basler_2.tiff')
        #print(img1.shape, img1.T.shape)
        t0 = time.time()
        c = extractWorms(img1, capture_radius = 1000, bin_factor=10, dark_bg = True, display=TRUE)
        t1 = time.time()
        plt.show()

        print(f'duration: {t1-t0}, {c}')


else:
    for i in range(15, 30):
        print(f'Frame difference: {i}')
        img1 = imread('/media/nif9202/Monika/KITP/Larvae/Larvae_testTracking/2022-06-15-22-51-41-basler_0.tiff')#.T
        img2 = imread(f'/media/nif9202/Monika/KITP/Larvae/Larvae_testTracking/2022-06-15-22-51-41-basler_{i}.tiff')#.T
        t0 = time.time()
        extractWorms(img1.T, img2.T, capture_radius = -1, bin_factor=20, minimal_difference = 0.01, dark_bg = True, display=True)
        t1 = time.time()
        plt.show()

        print(f'duration: {t1-t0}')
