#!/usr/bin/env python3
import datetime
import os
from pickle import TRUE
import time
from skimage.io import imread
import numpy as np
import matplotlib.pylab as plt
import sys
sys.path.append(".")
from Macroscope_macros import extractWorms, extractWormsDiff

single = True
diff = True
# test a single image pair to optimize parameters
if single:
    if diff:
        img1 = imread('/media/nif9202/Monika/KITP/Larvae/Larvae_testTracking/2022-06-15-22-51-41-basler_0.tiff')
        img2 = imread('/media/nif9202/Monika/KITP/Larvae/Larvae_testTracking/2022-06-15-22-51-41-basler_10.tiff')
        print(img1.shape, img1.T.shape)
        t0 = time.time()
        c = extractWormsDiff(img1, img2, capture_radius = 300, bin_factor=10, area = 200, threshold = 12, dark_bg = True, display=True)
        t1 = time.time()
        plt.show()

        print(f'duration: {t1-t0}, {c}')
    else:
        img1 = imread('/media/nif9202/Monika/KITP/Larvae/Larvae_testTracking/2022-06-15-22-51-41-basler_170.tiff')
        #img2 = imread('/media/nif9202/Monika/KITP/Larvae/Larvae_testTracking/2022-06-15-22-51-41-basler_2.tiff')
        #print(img1.shape, img1.T.shape)
        t0 = time.time()
        c = extractWorms(img1, capture_radius = 1000, bin_factor=25, dark_bg = True, display=TRUE)
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
