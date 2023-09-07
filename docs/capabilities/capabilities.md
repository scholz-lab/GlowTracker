---
title: Capabilities
layout: home
---

# Capabilities

The macroscope GUI is designed to allow simple interaction with the stage and camera, and to start pre-built tracking routines. It does not provide access to all camera and stage settings. Use the camera and stage software to set these parameters and load these presets into the GUI.

## Homing
Homing will drive the three axes of the stage to their origin (0,0,0). This has to be done whenever the stage has been unplugged to enable correct absolute motion.
Problem: Stage not homing
Solution: Toggle on homing on start up in the settings and restart the GUI

## Calibration
To track, the software needs to be able to convert changes in the image to change in the real space i.e, what number of pixels corresponds to what motion of the stage. The ‘calibrate’ button moves the stage a defined amount, takes two images and compares the shift. If successful, the calibration will display a pixelsize in um/px and a rotation of the camera relative to the stage.

Problem : Calibration fails with inf pixelsize
This could be either due to motion of the object in the image (if an animal), non-ideal image quality or because the step size is too small or too large. 
Solution 1:
First, rerun the calibration a few times to see if you get a successful value. Otherwise, adjust the calibration step size in the settings. If all fails, you can also manually enter calibration parameters in the GUI settings.
Solution 2:
Use a reference slide that contains well-defined structures to run the calibration protocol.


## Tracking
Tracking is toggled on and off using the check box in the lower right of the GUI. When tracking is started, the display ROI will be cropped to the size specified in the Settings->Tracking. The settings also specify which of 3 tracking algorithms is used. 
Settings that apply to all algorithms are

- Binning: To speed up processing, the image is downsampled to a smaller size.
- Minstep: This specifies the minimum displacement at which the stage will follow. E.g. if the animal moves 5 um, but the minstep = 10 um, the stage will not move. This is intended to reduce unnecessary  ‘jitter’ due to the center of mass of the animal moving. 
- Capture radius: If a radius is set, the system only considers objects in that zone but does not further crop the image. Useful when tracking in a dense population where other objects may appear in the field of view.
- Dark background: This enables switching between tracking in fluorescence (object bright, background dark) and brightfield (object dark, background light) modes.


Tracking modes
1. CMS: This detects the center of mass of an object in the image and centers that point.
2. Min/Max: This identifies the pixel with the lowest (highest) intensity and centers this one
3. Difference: This algorithm calculates the difference between the latest and previous images and detects the location of largest change.</li>


Problem: GUI quits when starting tracking or the tracking seems to make very large steps
Solution: Check that the calibration values (pixel size and rotation) are correct. 

Problem:  GUI quits in the middle of a recording
Solution: This can happen if disk space is full,( but further causes need to be documented).

Problem: Tracking does not properly follow the animal
Solution: There are multiple possible causes. Either, the object is not properly detected, the image analysis is too slow, the stage speed too low, …
To check the image analysis, record a few images with the camera settings you want to use (1-2 images are sufficient). Open the code in TestFunctions/testObjectDetection.py
Here you can run your acquired images through the image analysis functions that are used during tracking and identify why it fails. This code can also be used to optimize tracking parameters.

Problem: Motion blur
Solution: Reduce the exposure time of the camera and compensate by hardware binning (set using pylon features file .psf) or a larger gain.
