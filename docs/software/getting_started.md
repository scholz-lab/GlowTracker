---
title: 3.2 Getting Started
layout: default
parent: 3. Software
# nav_order: -1
---

# Getting Started

After you have finished constructing your macroscope, connect the camera and stage's USBs to your computer.
1. [Opening up the GUI](#opening-the-gui) 
2. [Connect to the stage.](#connect-to-the-stage)
3. [Connect to the camera.](#connect-to-the-camera)
4. [Calibrate the camera and stage relationship.](#calibrate-the-camera-and-stage-relationship) 
5. [Dual-color calibration (optional).](#dual-color-calibration) 
6. [Set recording file path.](#set-recording-file-path) 

## Opening up the GUI <a name="opening-the-gui"></a>
<p align="center">
  <img src="../custom_assets/images/gui_annotation.png" alt="macroscope" width="100%"/>
  <a name="gui-image"></a>
</p>

If you have never connected to the Zaber stage before, install and open [Zaber Launcher](https://software.zaber.com/zaber-launcher/download). 
In there you can find the connection port name to your stage, usually `COM3` for Windows or `/dev/ttyUSB0` for Linux, and you can also update your stage firmware to the latest version. 
Afterward, close the Zaber Launcher and open up the GlowTracker. 
Open the GUI setting (TODO: point to the button), put your connection port name in the section `Stage`, field `Stage serial port`. 
Close the setting file and click the connect to stage button (TODO: point to the button), if the button turns green then you have successfully connected the stage.

You can now control the stage movement by the stage control buttons ([GUI image](#gui-image): Group 2) or the arrow keys on your keyboard. The left-right  arrow 


## Connect to the stage <a name="connect-to-the-stage"></a>
In `macroscope.ini`, specify path to your pylon default camera setting `default_settings`. This is a `.pfs` file that can be obtain from the `pylon Viewver` software that you have downloaded.

## Connect to the camera <a name="connect-to-the-camera"></a>

## Calibrate the camera and stage relationship <a name="calibrate-the-camera-and-stage-relationship"></a>

## Dual-color calibration (optional) <a name="dual-color-calibration"></a>

## Set recording file path <a name="set-recording-file-path"></a>