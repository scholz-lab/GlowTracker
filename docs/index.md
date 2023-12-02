---
title: 1 Getting Started
layout: default
nav_order: 1
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
  <img src="custom_assets/images/gui_annotation.png" alt="macroscope" width="100%"/>
  <a name="gui-image"></a>
</p>

## Connect to the stage <a name="connect-to-the-stage"></a>
If you have never connected to the Zaber stage before, install and open [Zaber Launcher](https://software.zaber.com/zaber-launcher/download). 
In there you can find the connection port name to your stage, usually `COM3` for Windows or `/dev/ttyUSB0` for Linux, and you can also update your stage firmware to the latest version. 
Afterward, close the Zaber Launcher and open up the GlowTracker. 
Open the **Settings** ![](custom_assets/images/buttons/ten.png){: .inline-image} , put your connection port name in the section **Stage**, field **Stage serial port**. 
Close the settings, then click the **Stage** ![](custom_assets/images/buttons/one.png){: .inline-image}. If the button turns green then you have successfully connected the stage.

You can now control the stage movement by the stage control buttons ([GUI image](#gui-image): Group 2) or the arrow keys on your keyboard. The left ⇦, right ⇨ keys control stage X axis, ⇧ ⇩ controls stage Y axis, and Page Up, Page Down, raises and lowers the stage. 
Press these in combination with the Shift key to move at a slower speed.

## Connect to the camera <a name="connect-to-the-camera"></a>
If you have never connected to the Basler camera before, install and open the [pylon Viewver](https://www.baslerweb.com/en/software/pylon/pylon-viewer/). In there you can test the connection to your camera, specifies your settings such as exposure time and gain, then export your camera setting file as **.pfs**. 
Open the **Settings** ![](custom_assets/images/buttons/ten.png){: .inline-image}, put your connection port name in the section **Camera**, field **Default camera settings**.
Close the settings, then click the **Camera** ![](custom_assets/images/buttons/one.png){: .inline-image}. If the button turns green then you have successfully connected the camera.
You can view from the camera by pressing the live view button.

## Calibrate the camera and stage relationship <a name="calibrate-the-camera-and-stage-relationship"></a>
Calibration is essential to translate motion of the object of interest in the image to the compensatory motion of the stage, that centers the object.

1. Select a sample that shows a lot of structure, for example a ruler, or a drop of fluorescent pigment on a slide.
    <table class="equal-column-table">
      <tr>
        <td>
          <figure class="center-figure">
            <img src="custom_assets/images/5 - calibration/camera and stage calibration/1 - fluorescent drop on slide.jpg" alt="Fluorescent drop">
            <figcaption>Fluorescent drop on a slide</figcaption>
          </figure>
        </td>
        <td>
          <figure class="center-figure">
            <img src="custom_assets/images/5 - calibration/camera and stage calibration/1 - viewing on macroscope.jpg" alt="Viewing fluorescent drop">
            <figcaption>Placing the slide on the center of objective</figcaption>
          </figure>
        </td>
      </tr>
    </table>
2. Open the **Calibrate** window ![](custom_assets/images/buttons/ten.png){: .inline-image}. Navigate to the tab **Camera & Stage Calibration**.

3. Click **Calibrate**. The resulting pixel size and rotation of the camera will be shown, along with a plot display the camera space and stage space.
    <figure class="center-figure">
      <img src="custom_assets/images/5 - calibration/camera and stage calibration/2.png" alt="Calibration result" width="70%">
      <figcaption>Camera and Stage calibration result.</figcaption>
    </figure>

4. You can test if the calibration is correct by turning on the **Move in image space mode** option in the **Settings** ![](custom_assets/images/buttons/ten.png){: .inline-image} under **Stage** section. After turning it on, when you move the stage using the arrow keys ⇦⇨⇧⇩, the resulting image should also moved accordingly and intuitively, e.g. moving up ⇧ should move the image up, and moving to the right ⇨ should also move image to the right.

    <figure class="center-figure">
      <img src="custom_assets/images/5 - calibration/camera and stage calibration/3.png" alt="Calibration result" width="70%">
      <figcaption>Turn on the Move in image space mode setting</figcaption>
    </figure>
   
## Dual-color calibration (optional) <a name="dual-color-calibration"></a>
(this step is only required in the dual-color configuration.)
To calibrate the relationship between the two color channels, which will later allow an accurate overlay of the two images, the image-splitter needs to be calibrated. 
1. Select a sample that shows either the same structure in both chanels (e.g., fluorescent beads, fluorescent tape) or that has sufficient bleed-through to appear in both channels.
    <table class="equal-column-table">
      <tr>
        <td>
          <figure class="center-figure">
            <img src="custom_assets/images/5 - calibration/dual color calibration/1 - fluorescent strip.jpg" alt="Fluorescent stripes">
            <figcaption>Fluorescent stripes on a slide</figcaption>
          </figure>
        </td>
        <td>
          <figure class="center-figure">
            <img src="custom_assets/images/5 - calibration/dual color calibration/2 - viewing strip.jpg" alt="View fluorescent stripes">
            <figcaption>Placing the slide on the center of objective</figcaption>
          </figure>
        </td>
      </tr>
    </table>

2. Open the calibration dialog by cliking on **Calibrate** ![](custom_assets/images/buttons/ten.png){: .inline-image}. Navigate the tab to the section called **Dual Color Calibration**.

3. Click **Calibrate**. The calibration result will be shown in the overlay.
    <figure class="center-figure">
      <img src="custom_assets/images/5 - calibration/dual color calibration/3 - dual color calibration.png" alt="Calibration result" width= "70%">
      <figcaption>Dual-color calibration result</figcaption>
    </figure>
   
## Set recording file path <a name="set-recording-file-path"></a>
To pick where your image files and recording data will be stored, select a location using the file dialog in ![](custom_assets/images/buttons/six.png){: .inline-image}. 
