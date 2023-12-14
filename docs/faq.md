---
title: 6. FAQ
layout: default
nav_order: 7
---

# Frequently Asked Questions

## Q: The stage is stalling or slipping.
Check that the movement is not obstructed by anything. 
If the LED indicator on some of the stages turns blue, it means that it is in installed mode. 
Try unplugging and plugging in the stage power source. 
If the problem still persists it could be that the stage is under a too heavy load, try lowering the stage's speed or maximum speed at **Settings** > **Stage** > **Stage speed**, **Stage max speed**, or changing the stage physical configuration.
If all else fails, you might need to use a stronger version of the stage, such as [LSM-150A](https://www.zaber.com/products/linear-stages/X-LSM/specs?part=X-LSM150A)
More stage's troubleshooting instructions can be found here [Troubleshooting X-series motion devices](https://www.zaber.com/manuals/X-LSM-E#m-9-troubleshooting-x-series-motion-devices)

## Q: The stage moves too fast or too slow.
The stage's movement speed can be set at **Settings** > **Stage** > **Stage speed**, there you can set both the normal movement speed and the slow movement speed (used when moving while holding Shift key) as well as their units.

## Q: The stage moves too far or too short.
1. First, make sure you have homed the stage the first time you are using it by going to **Settings** > **Stage** > **Home stage on startup**

    <figure class="center-figure">
        <img src="custom_assets/images/home_stage_setting.png" alt="Home stage on startup" width="80%">
    </figure>
then reconnect the stage by clicking the **Stage** button to turn red ![](custom_assets\images\buttons\connection_off.png){: .inline-image} and then click again to turn green ![](custom_assets\images\buttons\connection_on.png){: .inline-image}, all the X, Y, and Z stages will the move to its zero position.

2. After homing the stage, or if the stage was already homed, set the stage limit to your desired distance by going to **Settings** > **Stage** > **Stage Limit**

    <figure class="center-figure">
        <img src="custom_assets/images/stage limit.png" alt="Stage limit" width="80%">
    </figure>
the numbers are in X, Y, and Z order. Afterward, close the settings, and reconnect the stage again by clicking the **Stage** button off ![](custom_assets\images\buttons\connection_off.png){: .inline-image} and on ![](custom_assets\images\buttons\connection_on.png){: .inline-image}.

It is *very important* that you set the stage Z limit such that your objective lens does not go too far down and collide with your plate or subject and break it (we have learned this the hard way).


## Q: The stage moves unintuitively.
1. Make sure to calibrate the camera and stage relationship by clicking the **Calibrate** button, then the **Camera & Stage Calibration** tab. This estimates the similarity transformation between the two. See the estimated result plot and decide if it's accurate to your setup or not. If not, try changing the object to something with a clearer structure in the field of view.

2. After calibrating the transformation, turn on the move in image space mode' option in **Settings** > **Stage** > **Move in image space mode**.
    <figure class="center-figure">
        <img src="custom_assets/images/5 - calibration/camera and stage calibration/3.png" alt="Stage limit" width="80%">
    </figure>


## Q: The camera is completely black.
Here are a couple of things to try:
- Increase camera exposure
- Increase camera gain
- Lower up or raise the stage

## Q: The dual-color calibration is not perfectly aligned.
Depending on your optics configuration, it can be quite hard to have a similar-structure image showing up on both sides of the dual-color image, which is important for the calibration algorithm. 
Try finding a fluorescent object that shows patterns on both sides. 
Sometimes we found that calibrating on the animal itself can also give a great result.
But if all else fails or you want a perfect pixel alignment, you can manually adjust the translation and rotation parameters in **Settings** > **DualColor** > **Translation X**, **Translation Y**, and **Rotation**.

## Q: The tracking doesn't work.
TODO: Write this section after implementing the tracking diagnostic.

