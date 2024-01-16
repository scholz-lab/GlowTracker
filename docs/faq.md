---
title: 6. FAQ
layout: default
nav_order: 7
---

# Frequently Asked Questions (FAQ)

## Q: The stage is stalling or slipping.
<p align="justify">
    Check that the movement is not obstructed by anything. 
    If the LED indicator on some of the stages turns blue, it means that it is in installed mode. 
    Try unplugging and plugging in the stage power source. 
    If the problem still persists it could be that the stage is under a too-heavy load, try lowering the stage's speed or maximum speed at <b>Settings</b> > <b>Stage</b> > <b>Stage speed</b>, <b>Stage max speed</b>, or changing the stage physical configuration.
    If all else fails, you might need to use a stronger version of the stage, such as <a href="https://www.zaber.com/products/linear-stages/X-LSM/specs?part=X-LSM150A"><i>LSM-150A</i></a>
    More stage's troubleshooting instructions can be found here <a href="https://www.zaber.com/manuals/X-LSM-E#m-9-troubleshooting-x-series-motion-devices"><i>Troubleshooting X-series motion devices</i></a>
</p>

## Q: The stage moves too fast or too slow.
<p align="justify">
    The stage's movement speed can be set at <b>Settings</b> > <b>Stage</b> > <b>Stage speed</b>, there you can set both the normal movement speed and the slow movement speed (used when moving while holding Shift key) as well as their units.
</p>

## Q: The stage moves too far or too short.
<p align="justify">
<ol>
    <li>
    First, make sure you have homed the stage the first time you are using it by going to <b>Settings</b> > <b>Stage</b> > <b>Home stage on startup</b><br/><br/>


        <figure class="center-figure">
            <img src="custom_assets/images/home_stage_setting.png" alt="Home stage on startup" width="80%">
        </figure><br/>
    
        then reconnect the stage by clicking the <b>Stage</b> button to turn red <img src="custom_assets\images\buttons\connection_off.png" class="inline-image"> and then click again to turn green <img src="custom_assets\images\buttons\connection_on.png" class="inline-image">, all the X, Y, and Z stages will the move to its zero position.
    </li>
    <li>
    After homing the stage, or if the stage was already homed, set the stage limit to your desired distance by going to <b>Settings</b> > <b>Stage</b> > <b>Stage Limit</b><br/><br/>

        <figure class="center-figure">
            <img src="custom_assets/images/stage limit.png" alt="Stage limit" width="80%">
        </figure><br/>

    the numbers are in X, Y, and Z order. Afterward, close the settings, and reconnect the stage again by clicking the <b>Stage</b> button off <img src="custom_assets\images\buttons\connection_off.png" class="inline-image"> and on <img src="custom_assets\images\buttons\connection_on.png" class="inline-image">.
    </li>
</ol>
</p>

<p align="justify">It is <i>crucial</i> that you set the stage Z limit such that your objective lens does not go too far down and collide with your plate or subject and break it (we have learned this the hard way).</p>


## Q: The stage moves unintuitively.
<p align="justify">
<ol>
    <li>
    Make sure to calibrate the camera and stage relationship by clicking the <b>Calibrate</b> button, then the <b>Camera & Stage Calibration</b> tab. This estimates the similarity transformation between the two. See the estimated result plot and decide if it's accurate to your setup or not. If not, try changing the object to something with a clearer structure in the field of view.
    </li>
    <li>
    After calibrating the transformation, turn on the move in image space mode' option in <b>Settings</b> > <b>Stage</b> > <b>Move in image space mode</b>.
    <figure class="center-figure">
        <img src="custom_assets/images/5 - calibration/camera and stage calibration/3.png" alt="Stage limit" width="80%">
    </figure>
    </li>
</ol>
</p>

## Q: The camera is completely black.
Here are a couple of things to try:
- Increase camera exposure
- Increase camera gain
- Lower up or raise the stage

## Q: The dual-color calibration is not perfectly aligned.
<p align="justify">
    Depending on your optics configuration, it can be quite hard to have a similar-structure image showing up on both sides of the dual-color image, which is important for the calibration algorithm. 
    Try finding a fluorescent object that shows patterns on both sides. 
    Sometimes we found that calibrating on the animal itself can also give a great result.
    But if all else fails or you want a perfect pixel alignment, you can manually adjust the translation and rotation parameters in <b>Settings</b> > <b>DualColor</b> > <b>Translation X</b>, <b>Translation Y</b>, and <b>Rotation</b>.
</p>

## Q: The tracking doesn't work.
TODO: Write this section after implementing the tracking diagnostic.

