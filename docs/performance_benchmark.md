---
title: 4.3 Performance benchmark
layout: default
parent: 4. Software
nav_order: 3
---
# Performance benchmark
<p align="justify">
    The performance of GlowTracker can be separated into two parts: <b>Image Acquisition</b>, and <b>Tracking</b>.
    Image acquisition is the process of acquiring images from the camera, until receiving a ready-to-use image in the host machine. 
    Tracking is the process of computing the key point position in the image and moving the stage to the appropriate location.
    The interactive widget below depicts the overall process and timeline, from image acquisition to tracking.
</p>

<p align="justify">
    To determine the performance of the system, we are interested in two things: how fast we can acquire images (Hz), and how fast we can track the object in the image (Hz). 
    Furthermore, we will calculate the ratio between the two to intuitively understand how much data is used for tracking versus skipped.
</p>


<div>
    <meta name=viewport content="width=device-width,initial-scale=1">  
    <meta charset="utf-8"/>
    <script src="https://www.geogebra.org/apps/deployggb.js"></script>
    <div id="ggb-element" style="height: 500px; width: 100%;"></div>
    <script type="text/javascript">

        var containerRect = document.getElementById('ggb-element').getBoundingClientRect();
        
        var params = {
            appName: "geometry", 
            material_id: "db34hnfh",
            autoHeight: true,
            width: containerRect.width,
            height: containerRect.height,
            showToolBar: false, 
            showMenuBar: false,
            showAlgebraInput: false, 
            showToolBarHelp: false,
            showResetIcon: true,
            errorDialogsActive: true,
            useBrowserForJS: false,
        };

        var ggbApplet = new GGBApplet(params, true);
        
        window.addEventListener("load", function() { 
            ggbApplet.inject('ggb-element');
        });
    </script>
</div>

## Image acquisition
<p align="justify">
    In an acquisition loop, the camera sensor receives a frame-start trigger signal, the sensor is exposed for a specified exposure time, then the sensor values are read out, internally processed in the camera, and sent to the host. After the host receives the image, it goes through an image processing pipeline one more time, then it is ready to be used in the application.
</p>

<p align="justify">
    Many factors decide the acquisition rate. 
    The most important factors that the user can control are exposure time, image size (or region-of-interest ROI), and binning mode. 
    The exposure time determines how long the sensor is exposed to light before being read out. 
    The shorter the exposure time, the higher the acquisition rate. 
    The lower the image size, the higher the acquisition rate.
    The binning mode, for example, if set to an additive mode, can increase the image brightness to compensate for the lower exposure time, but also reduce the effective image resolution.
    Deciding these factors depends on the equipment setup and the animal that is going to be studied.
    For more information on what are the parameters that affect the image acquisition rate, please visit <a href="https://docs.baslerweb.com/resulting-frame-rate"><i>Basler: Resulting frame rate</i></a>.
</p>

<p align="justify">
    With all these affecting parameters in mind, the total time from beginning to receiving a frame-start trigger signal to having a useable image in the application is called a <i>one-frame time</i>, and it is noticeably longer than just the exposure time because it also contains the sensor readout time and image processing time.
    Fortunately, we can operate the camera in a <a href="https://docs.baslerweb.com/electronic-shutter-types#rolling-shutter"><i>rolling shutter</i></a> mode, which exposes each row of the sensor consecutively with a small time offset (8 µs in our model) and also simultaneously read the row value out after it is finished. 
    This significantly reduces the waiting time for the sensor readout and effectively increases the acquisition rate to almost equal the exposure time plus some constants.

    The category of time that we will be using to benchmark is the effective image-receiving time, which is the timestamp at a point where the image is finished processing and is ready to be used in the application. The duration between each timestamp is essentially the <b>image acquisition rate</b>.
</p>


## Tracking
<p align="justify">
    Once an image is received, the application calculates the location of interest and instructs the stage to move accordingly until the location of interest is at the center of the image. The application then waits for a new image to begin tracking once again.
</p>

<p align="justify">
    The tracking algorithm is explained in <a href="{% link software/tracking_explanation.md %}#tracking"><i>Code explanation: Tracking</i></a>, and the amount of time to compute is denoted as <i>Compute Tracking</i> in the timeline widget above.
    The time it takes to communicate to the stage and then wait until it is moved to the specified location is called <i>Communicate to Stage</i> and <i>Stage Moving</i> respectively. 
    Additionally, we will have to wait for the camera to begin a new acquisition cycle.
    This is because if we were to use the latest image that we have in the application, the image could be exposed during the stage movement, resulting in a motion blur. The position of the object may also be different, resulting in inaccurate tracking.
    This amount of time depends on the tracking object, the camera, and the stage.
    If the tracking objects move relatively fast in each frame, then the stage moving time increases.
    If the image acquisition time is fast, then the waiting time for an acquisition cycle decreases.
    If the stage movement speed profile is fast (depending on the hardware <a href="https://www.zaber.com/protocol-manual?protocol=ASCII#topic_setting_motion_accel_ramptime"><i>configuration</i></a>), then the waiting time for the stage to finish moving decreases.
</p>

<p align="justify">
    The category of time that we will be using to benchmark is the effective image-tracking time, which is the time stamp at starting tracking of an image. The duration between each timestamp is the <b>tracking rate</b>.
</p>

## Benchmark
<p align="justify">
    We benchmarked image acquisition and tracking rates using varying exposure times. By definition, the image acquisition rate is mainly an inverse of the exposure time with some constant factor. 
    We would like to know how fast our application can track with regard to image acquisition rate. 
    The benchmarking is performed with maximum image ROI (3088 x 2064 pixels), no binning, in a laptop with 12th Gen Intel(R) Core(TM) i7-1255U 1.70 GHz CPU, 16 GB of RAM, and on a Windows 10 64-bit operating system. The results are shown in the plot below.
</p>

<figure class="center-figure">
    <img src="custom_assets/images/performance/image_acquisition_vs_tracking_rate.png" alt="image acquisition vs tracking rate">
</figure>

<p align="justify">
    Here we can observe that the relationship between the image acquisition rate and tracking rate is almost linearly proportional to each other. 
    This makes sense as the faster we can acquire images, the faster we can track them. 
    However, we notice that there is a spike in the tracking rate where the image acquisition rate is equal to 33.333 Hz, which is a result of setting the exposure time to 30 ms. 
    We have yet to find a definitive answer to this. 
    Our intuition for now is that there may be some acquisition rate that happens to synchronize perfectly with the tracking loop, which helps reduce the synchronization waiting time.
</p>

<p align="justify">
    Additionally, we would like to have an intuitive understanding of how fast the application can track compared to how fast it can acquire images. 
    We can do this by dividing the image acquisition rate by the tracking rate. 
    This gives us the ratio between the two called <b>Frames per track</b>, depicting how many frames have been acquired for the tracking of a frame to be completed. 
    We then plot this ratio against the image acquisition rate in the plot below.
</p>

<figure class="center-figure">
    <img src="custom_assets/images/performance/image_acquisition_vs_frames_per_track.png" alt="image_acquisition vs frames per track">
</figure>

<p align="justify">
    Based on the plot above, a similar behavior to the previous plot can be observed. 
    The frames-per-track ratio is mostly linearly proportionate to the image acquisition rate. 
    This is because the compute-tracking time and communicate-to-stage time are independent of the image acquisition rate. 
    Meaning no matter how fast we acquire images, we still have to wait for the same fixed amount of time to compute tracking, and so the frames-per-track ratio increases. 
    The peak of tracking rate at 33.33 Hz image acquisition in the previous figure also now becomes a dip in this figure because it means that it has a higher tracking rate than its neighbors. 
    The frames-per-track ratio has a lower bound of 2. 
    This is because if the tracking scheme is so fast that it can be completed within an image acquisition cycle, it would still need to wait for a new acquisition cycle to receive a correct image to use for the next tracking.
</p>

## Conclusion
<p align="justify">
    The performance of the application depends on many variables, the hardware setup, the software, and the studied subject.
    In our experience working with C. elegans and P. pacificus, using the described setup, we found that the range of exposure time to give a good quality image while still being relatively responsive is ranging from 20 ms to 60 ms, which yield the respective acquisition rates from 50 Hz to 16.67 Hz, and tracking rate from 11 Hz to 6.5 Hz, or the frames-per-track ratio of 5 to 2.5 times.
</p>
