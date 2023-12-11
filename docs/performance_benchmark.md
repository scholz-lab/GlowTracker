---
title: Performance benchmark
layout: default
---
# Performance benchmark
The performance of GlowTracker can be separated into two parts: **Image Acquisition**, and **Tracking**.
Image acquisition is the process of acquiring images from the camera, until receiving a ready-to-use image in the host machine. 
Tracking is the process of computing the key point position in the image and moving the stage to the according location.
The interactive widget below depicts the overall process and timeline, from image acquisition to tracking.

In order to determine the performance of the system, we are now interested in two things: how fast can we acquire images (Hz), and how fast can we track the object in the image (Hz). Additionally, we will also look at the ratio between the two, which will give an intuitive understanding of how much is the data is being used for tracking or skipped.

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
In an acquisition loop, the camera sensor receives a frame start trigger signal, the sensor is exposed for a period of time, then the sensor values are read out, internally processed in the camera, and sent to the host. After the host receives the image, it goes through an image processing pipeline one last time, and now is ready to be used in the application.
<!-- https://docs.baslerweb.com/resulting-frame-rate -->
<!-- https://www.baslerweb.com/en/tools/frame-rate-calculator/ -->