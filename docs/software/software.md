---
title: 3. Software
layout: default
has_children: true
nav_order: 4
---

# Control software
The macroscope GUI is intended to provide a tracking interface for manual and automated tracking. Limited access to camera parameters is provided, but we assume users will use the pylon software from Basler to adjust and save complex configurations as we do not wish to duplicate this work. The GUI is build as a Kivy app which connects to the macroscope hardware (currently two USB devices). GUI functionality is implemented mostly in the Kivy file, whereas device functionality is relayed to specific modules.

<p align="center">
    <img src="../custom_assets/images/gui_annotation.png" width="100%">
</p>
---
{: .no_toc }

