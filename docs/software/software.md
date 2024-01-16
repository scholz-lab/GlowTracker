---
title: 4. Software
layout: default
has_children: true
nav_order: 5
---

# Control software
<p align="justify">
    The purpose of the macroscope GUI is to provide a tracking interface for both manual and automated tracking. Although there is limited access to camera parameters, users are expected to use the pylon software from BASLER to adjust and save complex configurations as we do not want to duplicate this work. The GUI is a Kivy app that connects to the macroscope hardware, which currently consists of two USB devices. Most of the GUI functionality is implemented in the Kivy file, while device functionality is relayed to specific modules.
</p>

<p align="center">
    <img src="../custom_assets/images/gui_annotation.png" width="100%">
</p>
---
{: .no_toc }

