---
title: 4.3 Known issues
layout: default
parent: 4. Software
# nav_order: -1
---

# Known issues

-  While the exposed camera parameters can be changed during live preview, the framerate is fixed to a display framerate. This might be undesired behavior.

To get controls for the zaber stage running:

- Install zaber motion package by running one of the following lines (python version must be 3.5 or higher to use Zaber library):
     
       conda install -c conda-forge zaber-motion
       conda install -c conda-forge/label/cf202003 zaber-motion
       
- See `SimpleStageExample.py` for two alternative approaches to begin the communication with the stage.
- The SDK to control the stage has two different levels of code. One is the ASCII lib, which is used to communicate to the stage on a lower level, which you can find <a href="https://www.zaber.com/support/docs/api/core-python/0.8.1/ascii.html#">here</a>. Then we have the API written for Python, which uses the ASCII lib under the hood to execute the commands (can be found <a href="https://www.zaber.com/software/docs/motion-library/ascii/references/python/#axis">here</a>)

  
- It is advisable to include lines homing your device at the beginning of your code to find its reference position to perform meaningful absolute position movements!      

- On Linux systems, accessing serial ports needs to be allowed for the user running the GUI. In Ubuntu and similar systems the user has to be added to the group *"dialout"*.
