# Macroscope
Documentation and scripts to control the behavior macroscope


To get the BASLER package running on windows 64 do the following:

 - install pylon (software from BASLER) including all interfaces
 - Get the pypylon package from GitHub: https://github.com/basler/pypylon
 - use pip to install the package using the proper wheel (in my case pypylon-1.5.4-cp37-cp37m-win_amd64.whl)
 - run this using the comandline within the package folder
 - pip install pypylon-1.5.4-cp37-cp37m-win_amd64.whl

To get the NI DAQ card working:

 - Install the NI device drivers: www.ni.com/de-de/support/downloads/drivers/download.ni-device-drivers.html#327643
 - Install the nidaqmx python package using pip: https://github.com/ni/nidaqmx-python 
 

To get controls for zaber stage running:

- install zaber motion package by running one of the following lines (python version must be 3.5 or higher to use Zaber library):
     
       conda install -c conda-forge zaber-motion
       conda install -c conda-forge/label/cf202003 zaber-motion
       
- See SimpleStageExample.py for two alternative approaches to begin the communication with the satge.
- The SDK to control the stage has two different levels of code. One is the ASCII lib, which is used to communicate to the stage on a lower level:
https://www.zaber.com/support/docs/api/core-python/0.8.1/ascii.html#
Then we have the API written for python, which uses the ASCII lib under the hood to execute the commands:
https://www.zaber.com/software/docs/motion-library/ascii/references/python/#axis
        
- It is advisable to include lines homing your device at the beginning of your code to find its reference position to perform meaningful absolute position movements!      

