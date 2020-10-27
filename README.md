# RaspberryPi
documentation and scripts for raspberry pi


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
       
- start programming for your device:

       from zaber_motion import Library           #retrieves information about zaber devices from internet
       from zaber_motion.ascii import Connection

       Library.toggle_device_db_store(True)       #stores information for offline use
    
       with Connection.open_serial_port("COM5") as connection:    #opens port, check for correct port in "ports (COM & LPT)" in your device manager and adjust in code
        device_list = connection.detect_devices()
        print("Found {} devices".format(len(device_list)))

        # The rest of your program goes here (indented)
        
- It is advisable to include lines homing your device at the beginning of your code to find its reference position in order to perform accurate absolute position movements!      
- For more information and how-to guides go to the Zaber Motion Library: https://www.zaber.com/software/docs/motion-library/ascii/
