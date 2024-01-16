# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from zaber_motion import Library
from zaber_motion.ascii import Connection
from zaber_motion import Units

Library.toggle_device_db_store(True)

with Connection.open_serial_port("com3") as connection:
    device_list = connection.detect_devices()
    print("Found {} devices.".format(len(device_list)))
    #for DeviceAddressConflictExeption use Zaber Console to assign distinct addresses for devices
    
    for device in device_list:
        print("Homing all axes of device with address {}.".format(device.device_address))
        device.all_axes.home()

    device1 = device_list[0]
    device2 = device_list[1]
    device3 = device_list[2]
    
    axis_x = device1.get_axis(1)
    axis_y = device2.get_axis(1)
    axis_z = device3.get_axis(1)

    axis_x.move_absolute(5, Units.LENGTH_MILLIMETRES, wait_until_idle=True)
    axis_y.move_absolute(7, Units.LENGTH_MILLIMETRES, wait_until_idle=False)
    axis_x.move_relative(2, Units.LENGTH_MILLIMETRES, wait_until_idle=True)
    axis_z.move_absolute(1, Units.LENGTH_CENTIMETRES, wait_until_idle=True)

    for device in device_list:
            device.all_axes.home()
