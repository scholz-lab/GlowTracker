from zaber_motion import Library
from zaber_motion.ascii import Connection

Library.toggle_device_db_store(True)

with Connection.open_serial_port("com3") as connection:
    device_list = connection.detect_devices()
    print("Found {} devices.".format(len(device_list)))
    
    for device in device_list:
        print("Homing all axes of device with address {}.".format(device.device_address))
        device.all_axes.home()
