from zaber_motion import Library
from zaber_motion.ascii import Connection
from zaber_motion import Units

Library.toggle_device_db_store(True)

with Connection.open_serial_port("com3") as connection:
    device_list = connection.detect_devices()
    print("Found {} devices.".format(len(device_list)))

    device1 = device_list[0]
    device2 = device_list[1]
    axis_x = device1.get_axis(1)
    axis_y = device2.get_axis(1)
    if len(device_list) > 2:
        device3 = device_list[2]
        axis_z = device3.get_axis(1)
    
    # axis_x.settings.set('limit.max', 56, Units.LENGTH_MILLIMETRES)
    # axis_y.settings.set('limit.max', 81.5, Units.LENGTH_MILLIMETRES)
    axis_z.settings.set('limit.max', 130, Units.LENGTH_MILLIMETRES)
