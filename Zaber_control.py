from zaber_motion import library, units
from zaber_motion.ascii import connection

#library.toggle_device_db_store(True)


def connect_stage(port='COM3'):
    """
    Connects to the zaber stage and pass the connection including the axes
    :param port: COM port to the stage
    :return: connection to the stage including the axes. Close it before closing using the close method
    """
    my_connection = connection.open_serial_port(port)
    my_connection.renumber_devices(first_address=1)
    device_list = my_connection.detect_devices()
    print("Found {} devices".format(len(device_list)))
    device1 = device_list[0]
    device2 = device_list[1]
    my_connection.axis_x = device1.get_axis(1)
    my_connection.axis_y = device2.get_axis(1)
    if len(device_list) > 2:
        device3 = device_list[2]
        my_connection.axis_z = device3.get_axis(1)
    return my_connection


# %% Stage homing & moving to starting position
def home_stage():
    '''
    homes all connected devices & moves axes to starting positions
    necessary if device was disconnected from power source
    '''
    with connection.open_serial_port('COM3') as connection:
        # set address 1 to the device that is nearest to the computer. From there the others are consecutive integers
        connection.renumber_devices(first_address=1)
        device_list = connection.detect_devices()

        for device in device_list:
            device.all_axes.home()

        device1 = device_list[0]
        device2 = device_list[1]
        axis_x = device1.get_axis(1)
        axis_y = device2.get_axis(1)
        if len(device_list) > 2:
            device3 = device_list[2]
            axis_z = device3.get_axis(1)

        print('Moving axis_x to starting position...')
        axis_x.move_absolute(20, units.LENGTH_MILLIMETRES, wait_until_idle=False)
        print('Moving axis_y to starting position...')
        axis_y.move_absolute(75, units.LENGTH_MILLIMETRES, wait_until_idle=True)
        print('Moving axis_z to starting position...')
        axis_z.move_absolute(130, units.LENGTH_MILLIMETRES, wait_until_idle=True)


# %% Range limits for axes
def set_rangelimits():
    '''
    sets limit for every device axis separately
    necessary to avoid collision with other set-up elements
    '''
    with connection.open_serial_port('COM3') as connection:
        connection.renumber_devices(first_address=1)
        device_list = connection.detect_devices()
        # assign connected devices to IDs & axes variables
        device1 = device_list[0]
        device2 = device_list[1]
        axis_x = device1.get_axis(1)
        axis_y = device2.get_axis(1)
        if len(device_list) > 2:
            device3 = device_list[2]
            axis_z = device3.get_axis(1)

        # set axes limits in millimetres (max. value is )
        axis_x.settings.set('limit.max', 160, units.LENGTH_MILLIMETRES)
        axis_y.settings.set('limit.max', 160, units.LENGTH_MILLIMETRES)
        axis_z.settings.set('limit.max', 155, units.LENGTH_MILLIMETRES)
