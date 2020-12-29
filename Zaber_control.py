from zaber_motion import Library, Units
from zaber_motion.ascii import Connection

#library.toggle_device_db_store(True)


def connect_stage(port='COM3'):
    """
    Connects to the zaber stage and pass the connection including the axes
    :param port: COM port to the stage
    :return: connection to the stage including the axes. Close it before closing using the close method
    """
    try:
        my_connection = Connection.open_serial_port(port)
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
            # set speed
        for ax in [my_connection.axis_x,my_connection.axis_y,my_connection.axis_z]:
            ax.settings.set("maxspeed", 20, Units.VELOCITY_MILLIMETRES_PER_SECOND)
            speed = my_connection.axis_z.settings.get("maxspeed", Units.VELOCITY_MILLIMETRES_PER_SECOND)
        print("Maximum speed [mm/s]:", speed)

        return my_connection
        

    except Exception:
        print(Exception)
        return None


# %% Stage homing & moving to starting position
def home_stage(connection):
    '''
    homes all connected devices & moves axes to starting positions
    necessary if device was disconnected from power source
    '''
   
    # set address 1 to the device that is nearest to the computer. From there the others are consecutive integers
    device_list = connection.detect_devices()

    for device in device_list:
        device.all_axes.home()
    
    move_to_start(connection, start = (20,75, 130))

# %% Stage homing & moving to starting position
def move_to_start(connection, start = (20,75, 130)):
    print('Moving axis_x to starting position...')
    connection.axis_x.move_absolute(start[0], Units.LENGTH_MILLIMETRES, wait_until_idle=False)
    print('Moving axis_y to starting position...')
    connection.axis_y.move_absolute(start[1], Units.LENGTH_MILLIMETRES, wait_until_idle=False)
    print('Moving axis_z to starting position...')
    connection.axis_z.move_absolute(start[2], Units.LENGTH_MILLIMETRES, wait_until_idle=False)


# %% Range limits for axes
def set_rangelimits(connection, limits = (160,160,155)):
    '''
    sets limit for every device axis separately
    necessary to avoid collision with other set-up elements
    '''
    # set axes limits in millimetres (max. value is )
    connection.axis_x.settings.set('limit.max', limits[0], Units.LENGTH_MILLIMETRES)
    connection.axis_y.settings.set('limit.max', limits[1], Units.LENGTH_MILLIMETRES)
    connection.axis_z.settings.set('limit.max', limits[2], Units.LENGTH_MILLIMETRES)
