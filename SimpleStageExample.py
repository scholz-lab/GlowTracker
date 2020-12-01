from zaber_motion import Library, Units
from zaber_motion.ascii import Connection

Library.toggle_device_db_store(True)


# %% Stage homing & moving to starting position
def home_stage():
    '''
    homes all connected devices & moves axes to starting positions
    necessary if device was disconnected from power source
    '''

    #
# with Connection.open_serial_port('COM3') as connection:
#     device_list = connection.detect_devices()
#     # set address 1 to the device that is nearest to the computer. From there the others are consecutive integers
#     connection.renumber_devices(first_address=1)
#
#     for device in device_list:
#         device.all_axes.home()
#
#     device1 = device_list[0]
#     device2 = device_list[1]
#     axis_x = device1.get_axis(1)
#     axis_y = device2.get_axis(1)
#     if len(device_list) > 2:
#         device3 = device_list[2]
#         axis_z = device3.get_axis(1)
#
#     print('Moving axis_x to starting position...')
#     axis_x.move_absolute(20, Units.LENGTH_MILLIMETRES, wait_until_idle=False)
    #     print('Moving axis_y to starting position...')
    #     axis_y.move_absolute(75, Units.LENGTH_MILLIMETRES, wait_until_idle=True)
    #     print('Moving axis_z to starting position...')
    #     axis_z.move_absolute(130, Units.LENGTH_MILLIMETRES, wait_until_idle=True)


MyConnection = Connection.open_serial_port('COM3')
MyConnection.renumber_devices(first_address=1)
device_list = MyConnection.detect_devices()
print("Found {} devices".format(len(device_list)))
device1 = device_list[0]
device2 = device_list[1]
axis_x = device1.get_axis(1)
axis_y = device2.get_axis(1)
if len(device_list) > 2:
    device3 = device_list[2]
    axis_z = device3.get_axis(1)
print('Moving axis_x to starting position...')
axis_x.move_absolute(0, Units.LENGTH_MILLIMETRES, wait_until_idle=False)
MyConnection.close()