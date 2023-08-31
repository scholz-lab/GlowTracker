from zaber_motion import Library, Units
from zaber_motion.ascii import Connection, Axis

import numpy as np
import time

if __name__ == '__main__':

    # Initialize the library
    Library.enable_device_db_store()
    library = Library()

    # get the device from the zaber_motion.ascii.Connection
    # X-LSM150A
    connection = Connection.open_serial_port('COM3')
    device_list = connection.detect_devices()
    device_1 = device_list[0]
    axis_x = device_1.get_axis(1)

    # Configure the scope
    scope = device_1.oscilloscope
    #   Clear old, previous scope channels
    scope.clear()
    #   Add channels
    scope.add_channel(1, "pos")
    scope.add_channel(1, "vel")   
    scope.add_channel(1, "accel")   # maximum acceleration
    
    recording_interval = 1.0    # Recording interval in milliseconds unit
    scope.set_timebase(recording_interval)     

    # Start capturing data. 
    scope.start()

    # Sleep for 100ms
    sleep_time = 100e-3
    time.sleep(sleep_time)

    # Move
    axis_x.move_relative(10, Units.LENGTH_MILLIMETRES, wait_until_idle= True)

    # Sleep for 100ms
    time.sleep(sleep_time)

    # Move
    axis_x.move_relative(-10, Units.LENGTH_MILLIMETRES, wait_until_idle= True)

    # Sleep for 100ms
    time.sleep(sleep_time)

    # Stop capturing data
    scope.stop()

    # Read data
    channels = scope.read()
    channel_1_data = channels[0].get_data(Units.LENGTH_MILLIMETRES)
    channel_2_data = channels[1].get_data(Units.VELOCITY_MILLIMETRES_PER_SECOND)
    channel_3_data = channels[2].get_data(Units.ACCELERATION_MILLIMETRES_PER_SECOND_SQUARED)
    connection.close()
    
    # Save data

    #   Generate time stamp array
    n = len(channel_1_data)
    timestamp_data = [ i * recording_interval for i in range(1, n+1) ]

    # Combine the data into a structured array
    dtype = np.dtype([
        ('timestamp_ms', np.float32), 
        ('pos', np.float32), 
        ('vel', np.float32), 
        ('accel', np.float32)
    ])
    combined_data = np.array(list(zip(
            timestamp_data, 
            channel_1_data, 
            channel_2_data, 
            channel_3_data
        )), 
        dtype= dtype
    )

    # Save the structured array to a CSV file
    header = ','.join(combined_data.dtype.names)
    np.savetxt('zaber_stage_log.csv', combined_data, delimiter=',', header=header, comments='')
