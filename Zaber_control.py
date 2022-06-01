import asyncio
from zaber_motion import Library, Units, Measurement
from zaber_motion.ascii import Connection, AlertEvent, AllAxes

#library.toggle_device_db_store(True)


class Stage:
    def __init__(self, port, **kwargs):
        """
        Initialize a wrapper around the stage and set some limits and speeds.
        :param port: COM port to the stage
        """
        self.connection = None
        self.units = {'cm': Units.LENGTH_CENTIMETRES, 'mm': Units.LENGTH_MILLIMETRES, \
            'um': Units.LENGTH_MICROMETRES, 'mms': Units.VELOCITY_MILLIMETRES_PER_SECOND, 'ums': Units.VELOCITY_MICROMETRES_PER_SECOND}
        connection = self.connect_stage(port)
        print(connection)
        if connection is not None:
            self.connection = connection
            self.assign_axes()
            self.set_maxspeed(speed = 20)
            
            
    def connect_stage(self, port='COM3'):
        """
        Connects to the zaber stage and pass the connection including the axes
        :param port: COM port to the stage
        :return: connection to the stage including the axes. Close it before closing using the close method
        """
        try:
            connection = Connection.open_serial_port(port)
            device_list = connection.detect_devices()
            print("Found {} devices".format(len(device_list)))
            if len(device_list) > 0:
                return connection
            else:
                return None
        except Exception:
            print(Exception)
            return None


    def assign_axes(self):
        """
        Order the axis and name them as x,y,z where x is the closest to the computer.
        """
        self.connection.renumber_devices(first_address=1)
        device_list = self.connection.detect_devices()
        print("Found {} devices".format(len(device_list)))
        self.connection.axis_x = device_list[0].get_axis(1)
        self.connection.axis_y = device_list[1].get_axis(1)
        self.no_axes = 2
        if len(device_list) > 2:
            self.connection.axis_z = device_list[2].get_axis(1)
            self.no_axes = 3


    def set_maxspeed(self, speed = 20):
        if self.connection is not None:
            for ax in [self.connection.axis_x, self.connection.axis_y, self.connection.axis_z]:
                ax.settings.set("maxspeed", speed, Units.VELOCITY_MILLIMETRES_PER_SECOND)
                speed = self.connection.axis_z.settings.get("maxspeed", Units.VELOCITY_MILLIMETRES_PER_SECOND)
            print("Maximum speed [mm/s]:", speed)
            

    #  Stage homing
    def home_stage(self):
        '''
        homes all connected devices & moves axes to starting positions
        necessary if device was disconnected from power source
        '''
        if self.connection is not None:
            device_list = self.connection.detect_devices()
            for device in device_list:
                device.all_axes.home()#wait_until_idle =False)
        

    # Stage moving to a given absolute position 
    def move_abs(self, pos = (20,75, 130), unit = 'mm', wait_until_idle = False):
        """"Move to a given absolute location
        Parameters: pos (tuple): location. Length indicates which axis to move eg. (1) would only move one axis.
                    stepsize (float): can be positive or negative
                    units(str): has to be 'mm' or 'um'
        """
        if self.connection is not None:
            if pos[0] is not None:
                self.connection.axis_x.move_absolute(pos[0], self.units[unit], wait_until_idle)
            if len(pos) > 1 and pos[1] is not None :
                self.connection.axis_y.move_absolute(pos[1], self.units[unit], wait_until_idle)
            if len(pos) > 2 and self.no_axes > 2:
                if pos[2] is not None:
                    self.connection.axis_z.move_absolute(pos[2], self.units[unit], wait_until_idle)
            
    # move single axis
    def move_x(self, step, unit = 'um', wait_until_idle = False):
        """Move to a given relative location
        Parameters: 
                    step (tuple): can be positive or negative, position indicates which axis to move eg. (0,1,0) moves y axis only.
                    units(str): string units, commonly used
        """
        if self.connection is not None:
            self.connection.axis_x.move_relative(step, self.units[unit], wait_until_idle)
    
    # move single axis
    def move_y(self, step, unit = 'um', wait_until_idle = False):
        """Move to a given relative location
        Parameters: 
                    step (tuple): can be positive or negative, position indicates which axis to move eg. (0,1,0) moves y axis only.
                    units(str): string units, commonly used
        """
        if self.connection is not None:
            self.connection.axis_y.move_relative(step, self.units[unit], wait_until_idle)
    
    # move single axis
    def move_z(self, step, unit = 'um', wait_until_idle = False):
        """Move to a given relative location
        Parameters: 
                    step (tuple): can be positive or negative, position indicates which axis to move eg. (0,1,0) moves y axis only.
                    units(str): string units, commonly used
        """
        if self.connection is not None:
            self.connection.axis_z.move_relative(step, self.units[unit], wait_until_idle)

    
    # define generic movement function 
    def move_rel(self, step, unit = 'um', wait_until_idle = False):
        """Move to a given relative location
        Parameters: 
                    step (tuple): can be positive or negative, position indicates which axis to move eg. (0,1,0) moves y axis only.
                    units(str): string units, commonly used
        """
       
        if step[0] != 0:
            self.move_x(step[0], unit = unit, wait_until_idle=wait_until_idle)
        if len(step) > 1 and step[1] != 0:
            self.move_y(step[1], unit = unit, wait_until_idle=wait_until_idle)
        if len(step) > 2 and self.no_axes >2 and step[2] !=0:
            self.move_z(step[1], unit = unit, wait_until_idle=wait_until_idle)
        
    ###TODO check if we can do , wait_until_idle = False here too
    def move_speed(self, velocity, unit = 'ums', wait_until_idle = False):
        """Move to a given relative location
        Parameters: 
                    velocity (tuple, float): can be positive or negative, position indicates which axis to move eg. (0,1,0) moves y axis only.
                    units(obj): has to be zaber units eg.  Units.LENGTH_MICROMETRES
        """
        if self.connection is not None:
            self.connection.axis_x.move_velocity(velocity[0], self.units[unit])
            if len(velocity) > 1:
                self.connection.axis_y.move_velocity(velocity[1], self.units[unit])
            if len(velocity) > 2 and self.no_axes >2:
                self.connection.axis_z.move_velocity(velocity[2], self.units[unit])
   

    def get_position(self, unit = 'mm', fast = True):
        """return the current position of the stage for all axes."""
        if fast:
            pos = []
            loop = []
            if self.connection is not None:
                loop.append(self.connection.axis_x.get_position_async(self.units[unit]))
                loop.append(self.connection.axis_y.get_position_async(self.units[unit]))
                if self.no_axes >2:
                    loop.append(self.connection.axis_z.get_position_async(self.units[unit]))

            move_coroutine = asyncio.gather(*loop)

            loop = asyncio.get_event_loop()
            pos = loop.run_until_complete(move_coroutine)
            return pos
        else:
            if self.connection is not None:
                pos.append(self.connection.axis_x.get_position(self.units[unit]))
                pos.append(self.connection.axis_y.get_position(self.units[unit]))
                if self.no_axes >2:
                    pos.append(self.connection.axis_z.get_position(self.units[unit]))


    def stop(self):
        if self.connection is not None:
            self.connection.axis_x.stop(wait_until_idle = False)
            self.connection.axis_y.stop(wait_until_idle = False)
            self.connection.axis_z.stop(wait_until_idle = False)

    def set_rangelimits(self, limits = (160,160,155), unit = 'mm'):
        '''
        Sets limit for every device axis separately. necessary to avoid collision with other set-up elements.
        Parameters: limits (tuple) 
        '''
        # set axes limits in millimetres (max. value is ?)
        if self.connection is not None:
            self.connection.axis_x.settings.set('limit.max', limits[0], self.units[unit])
            self.connection.axis_y.settings.set('limit.max', limits[1], self.units[unit])
            self.connection.axis_z.settings.set('limit.max', limits[2], self.units[unit])


    def on_connect(self, home = True, start = (20,75, 130), limits =(160,160,155), coords = None):
        """startup routine to home, set range and move to start if desired. """
        if home:
            self.home_stage()
        self.set_rangelimits(limits)
        self.move_abs(start)
        
        device_list = self.connection.detect_devices()
        for device in device_list:
            device.all_axes.wait_until_idle(throw_error_on_fault = True)
        coords = self.get_position(fast = False)
        
    
    def disconnect(self):
        """close com port connection."""
        if self.connection is not None:
            self.connection.close()
