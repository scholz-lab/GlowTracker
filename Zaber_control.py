from zaber_motion import Library, Units
from zaber_motion.ascii import Connection

#library.toggle_device_db_store(True)


class Stage:
    def __init__(self, port, **kwargs):
        """
        Initialize a wrapper around the stage and set some limits and speeds.
        :param port: COM port to the stage
        """
        self.connection = self.connect_stage(port)
        self.units = {'cm': Units.LENGTH_CENTIMETRES, 'mm': Units.LENGTH_MILLIMETRES, 'um': Units.LENGTH_MICROMETRES}
        if self.connection is not None:
            self.assign_axes()
            self.set_maxspeed(speed = 20)

    def connect_stage(self, port='COM3'):
        """
        Connects to the zaber stage and pass the connection including the axes
        :param port: COM port to the stage
        :return: connection to the stage including the axes. Close it before closing using the close method
        """
        try:
            return Connection.open_serial_port(port)
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
        if len(device_list) > 2:
            self.connection.axis_z = device_list[2].get_axis(1)



    def set_maxspeed(self, speed = 20):
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
        device_list = self.connection.detect_devices()

        for device in device_list:
            device.all_axes.home()#wait_until_idle =False)
    

    # Stage moving to a given absolute position 
    def move_abs(self, pos = (20,75, 130), unit = 'mm'):
        """"Move to a given absolute location
        Parameters: pos (tuple): location. Length indicates which axis to move eg. (1) would only move one axis.
                    stepsize (float): can be positive or negative
                    units(obj): has to be 'mm' or 'um'
        """
        self.connection.axis_x.move_absolute(pos[0], self.units[unit], wait_until_idle=False)
        if len(pos) > 1:
            self.connection.axis_y.move_absolute(pos[1], self.units[unit], wait_until_idle=False)
        if len(pos) > 2:
            self.connection.axis_z.move_absolute(pos[2], self.units[unit], wait_until_idle=False)
        

    # define generic movement function 
    def move_rel(self, step, unit = 'um'):
        """Move to a given relative location
        Parameters: 
                    step (tuple): can be positive or negative, position indicates which axis to move eg. (0,1,0) moves y axis only.
                    units(obj): has to be zaber units eg.  Units.LENGTH_MICROMETRES
        """
        self.connection.axis_x.move_relative(step[0], self.units[unit], wait_until_idle=False)
        self.connection.axis_y.move_relative(step[1], self.units[unit], wait_until_idle=False)
        self.connection.axis_z.move_relative(step[2], self.units[unit], wait_until_idle=False)
    
        
    def set_rangelimits(self, limits = (160,160,155), unit = 'mm'):
        '''
        Sets limit for every device axis separately. necessary to avoid collision with other set-up elements.
        Parameters: limits (tuple) 
        '''
        # set axes limits in millimetres (max. value is ?)
        self.connection.axis_x.settings.set('limit.max', limits[0], self.units[unit])
        self.connection.axis_y.settings.set('limit.max', limits[1], self.units[unit])
        self.connection.axis_z.settings.set('limit.max', limits[2], self.units[unit])


    def on_connect(self, home = True, start = (20,75, 130), limits =(160,160,155)):
        """startup routine to home, set range and move to start if desired. """
        if home:
            self.home_stage()
        self.set_rangelimits(limits)
        self.move_abs(start)
    
    def disconnect(self):
        """close com port connection."""
        self.connection.close()