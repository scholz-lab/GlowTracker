import asyncio
from zaber_motion import Library, Units, Measurement, MotionLibException
from zaber_motion.ascii import Connection, AlertEvent, AllAxes, Axis
from dataclasses import dataclass

from typing import Tuple, TypeAlias

from enum import Enum

# Declare common type
Vec3: TypeAlias = Tuple[float, float, float]

Library.enable_device_db_store()

@dataclass
class StageState:
    """State machine for the Zaber's stage"""
    isMoving_x: bool = False
    isMoving_y: bool = False
    isMoving_z: bool = False

class AxisEnum(Enum):
    """Enum for refering to which axis"""
    X = 1
    Y = 2
    Z = 3
    ALL = 4


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
            self.connection: Connection = connection
            self.assign_axes()
            self.set_maxspeed(speed = 20)
        
        self.state = StageState()
            
            
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
        except Exception as e:
            print(e)
            return None


    def assign_axes(self) -> None:
        """
        Order the axis and name them as x,y,z where x is the closest to the computer.
        """
        self.connection.renumber_devices(first_address=1)
        device_list = self.connection.detect_devices()

        print("Found {} devices".format(len(device_list)))

        # Get each axes' handler and inject them into the 
        #   Zaber's Connection class for ease of access.
        self.connection.axis_x: AxisEnum = device_list[0].get_axis(1)
        self.connection.axis_y: AxisEnum = device_list[1].get_axis(1)
        
        # Activate devices
        self.connection.axis_x.device.identify()
        self.connection.axis_y.device.identify()
        
        self.no_axes = 2

        # Optional 3rd axis
        if len(device_list) > 2:
            self.connection.axis_z: AxisEnum = device_list[2].get_axis(1)
            self.connection.axis_z.device.identify()
            self.no_axes = 3


    def set_maxspeed(self, speed = 20):
        if self.connection is not None:
            if self.no_axes ==3:
                for ax in [self.connection.axis_x, self.connection.axis_y, self.connection.axis_z]:
                    ax.settings.set("maxspeed", speed, Units.VELOCITY_MILLIMETRES_PER_SECOND)
                    speed = ax.settings.get("maxspeed", Units.VELOCITY_MILLIMETRES_PER_SECOND)
                    print("Maximum speed [mm/s]:", speed)
            elif self.no_axes == 2:
                for ax in [self.connection.axis_x, self.connection.axis_y]:
                    ax.settings.set("maxspeed", speed, Units.VELOCITY_MILLIMETRES_PER_SECOND)
                    speed = ax.settings.get("maxspeed", Units.VELOCITY_MILLIMETRES_PER_SECOND)
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
        

    def start_move(self, velocity: Vec3, unit = 'ums') -> None:
        """Start moving in a given velocity's direction.
            ALWAYS call in conjuction with self.stop() to stop moving.
        Parameters: 
                    velocity (tuple, float): can be positive or negative, position indicates which axis to move eg. (0,1,0) moves y axis only.
                    units(obj): has to be zaber units eg.  Units.LENGTH_MICROMETRES
        """
        if self.connection is None:
            return

        # Move each axis simultaneously
        if self.no_axes >= 1 and not self.state.isMoving_x and velocity[0] != 0:
            self.state.isMoving_x = True
            self.connection.axis_x.move_velocity(velocity[0], self.units[unit])
        
        if self.no_axes >= 2 and not self.state.isMoving_y and velocity[1] != 0:
            self.state.isMoving_y = True
            self.connection.axis_y.move_velocity(velocity[1], self.units[unit])

        if self.no_axes >= 3 and not self.state.isMoving_z and velocity[2] != 0:
            self.state.isMoving_z = True
            self.connection.axis_z.move_velocity(velocity[2], self.units[unit])


    def stop(self, stopAxis: AxisEnum = AxisEnum.ALL) -> None:
        """Stop movement of an axis or all axes

        Args:
            stopAxis (AxisEnum, optional): Specific axis to stop. Defaults to AxisEnum.ALL.
        """
        if self.connection is None:
            return
        
        if stopAxis == AxisEnum.ALL:
            self.connection.axis_x.stop(wait_until_idle = False)
            self.connection.axis_y.stop(wait_until_idle = False)
            if self.no_axes == 3:
                self.connection.axis_z.stop(wait_until_idle = False)
            
            self.state.isMoving_x = False
            self.state.isMoving_y = False
            self.state.isMoving_z = False


        elif stopAxis == AxisEnum.X:
            self.connection.axis_x.stop(wait_until_idle = False)
            self.state.isMoving_x = False
        
        elif stopAxis == AxisEnum.Y:
            self.connection.axis_y.stop(wait_until_idle = False)
            self.state.isMoving_y = False
        
        elif stopAxis == AxisEnum.Z and self.no_axes == 3:
            self.connection.axis_z.stop(wait_until_idle = False)
            self.state.isMoving_z = False


    def get_position(self, unit = 'mm', isAsync = True) -> Vec3 | None:
        """Get the current position of the stage for all axes."""
        # print('Stage::get_position()')
        # TODO: Investigate slowness
        if self.connection is None:
            return None
        
        pos: Vec3 | None = None
        
        try:
            if isAsync:

                loop = []
                loop.append(self.connection.axis_x.get_position_async(self.units[unit]))
                loop.append(self.connection.axis_y.get_position_async(self.units[unit]))

                if self.no_axes == 3:
                    loop.append(self.connection.axis_z.get_position_async(self.units[unit]))

                move_coroutine = asyncio.gather(*loop)
                event_loop = asyncio.get_event_loop()
                pos = event_loop.run_until_complete(move_coroutine)
                
            else:

                pos = []
                pos.append(self.connection.axis_x.get_position(self.units[unit]))
                pos.append(self.connection.axis_y.get_position(self.units[unit]))

                if self.no_axes == 3:
                    pos.append(self.connection.axis_z.get_position(self.units[unit]))
            
        except MotionLibException as e:
            # Handle exception
            #   This is usually a DeviceNotIdentifiedException from trying 
            #   get_position_async() while device is not fully initiated
            print(e)
        
        return pos


    def set_rangelimits(self, limits = (160,160,155), unit = 'mm'):
        '''
        Sets limit for every device axis separately. necessary to avoid collision with other set-up elements.
        Parameters: limits (tuple) 
        '''
        # set axes limits in millimetres (max. value is ?)
        if self.connection is not None:
            self.connection.axis_x.settings.set('limit.max', limits[0], self.units[unit])
            self.connection.axis_y.settings.set('limit.max', limits[1], self.units[unit])
            if self.no_axes == 3:
                self.connection.axis_z.settings.set('limit.max', limits[2], self.units[unit])


    def on_connect(self, home = True, startloc = True,  start = (20,75, 130), limits =(160,160,155)) -> None:
        """startup routine to home, set range and move to start if desired. """

        if home:
            self.home_stage()
        
        self.set_rangelimits(limits)

        if startloc:
            self.move_abs(start)
        
        device_list = self.connection.detect_devices()
        for device in device_list:
            device.all_axes.wait_until_idle(throw_error_on_fault = True)

    
    def disconnect(self):
        """close com port connection."""
        if self.connection is not None:
            self.connection.close()
    
    
    def is_busy(self):
        """report status"""
        if self.connection is not None:
            device_list = self.connection.detect_devices()
            for device in device_list:
                if device.all_axes.is_busy():#wait_until_idle =False)
                    return True
        return False
