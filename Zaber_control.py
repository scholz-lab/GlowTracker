import asyncio
from zaber_motion import Library, Units, MotionLibException
from zaber_motion.units import LITERALS_TO_UNITS, units_from_literals
from zaber_motion.ascii import Connection, Axis, Device
from dataclasses import dataclass
import math
from typing import List, Tuple, TypeAlias
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
    def __init__(self, port:str , maxspeed: float = 20, maxspeed_unit: str = 'mm/s',
                 accel: float = 60, accel_unit: str = 'mm/s^2'):
        """Initialize a wrapper around the stage set stage parameters.

        Args:
            port (str): connection port
            maxspeed (float, optional): maximum axes' speed. Defaults to 20.
            maxspeed_unit (str, optional): maximum axes' speed unit. Defaults to 'mm/s'.
            accel (float, optional): axes' acceleration. Defaults to 60.
            accel_unit (str, optional): axes' acceleration unit. Defaults to 'mm/s^2'.
        """                
        
        # Define class properties
        self.connection: Connection | None = None
        self.axis_x: Axis | None = None
        self.axis_y: Axis | None = None
        self.axis_z: Axis | None = None

        self.devices: List[Device] = []
        
        # Try connecting to the stage
        self.connection = self.connect_stage(port)
        print(self.connection)
        
        if self.connection is not None:
            self.assign_axes()
            self.maxspeed = self.set_maxspeed(maxspeed, LITERALS_TO_UNITS.get(maxspeed_unit))
            self.accel = self.set_accel(accel, LITERALS_TO_UNITS.get(accel_unit))
        
        self.state = StageState()
            
            
    def connect_stage(self, port='COM3'):
        """
        Connects to the zaber stage and pass the connection including the axes
        :param port: COM port to the stage
        :return: connection to the stage including the axes. Close it before closing using the close method
        """
        try:
            self.connection = Connection.open_serial_port(port)
            
            device_list = self.connection.detect_devices()

            print("Found {} devices".format(len(device_list)))
            if len(device_list) > 0:
                return self.connection
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
        self.devices = self.connection.detect_devices()

        print("Found {} devices".format(len(self.devices)))

        # Get each axes' handler and inject them into the 
        #   Zaber's Connection class for ease of access.
        self.axis_x = self.devices[0].get_axis(1)
        self.axis_y = self.devices[1].get_axis(1)
        
        # Activate devices
        self.axis_x.device.identify()
        self.axis_y.device.identify()
        
        self.no_axes = 2

        # Optional 3rd axis
        if len(self.devices) > 2:
            self.axis_z = self.devices[2].get_axis(1)
            self.axis_z.device.identify()
            self.no_axes = 3


    def set_maxspeed(self, maxspeed: float = 20.0, unit: Units = Units.VELOCITY_MILLIMETRES_PER_SECOND) -> float:
        """Set maximum speed to every axes.

        Args:
            speed (float, optional): maximum speed. Defaults to 20.0.
            unit (Units, optional): unit of the maximum speed. Defaults to Units.VELOCITY_MILLIMETRES_PER_SECOND.

        Returns:
            maxspeed (float): the device returned maximum speed, indicating the actual value it is set to
        """

        if self.connection is None:
            return

        axes: List[Axis] = []
        
        if self.no_axes == 2:
            axes = [self.axis_x, self.axis_y]
        elif self.no_axes == 3:
            axes = [self.axis_x, self.axis_y, self.axis_z]
        
        for axis in axes:
            # Set axis max speed
            axis.settings.set("maxspeed", maxspeed, unit)

            # Retrieve actual axis max speed
            self.maxspeed = axis.settings.get("maxspeed", unit)
            print(f'Maximum speed: {self.maxspeed}[{units_from_literals(unit)}]')
            
        return self.maxspeed
    

    def set_accel(self, accel: float = 60, unit: Units = Units.ACCELERATION_MILLIMETRES_PER_SECOND_SQUARED) -> float:
        """Set acceleration to every axes.

        Args:
            accel (float, optional): acceleration. Defaults to 60.0.
            unit (Units, optional): unit of the acceleration. Defaults to Units.ACCELERATION_MILLIMETRES_PER_SECOND_SQUARED.

        Returns:
            accel (float): the device returned acceleration, indicating the actual value it is set to
        """
        if self.connection is None:
            return

        axes: List[Axis] = []
        
        if self.no_axes == 2:
            axes = [self.axis_x, self.axis_y]
        elif self.no_axes == 3:
            axes = [self.axis_x, self.axis_y, self.axis_z]
        
        for axis in axes:
            # Set axis acceleration
            axis.settings.set("accel", accel, unit)

            # Retrieve actual axis acceleration 
            self.accel = axis.settings.get("accel", unit)
            print(f'Acceleration: {self.accel}[{units_from_literals(unit)}]')

        return self.accel


    #  Stage homing
    def home_stage(self):
        '''
        homes all connected devices & moves axes to starting positions
        necessary if device was disconnected from power source
        '''
        for device in self.devices:
            device.all_axes.home(wait_until_idle= True)
        

    # Stage moving to a given absolute position 
    def move_abs(self, position: List[float], unit: str = 'mm', wait_until_idle: bool = False) -> None:
        """Move to a given absolute position.

        Args:
            position (List[float]): The absolute position in order of x, y, z. Supports from 1 axis to 3 axes.
            unit (str, optional): Unit of the position. Defaults to 'mm'.
            wait_until_idle (bool, optional): Is the function return only after all axes finished moving. Defaults to False.
        """
        if self.connection is None:
            return
        
        pos_len = len(position) 
        
        if pos_len >= 1 and self.axis_x is not None and position[0] != 0:
            self.axis_x.move_absolute(position[0], LITERALS_TO_UNITS.get(unit), wait_until_idle)

        if pos_len >= 2 and self.axis_y is not None and position[1] != 0:
            self.axis_y.move_absolute(position[1], LITERALS_TO_UNITS.get(unit), wait_until_idle)
        
        if pos_len == 3 and self.axis_z is not None and position[2] != 0:
            self.axis_z.move_absolute(position[2], LITERALS_TO_UNITS.get(unit), wait_until_idle)

            
    # move single axis
    def move_x(self, step, unit = 'um', wait_until_idle = False):
        """Move to a given relative location
        Parameters: 
                    step (tuple): can be positive or negative, position indicates which axis to move eg. (0,1,0) moves y axis only.
                    units(str): string units, commonly used
        """
        if self.axis_x is not None:
            self.axis_x.move_relative(step, LITERALS_TO_UNITS.get(unit), wait_until_idle)
    
    # move single axis
    def move_y(self, step, unit = 'um', wait_until_idle = False):
        """Move to a given relative location
        Parameters: 
                    step (tuple): can be positive or negative, position indicates which axis to move eg. (0,1,0) moves y axis only.
                    units(str): string units, commonly used
        """
        if self.axis_y is not None:
            self.axis_y.move_relative(step, LITERALS_TO_UNITS.get(unit), wait_until_idle)
    
    # move single axis
    def move_z(self, step, unit = 'um', wait_until_idle = False):
        """Move to a given relative location
        Parameters: 
                    step (tuple): can be positive or negative, position indicates which axis to move eg. (0,1,0) moves y axis only.
                    units(str): string units, commonly used
        """
        if self.axis_z is not None:
            self.axis_z.move_relative(step, LITERALS_TO_UNITS.get(unit), wait_until_idle)


    # define generic movement function 
    def move_rel(self, steps: Tuple[float], unit: str = 'um', wait_until_idle: bool = False) -> None:
        """Move to a given relative steps

        Args:
            steps (Tuple[float]): The relative movement step vector in order of x, y, z. Supports from 1 axis to 3 axes.
            unit (str, optional): Unit of the steps. Defaults to 'um'.
            wait_until_idle (bool, optional): Is the function return only after all axes finished moving. Defaults to False.
        """        
        if self.connection is None:
            return
        
        pos_len = len(steps) 
       
        if pos_len >= 1 and steps[0] != 0:
            self.move_x(steps[0], unit = unit, wait_until_idle=wait_until_idle)

        if pos_len >= 2 and steps[1] != 0:
            self.move_y(steps[1], unit = unit, wait_until_idle=wait_until_idle)
        
        if pos_len == 3 and steps[2] != 0:
            self.move_z(steps[1], unit = unit, wait_until_idle=wait_until_idle)
        

    def start_move(self, velocity: Vec3, unit: str = 'um/s') -> None:
        """Start moving in a given velocity's direction.
            ALWAYS call in conjuction with self.stop() to stop moving.
        Parameters: 
                    velocity (Vec3, float): can be positive or negative, position indicates which axis to move eg. (0,1,0) moves y axis only.
                    units(str, optional): has to be zaber units eg.  Units.LENGTH_MICROMETRES
        """
        # Move each axis simultaneously
        if self.axis_x is not None and not self.state.isMoving_x and velocity[0] != 0:
            self.state.isMoving_x = True
            self.axis_x.move_velocity(velocity[0], LITERALS_TO_UNITS.get(unit))
        
        if self.axis_y is not None and not self.state.isMoving_y and velocity[1] != 0:
            self.state.isMoving_y = True
            self.axis_y.move_velocity(velocity[1], LITERALS_TO_UNITS.get(unit))

        if self.axis_z is not None and not self.state.isMoving_z and velocity[2] != 0:
            self.state.isMoving_z = True
            self.axis_z.move_velocity(velocity[2], LITERALS_TO_UNITS.get(unit))


    def stop(self, stopAxis: AxisEnum = AxisEnum.ALL) -> None:
        """Stop movement of an axis or all axes

        Args:
            stopAxis (AxisEnum, optional): Specific axis to stop. Defaults to AxisEnum.ALL.
        """
        if self.connection is None:
            return
        
        if stopAxis == AxisEnum.ALL:
            self.axis_x.stop(wait_until_idle = False)
            self.axis_y.stop(wait_until_idle = False)
            if self.axis_z is not None:
                self.axis_z.stop(wait_until_idle = False)
            
            self.state.isMoving_x = False
            self.state.isMoving_y = False
            self.state.isMoving_z = False


        elif stopAxis == AxisEnum.X:
            self.axis_x.stop(wait_until_idle = False)
            self.state.isMoving_x = False
        
        elif stopAxis == AxisEnum.Y:
            self.axis_y.stop(wait_until_idle = False)
            self.state.isMoving_y = False
        
        elif stopAxis == AxisEnum.Z and self.no_axes == 3:
            self.axis_z.stop(wait_until_idle = False)
            self.state.isMoving_z = False


    def get_position(self, unit: str = 'mm', isAsync: bool = True) -> Vec3 | None:
        """Get the current position of the stage for all axes.

        Args:
            unit (str, optional): Unit to get position in. Defaults to 'mm'.
            isAsync (bool, optional): Is running in async mode. Defaults to True.

        Returns:
            pos (Vec3 | None): Position. Return None if the execution is unsuccessful.
        """

        if self.connection is None:
            return None
        
        pos: Vec3 | None = None
        
        try:
            if isAsync:

                loop = []

                loop.append(self.axis_x.get_position_async(LITERALS_TO_UNITS.get(unit)))
                loop.append(self.axis_y.get_position_async(LITERALS_TO_UNITS.get(unit)))
                if self.axis_z is not None:
                    loop.append(self.axis_z.get_position_async(LITERALS_TO_UNITS.get(unit)))

                move_coroutine = asyncio.gather(*loop)
                event_loop = asyncio.get_event_loop()
                pos = event_loop.run_until_complete(move_coroutine)
                
            else:

                pos = []
                
                pos.append(self.axis_x.get_position(LITERALS_TO_UNITS.get(unit)))
                pos.append(self.axis_y.get_position(LITERALS_TO_UNITS.get(unit)))
                if self.axis_z is not None:
                    pos.append(self.axis_z.get_position(LITERALS_TO_UNITS.get(unit)))
            
        except MotionLibException as e:
            # Handle exception
            #   This is usually a DeviceNotIdentifiedException from trying 
            #   get_position_async() while device is not fully initiated
            print(e)
        
        return pos


    def set_rangelimits(self, limits: List[float] = (160,160,155), unit: str = 'mm') -> None:
        """Sets limit for every device axis separately. necessary to avoid collision with other set-up elements.

        Args:
            limits (List[float], optional): Axis range limit. Defaults to (160,160,155).
            unit (str, optional): Axis limit. Defaults to 'mm'.
        """        
        # set axes limits in millimetres (max. value is ?)
        if self.connection is None:
            return
        
        self.axis_x.settings.set('limit.max', limits[0], LITERALS_TO_UNITS.get(unit))
        self.axis_y.settings.set('limit.max', limits[1], LITERALS_TO_UNITS.get(unit))
        if self.axis_z is not None:
            self.axis_z.settings.set('limit.max', limits[2], LITERALS_TO_UNITS.get(unit))


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
    
    
    def is_busy(self) -> bool:
        """Check if any of the devices is busy

        Returns:
            isBusy (bool): boolean indicating if any of the devices is busy
        """
        for device in self.devices:
            if device.all_axes.is_busy():
                return True
        
        return False
    
    
    def estimateTravelTime( self, dist: float ) -> float:
        """Estimate the travel time with assumption of 0 acceleration ramping time.
        The acceleration ramping time is a feature to set the acceleration to a linear function
        instead of a constant value, which makes the velocity become a smooth curve (quadratic)
        instead of a linear line, which makes reduce motion jerkness.
            However, it is a bit more tricky to compute an estimated time with such a profile,
        and since 0 acceleration ramp time is the default setting, we will focus only in this case.
        https://www.zaber.com/protocol-manual#topic_setting_motion_accel_ramptime
            In the case of no acceleration time, the velocity function is a linear pice-wise 
        function consist of 3 parts: ramp-up (increase velocity), stable (stable at maximum velocity),
        and ramp-down. Which forms a trapezoid shape.
            By Zaber's design, the acceleration when ramping up and ramping down are scalar value
        with the same size but in an opposite direction, described by 'motion.accelonly'
        and 'motion.decelonly' respectively
        (https://www.zaber.com/protocol-manual?device=X-LSM150A&peripheral=N%2FA&version=7.34&protocol=ASCII#topic_setting_motion).
        This resulting in a vertically symmetric trapezoid shape (i.e. isosceles trapezoid),
        which we can derive an estimated travel time (x-axis in the velocity graph) which 
        is the width of the shape by the given acceleration and distance (area of the graph).
            However, in the case where the travelling speed did not get ramp up fast enough to
        reach the maximum speed before started to slowing down, the shape becomes a triangle,
        which simplifies the computation by a bit, but will have a different closed-form solution.

            Unit of the return value depends on the unit of the inputs. If the inputs
        have same exponential unit e.g. accel: mm/s^2, dist: mm, maxspeed: mm/s, the
        resulting output will be in the second unit (s). Otherwise, please handle it
        accordingly.
        
        Args:
            dist (float): travel distance

        Returns:
            time (float): estimated travel time
        """

        # Compute minimum distance to reach max velocity
        dist_to_reach_v_max = self.maxspeed*self.maxspeed / self.accel

        estimated_travel_time = 0

        if dist <= dist_to_reach_v_max:
            # Isosceles Triangle shape
            estimated_travel_time = 2.0 * math.sqrt( dist / self.accel )
            
        else:
            # Isosceles Trapezoid shape
            time_ramp_up = self.maxspeed / self.accel

            dist_travel_at_vel_max = dist - (self.maxspeed * self.maxspeed / self.accel)

            time_travel_at_vel_max = dist_travel_at_vel_max / self.maxspeed

            estimated_travel_time = 2.0 * time_ramp_up + time_travel_at_vel_max
        
        return estimated_travel_time
