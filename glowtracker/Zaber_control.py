import asyncio
from collections import deque
import threading
import time
from zaber_motion import Library, Units, MotionLibException, MovementFailedException, CommandFailedException
from zaber_motion.units import units_from_literals, LITERALS_TO_UNITS, UnitsAndLiterals, Units
from zaber_motion.ascii import Connection, Axis, Device
from dataclasses import dataclass
import math
from typing import List, Tuple, TypeAlias, Literal
from enum import Enum

# Default settings
DEFAULT_MAXSPEED = 20.0
DEFAULT_MAXSPEED_UNIT = 'mm/s'
DEFAULT_ACCEL = 60.0
DEFAULT_ACCEL_UNIT = 'mm/s^2'
POSITION_POLL_INTERVAL = 0.2
JOG_SAFETY_POLL_INTERVAL = 0.1
POSITION_POLLER_JOIN_TIMEOUT = 2.0
_POLL_INTERRUPTED = object()

# Declare common type
Vec3: TypeAlias = Tuple[float, float, float]

Library.enable_device_db_store()

# Unit conversion
UNITS_TO_LITERALS = {value: key for key, value in LITERALS_TO_UNITS.items()}

def units_to_literals(units: UnitsAndLiterals) -> str:
    if isinstance(units, str):
        return units

    converted = UNITS_TO_LITERALS.get(units)
    if converted is None:
        raise ValueError(f"Invalid units: {units}")

    return converted

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
    def __init__(self, port:str , maxspeed: float = DEFAULT_MAXSPEED, maxspeed_unit: str = DEFAULT_MAXSPEED_UNIT,
                 accel: float = DEFAULT_ACCEL, accel_unit: str = DEFAULT_ACCEL_UNIT):
        """Initialize a wrapper around the stage set stage parameters.

        Args:
            port (str): connection port
            maxspeed (float, optional): maximum axes' speed. Defaults to 20.
            maxspeed_unit (str, optional): maximum axes' speed unit. Defaults to DEFAULT_MAXSPEED_UNIT.
            accel (float, optional): axes' acceleration. Defaults to DEFAULT_ACCEL.
            accel_unit (str, optional): axes' acceleration unit. Defaults to DEFAULT_ACCEL_UNIT.
        """

        # Define class properties
        self.connection: Connection | None = None
        self.axis_x: Axis | None = None
        self.axis_y: Axis | None = None
        self.axis_z: Axis | None = None

        self.devices: List[Device] = []
        self._accel_signature = None

        # Try connecting to the stage
        self.connection = self.connect_stage(port)

        if self.connection is not None:
            print(f'Connection to stage: {self.connection}')
            self.assign_axes()
            self.maxspeed = self.set_maxspeed(maxspeed, units_from_literals(maxspeed_unit))
            self.accel = self.set_accel(accel, units_from_literals(accel_unit))

        self.state = StageState()

        self._jog_velocity = [0.0, 0.0, 0.0]
        self._disconnecting = False

        self._last_pos: List[float] | None = None
        self._last_pos_time: float = 0.0
        self._position_cache_condition = threading.Condition()
        self._position_read_lock = threading.Lock()
        self._position_poller_lock = threading.Lock()
        self._position_poll_stop = threading.Event()
        self._position_poll_wake = threading.Event()
        self._position_poll_thread: threading.Thread | None = None
        self._jog_command_lock = threading.Lock()
        self._jog_commands = deque()
        self._requested_jog_axes = [False, False, False]
        self._jog_generations = [0, 0, 0]
        self._queued_stop_generations = [-1, -1, -1]


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
            print(f'Connect stage error: {e}')
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


    def set_maxspeed(self, maxspeed: float = DEFAULT_MAXSPEED, unit: str = DEFAULT_MAXSPEED_UNIT) -> float:
        """Set maximum speed to every axes.

        Args:
            speed (float, optional): maximum speed. Defaults to DEFAULT_MAXSPEED.
            unit (str, optional): unit of the maximum speed. Defaults to DEFAULT_MAXSPEED_UNIT.

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
            try:
                axis.settings.set("maxspeed", maxspeed, units_from_literals(unit))

            except CommandFailedException as e:
                print(f'Setting maxspeed error: {e}')

            # Retrieve actual axis max speed
            self.maxspeed = axis.settings.get("maxspeed", unit)
            print(f'Maximum speed: {self.maxspeed:.4f} {units_to_literals(unit)}')

        return self.maxspeed


    def set_accel(self, accel: float = DEFAULT_ACCEL, unit: str = DEFAULT_ACCEL_UNIT) -> float:
        """Set acceleration to every axes.

        Args:
            accel (float, optional): acceleration. Defaults to DEFAULT_ACCEL.
            unit (str, optional): unit of the acceleration. Defaults to DEFAULT_ACCEL_UNIT.

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
            try:
                # Set axis acceleration
                axis.settings.set("accel", accel, units_from_literals(unit))

            except CommandFailedException as e:
                print(f'Setting acceleration error: {e}')

            # Retrieve actual axis acceleration
            self.accel = axis.settings.get("accel", units_from_literals(unit))
            print(f'Acceleration: {self.accel:.4f} {units_to_literals(unit)}')

        self._accel_signature = (float(accel), units_to_literals(unit))
        return self.accel


    def set_motion(self, maxspeed: float, accel: float, maxspeed_unit: str = DEFAULT_MAXSPEED_UNIT, accel_unit: str = DEFAULT_ACCEL_UNIT) -> None:
        self.set_maxspeed(maxspeed, maxspeed_unit)
        self.set_accel(accel, accel_unit)


    #  Stage homing
    def home_stage(self, cancel_event=None) -> bool:
        '''
        homes all connected devices & moves axes to starting positions
        necessary if device was disconnected from power source
        '''
        if self.connection is None:
            return False

        def cancelled():
            return cancel_event is not None and cancel_event.is_set()

        if cancelled():
            return False
        if self.axis_z is not None:
            self.axis_z.home(wait_until_idle= True)
        if cancelled():
            return False
        self.axis_y.home(wait_until_idle= False)
        if cancelled():
            return False
        self.axis_x.home(wait_until_idle= True)
        return not cancelled()

    def wait_until_idle(self) -> None:
        """Wait until all axes
        """
        if self.connection is None:
            return

        self.axis_x.wait_until_idle()
        self.axis_y.wait_until_idle()
        if self.axis_z is not None:
            self.axis_z.wait_until_idle()


    # Stage moving to a given absolute position
    KEEPOUT_Y = 45.0      # mm
    KEEPOUT_Z = 130.0
    KEEPOUT_MARGIN = 1.0

    _UNIT_TO_MM = {'mm': 1.0, 'um': 0.001, 'cm': 10.0}

    def is_safe(self, x: float, y: float, z: float) -> bool:
        y_lim = self.KEEPOUT_Y + self.KEEPOUT_MARGIN
        z_lim = self.KEEPOUT_Z - self.KEEPOUT_MARGIN
        return not (y < y_lim and z > z_lim)

    def _execute_safe_moves(self, target: List[float], cur: List[float], wait_until_idle: bool) -> None:
        mm = units_from_literals('mm')
        z_lim = self.KEEPOUT_Z - self.KEEPOUT_MARGIN

        def go_xy():
            self.axis_x.move_absolute(target[0], mm, False)
            self.axis_y.move_absolute(target[1], mm, False)
            self.axis_x.wait_until_idle()
            self.axis_y.wait_until_idle()

        def go_z():
            if self.axis_z is not None:
                self.axis_z.move_absolute(target[2], mm, wait_until_idle)

        if self.axis_z is not None and target[2] > z_lim:
            go_xy()
            go_z()
        elif self.axis_z is not None and cur[2] > z_lim:
            self.axis_z.move_absolute(target[2], mm, True)
            go_xy()
        else:
            go_xy()
            go_z()

    def move_abs(self, position: List[float], unit: str = 'mm', wait_until_idle: bool = False) -> bool:
        """Move to a given absolute position.

        Args:
            position (List[float]): The absolute position in order of x, y, z. Supports from 1 axis to 3 axes.
            unit (str, optional): Unit of the position. Defaults to 'mm'.
            wait_until_idle (bool, optional): Is the function return only after all axes finished moving. Defaults to False.

        Returns:
            bool: True if the move completed, False if refused or faulted.
        """
        if self.connection is None:
            return False

        factor = self._UNIT_TO_MM.get(unit)
        if factor is None:
            print(f'move_abs: unknown unit {unit!r}; refusing for safety')
            return False

        cur = self.get_position(unit='mm', isAsync=False)
        if cur is None:
            print('move_abs: cannot read current position; refusing')
            return False

        target = list(cur)
        for i in range(min(len(position), 3)):
            target[i] = float(position[i]) * factor

        if self.axis_z is not None and not self.is_safe(*target):
            print(f'move_abs refused: target {target} mm is inside the keep-out zone')
            return False

        try:
            self._execute_safe_moves(target, cur, wait_until_idle)
        except MotionLibException as e:
            print(f'move_abs to {target} mm failed: {e}')
            return False

        return True


    # move single axis
    def move_x(self, step: float, unit: str = 'um', wait_until_idle: bool = False) -> bool:
        """Move to a given relative location

        Args:
            step (float): can be positive or negative, position indicates which axis to move eg. (0,1,0) moves y axis only.
            unit (str, optional): Unit of the step. Defaults to 'um'.
            wait_until_idle (bool, optional): is wait until finished moving. Defaults to False.

        Returns:
            bool: True if the move command was issued without fault, False otherwise.
        """
        try:
            if self.axis_x is not None:
                self.axis_x.move_relative(float(step), units_from_literals(unit), wait_until_idle)

        except MotionLibException as e:
            print(f'move_x by {step} {unit} failed: {e}')
            return False

        return True


    # move single axis
    def move_y(self, step, unit = 'um', wait_until_idle = False, check_safety: bool = True) -> bool:
        """Move to a given relative location.
        Parameters:
                    step (tuple): can be positive or negative, position indicates which axis to move eg. (0,1,0) moves y axis only.
                    units(str): string units, commonly used
                    check_safety (bool): when False, skip the keep-out check and its position read. Used in the tracking loop where Z is fixed.
        Returns:
                    bool: True if the move command was issued without fault, False otherwise.
        """
        if check_safety:
            factor = self._UNIT_TO_MM.get(unit)
            if factor is None:
                print(f'move_y: unknown unit {unit!r}; refusing for safety')
                return False
            cur = self._safe_position_mm()
            if cur is None:
                print('move_y: cannot read current position; refusing')
                return False
            if self.axis_z is not None and len(cur) > 2:
                target_y = cur[1] + float(step) * factor
                if not self.is_safe(cur[0], target_y, cur[2]):
                    print(f'move_y refused: would enter keep-out (y={target_y:.1f}, z={cur[2]:.1f})')
                    return False
        try:
            if self.axis_y is not None:
                self.axis_y.move_relative(float(step), units_from_literals(unit), wait_until_idle)

        except MotionLibException as e:
            print(f'move_y by {step} {unit} failed: {e}')
            return False

        return True


    # move single axis
    def move_z(self, step, unit = 'um', wait_until_idle = False) -> bool:
        """Move to a given relative location.
        Parameters:
                    step (tuple): can be positive or negative, position indicates which axis to move eg. (0,1,0) moves y axis only.
                    units(str): string units, commonly used
        Returns:
                    bool: True if the move command was issued without fault, False otherwise.
        """
        factor = self._UNIT_TO_MM.get(unit)
        if factor is None:
            print(f'move_z: unknown unit {unit!r}; refusing for safety')
            return False
        cur = self._safe_position_mm()
        if cur is None:
            print('move_z: cannot read current position; refusing')
            return False
        if self.axis_z is not None and len(cur) > 2:
            target_z = cur[2] + float(step) * factor
            if not self.is_safe(cur[0], cur[1], target_z):
                print(f'move_z refused: would enter keep-out (y={cur[1]:.1f}, z={target_z:.1f})')
                return False
        try:
            if self.axis_z is not None:
                self.axis_z.move_relative(float(step), units_from_literals(unit), wait_until_idle)

        except MotionLibException as e:
            print(f'move_z by {step} {unit} failed: {e}')
            return False

        return True


    def move_rel(self, steps: Tuple[float], unit: str = 'um', wait_until_idle: bool = False) -> bool:
        """Move by relative steps, routed through the same collision-safe planner
        as move_abs. Refuses if the resulting position is inside the keep-out zone.

        Args:
            steps (Tuple[float]): The relative movement step vector in order of x, y, z. Supports from 1 axis to 3 axes.
            unit (str, optional): Unit of the steps. Defaults to 'um'.
            wait_until_idle (bool, optional): Is the function return only after all axes finished moving. Defaults to False.

        Returns:
            bool: True if the move completed, False if refused or faulted.
        """
        if self.connection is None:
            return False

        factor = self._UNIT_TO_MM.get(unit)
        if factor is None:
            print(f'move_rel: unknown unit {unit!r}; refusing for safety')
            return False

        cur = self.get_position(unit='mm', isAsync=False)
        if cur is None:
            print('move_rel: cannot read current position; refusing')
            return False

        target = list(cur)
        for i in range(min(len(steps), 3)):
            target[i] = cur[i] + float(steps[i]) * factor

        if self.axis_z is not None and not self.is_safe(*target):
            print(f'move_rel refused: target {target} mm is inside the keep-out zone')
            return False

        try:
            self._execute_safe_moves(target, cur, wait_until_idle)
        except MotionLibException as e:
            print(f'move_rel by {steps} {unit} failed: {e}')
            return False

        return True


    def move_xy(self, x: float, y: float, unit: str = 'mm', wait_until_idle: bool = True) -> bool:
        if self.connection is None:
            return False
        try:
            u = units_from_literals(unit)
            self.axis_x.move_absolute(float(x), u, False)
            self.axis_y.move_absolute(float(y), u, False)
            if wait_until_idle:
                self.axis_x.wait_until_idle()
                self.axis_y.wait_until_idle()
        except MotionLibException as e:
            print(f'move_xy to ({x}, {y}) {unit} failed: {e}')
            return False
        return True

    def start_move(self, velocity: Vec3, unit: str = 'um/s') -> bool:
        """Start moving in a given velocity's direction.
            ALWAYS call in conjuction with self.stop() to stop moving.
        Parameters:
                    velocity (Vec3, float): can be positive or negative, position indicates which axis to move eg. (0,1,0) moves y axis only.
                    units(str, optional): has to be zaber units eg.  Units.LENGTH_MICROMETRES
        """
        if self.connection is None or self._disconnecting:
            return False

        safety_poll_required = self.axis_z is not None \
            and (velocity[1] != 0 or velocity[2] != 0)
        if safety_poll_required and not self.start_position_poller():
            self.emergency_stop()
            return False

        try:
            # Move each axis simultaneously
            if self.axis_x is not None and not self.state.isMoving_x and velocity[0] != 0:
                self.state.isMoving_x = True
                self._jog_velocity[0] = velocity[0]
                self.axis_x.move_velocity(float(velocity[0]), units_from_literals(unit))

            if self.axis_y is not None and not self.state.isMoving_y and velocity[1] != 0:
                self.state.isMoving_y = True
                self._jog_velocity[1] = velocity[1]
                self.axis_y.move_velocity(float(velocity[1]), units_from_literals(unit))

            if self.axis_z is not None and not self.state.isMoving_z and velocity[2] != 0:
                self.state.isMoving_z = True
                self._jog_velocity[2] = velocity[2]
                self.axis_z.move_velocity(float(velocity[2]), units_from_literals(unit))
        except MotionLibException as e:
            print(f'start_move at velocity {velocity} {unit} failed: {e}')
            self.emergency_stop()
            return False

        if safety_poll_required:
            self._position_poll_wake.set()

        return True

    def request_start_move(
            self,
            velocity: Vec3,
            unit: str = 'um/s',
            accel: float | None = None,
            accel_unit: str | None = None) -> bool:
        if self.connection is None or self._disconnecting:
            return False
        if not self.start_position_poller():
            return False
        with self._jog_command_lock:
            moving_axes = [value != 0 for value in velocity]
            if any(
                    requested and moving
                    for requested, moving
                    in zip(self._requested_jog_axes, moving_axes)):
                return False
            for index, moving in enumerate(moving_axes):
                if moving:
                    self._requested_jog_axes[index] = True
                    self._jog_generations[index] += 1
            self._jog_commands.append(
                ('start', tuple(velocity), unit, accel, accel_unit)
            )
        self._position_poll_wake.set()
        return True

    def request_stop(self, stopAxis: AxisEnum = AxisEnum.ALL) -> bool:
        if self.connection is None or self._disconnecting:
            return False
        if not self.start_position_poller():
            return False
        with self._jog_command_lock:
            if stopAxis == AxisEnum.ALL:
                indices = range(3)
            else:
                indices = [stopAxis.value - 1]
            indices = list(indices)
            if all(
                    self._queued_stop_generations[index]
                    == self._jog_generations[index]
                    for index in indices):
                return True
            for index in indices:
                self._requested_jog_axes[index] = False
                self._queued_stop_generations[index] = self._jog_generations[index]
            self._jog_commands.append(('stop', stopAxis))
        self._position_poll_wake.set()
        return True

    def _has_jog_commands(self) -> bool:
        with self._jog_command_lock:
            return bool(self._jog_commands)

    def _clear_jog_commands(self) -> None:
        with self._jog_command_lock:
            self._jog_commands.clear()
            self._requested_jog_axes = [False, False, False]
            self._jog_generations = [0, 0, 0]
            self._queued_stop_generations = [-1, -1, -1]

    def _process_jog_commands(self) -> None:
        while not self._disconnecting:
            with self._jog_command_lock:
                if not self._jog_commands:
                    return
                command = self._jog_commands.popleft()

            try:
                if command[0] == 'start':
                    _, velocity, unit, accel, accel_unit = command
                    if accel is not None and accel_unit is not None:
                        signature = (float(accel), units_to_literals(accel_unit))
                        if signature != self._accel_signature:
                            self.set_accel(float(accel), accel_unit)
                    self.start_move(velocity, unit)
                else:
                    self._stop_jog_no_response(command[1])
            except Exception as e:
                print(f'Stage jog command failed: {e}')
                self.emergency_stop()
                return

    def _stop_jog_no_response(self, stopAxis: AxisEnum = AxisEnum.ALL) -> bool:
        if self.connection is None:
            return False

        axes = (
            (AxisEnum.X, self.axis_x, 'isMoving_x', 0),
            (AxisEnum.Y, self.axis_y, 'isMoving_y', 1),
            (AxisEnum.Z, self.axis_z, 'isMoving_z', 2),
        )
        selected = [
            entry for entry in axes
            if stopAxis == AxisEnum.ALL or entry[0] == stopAxis
        ]

        try:
            for _, axis, _, _ in selected:
                if axis is not None:
                    axis.generic_command_no_response('stop')
        except Exception as e:
            print(f'Stage jog stop failed: {e}')
            self.emergency_stop()
            return False

        for _, axis, state_name, index in selected:
            if axis is not None:
                setattr(self.state, state_name, False)
                self._jog_velocity[index] = 0.0
        self._position_poll_wake.set()
        return True

    def _check_jog_safety(self, pos: List[float]) -> None:
        y_lim = self.KEEPOUT_Y + self.KEEPOUT_MARGIN
        z_lim = self.KEEPOUT_Z - self.KEEPOUT_MARGIN
        BUFFER = 5.0

        y, z = pos[1], pos[2]
        vy, vz = self._jog_velocity[1], self._jog_velocity[2]

        if self.state.isMoving_y and z > z_lim and vy < 0 and y < y_lim + BUFFER:
            self.stop(AxisEnum.Y)
        if self.state.isMoving_z and y < y_lim and vz > 0 and z > z_lim - BUFFER:
            self.stop(AxisEnum.Z)

    def _position_poll_loop(self) -> None:
        while not self._position_poll_stop.is_set():
            if self.connection is None or self._disconnecting:
                break

            self._process_jog_commands()
            if self._position_poll_stop.is_set() \
                    or self.connection is None or self._disconnecting:
                break

            safety_poll = self.axis_z is not None and (
                self.state.isMoving_y or self.state.isMoving_z
            )
            try:
                pos = self._get_polled_position()
            except Exception as e:
                print(f'Stage position polling failed: {e}')
                pos = None

            if pos is _POLL_INTERRUPTED:
                continue

            if safety_poll:
                if pos is None or len(pos) < 3:
                    self.emergency_stop()
                else:
                    self._check_jog_safety(pos)

            interval = JOG_SAFETY_POLL_INTERVAL if safety_poll else POSITION_POLL_INTERVAL
            self._position_poll_wake.wait(interval)
            self._position_poll_wake.clear()

    def _get_polled_position(self):
        positions = []
        axes = [self.axis_x, self.axis_y]
        if self.axis_z is not None:
            axes.append(self.axis_z)

        with self._position_read_lock:
            for axis in axes:
                if self._has_jog_commands() or self._position_poll_stop.is_set():
                    return _POLL_INTERRUPTED
                positions.append(axis.get_position(units_from_literals('mm')))

        if self.axis_z is None:
            positions.append(0.0)

        if self._has_jog_commands() or self._position_poll_stop.is_set():
            return _POLL_INTERRUPTED
        self._cache_position(positions, 'mm')
        return positions

    def start_position_poller(self) -> bool:
        if self.connection is None or self._disconnecting:
            return False
        with self._position_poller_lock:
            if self._position_poll_thread is not None \
                    and self._position_poll_thread.is_alive():
                return True
            self._position_poll_stop.clear()
            self._position_poll_wake.clear()
            self._position_poll_thread = threading.Thread(
                target=self._position_poll_loop,
                daemon=True,
                name='StagePositionPoller',
            )
            self._position_poll_thread.start()
        return True

    def stop_position_poller(self, timeout: float = POSITION_POLLER_JOIN_TIMEOUT) -> bool:
        self._clear_jog_commands()
        self._position_poll_stop.set()
        self._position_poll_wake.set()
        with self._position_poller_lock:
            thread = self._position_poll_thread
        if thread is None or thread is threading.current_thread():
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def get_cached_position(self, unit: str = 'mm', max_age: float | None = None) -> List[float] | None:
        factor = self._UNIT_TO_MM.get(unit)
        if factor is None:
            return None
        with self._position_cache_condition:
            if self._last_pos is None:
                return None
            if max_age is not None and time.monotonic() - self._last_pos_time > max_age:
                return None
            return [value / factor for value in self._last_pos]


    def emergency_stop(self) -> bool:
        self._clear_jog_commands()
        if self.connection is None:
            self.state = StageState()
            self._jog_velocity = [0.0, 0.0, 0.0]
            return False

        stopped = False
        try:
            self.connection.stop_all(wait_until_idle=False)
            stopped = True
        except Exception as e:
            print(f'Stage stop-all failed: {e}')
            for axis in (self.axis_x, self.axis_y, self.axis_z):
                if axis is None:
                    continue
                try:
                    axis.stop(wait_until_idle=False)
                    stopped = True
                except Exception as axis_error:
                    print(f'Stage axis stop failed: {axis_error}')
        finally:
            self.state = StageState()
            self._jog_velocity = [0.0, 0.0, 0.0]

        return stopped


    def stop(self, stopAxis: AxisEnum = AxisEnum.ALL) -> None:
        """Stop movement of an axis or all axes

        Args:
            stopAxis (AxisEnum, optional): Specific axis to stop. Defaults to AxisEnum.ALL.
        """
        if self.connection is None:
            return

        if stopAxis == AxisEnum.ALL:
            self.emergency_stop()
            return

        try:
            if stopAxis == AxisEnum.X:
                self.axis_x.stop(wait_until_idle = False)
                self.state.isMoving_x = False
                self._jog_velocity[0] = 0.0

            elif stopAxis == AxisEnum.Y:
                self.axis_y.stop(wait_until_idle = False)
                self.state.isMoving_y = False
                self._jog_velocity[1] = 0.0

            elif stopAxis == AxisEnum.Z and self.no_axes == 3:
                self.axis_z.stop(wait_until_idle = False)
                self.state.isMoving_z = False
                self._jog_velocity[2] = 0.0

        except MotionLibException as e:
            print(e)


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
            with self._position_read_lock:
                if isAsync:

                    loop = []

                    loop.append(self.axis_x.get_position_async(units_from_literals(unit)))
                    loop.append(self.axis_y.get_position_async(units_from_literals(unit)))
                    if self.axis_z is not None:
                        loop.append(self.axis_z.get_position_async(units_from_literals(unit)))

                    move_coroutine = asyncio.gather(*loop)
                    event_loop = asyncio.get_event_loop()
                    pos = event_loop.run_until_complete(move_coroutine)

                else:

                    pos = []

                    pos.append(self.axis_x.get_position(units_from_literals(unit)))
                    pos.append(self.axis_y.get_position(units_from_literals(unit)))
                    if self.axis_z is not None:
                        pos.append(self.axis_z.get_position(units_from_literals(unit)))

        except MotionLibException as e:
            # Handle exception
            #   This is usually a DeviceNotIdentifiedException from trying
            #   get_position_async() while device is not fully initiated
            print(e)
            return None

        if self.axis_z is None:
            pos = list(pos) + [0.0]

        if pos is not None:
            self._cache_position(pos, unit)

        return pos

    def _cache_position(self, pos, unit: str) -> None:
        factor = self._UNIT_TO_MM.get(unit, 1.0)
        with self._position_cache_condition:
            self._last_pos = [p * factor for p in pos]
            self._last_pos_time = time.monotonic()
            self._position_cache_condition.notify_all()

    def _safe_position_mm(self, max_age: float = 0.3) -> List[float] | None:
        pos = self.get_cached_position(unit='mm', max_age=max_age)
        if pos is not None:
            return pos
        if self._position_poll_thread is not None and self._position_poll_thread.is_alive():
            with self._position_cache_condition:
                previous_update = self._last_pos_time
                self._position_poll_wake.set()
                self._position_cache_condition.wait_for(
                    lambda: self._last_pos_time > previous_update
                    or self._position_poll_stop.is_set(),
                    timeout=max(POSITION_POLL_INTERVAL * 2, 0.5),
                )
            return self.get_cached_position(unit='mm', max_age=max_age)
        return self.get_position(unit='mm', isAsync=False)


    def set_rangelimits(self, limits: List[float] = (160,160,155), unit: str = 'mm') -> List[float]:
        """Sets limit for every device axis separately. necessary to avoid collision with other set-up elements.

        Args:
            limits (List[float], optional): Axis range limit. Defaults to (160,160,155).
            unit (str, optional): Axis limit. Defaults to 'mm'.

        Returns:
            rangelimits List[float]: the device returned maximum ranges, indicating the actual value it is set to. The list is of lenght 2 or 3 depending how many axes there are
        """
        # set axes limits in millimetres (max. value is ?)
        if self.connection is None:
            return

        rangelimits: List[float] = [0, 0]

        # Axis 1
        try:
            self.axis_x.settings.set('limit.max', limits[0], units_from_literals(unit))

        except CommandFailedException as e:
            print(f'Setting stage limit error: {e}')

        rangelimits[0] = self.axis_x.settings.get('limit.max', units_from_literals(unit))

        # Axis 2
        try:
            self.axis_y.settings.set('limit.max', limits[1], units_from_literals(unit))

        except CommandFailedException as e:
            print(f'Setting stage limit error: {e}')

        rangelimits[1] = self.axis_y.settings.get('limit.max', units_from_literals(unit))

        # Optional, Axis 3
        if self.axis_z is not None:
            try:
                self.axis_z.settings.set('limit.max', limits[2], units_from_literals(unit))

            except CommandFailedException as e:
                print(f'Setting stage limit error: {e}')

            rangelimits.append( self.axis_z.settings.get('limit.max', units_from_literals(unit)) )

        return rangelimits


    def on_connect(self, home = True, startloc = True,  start = (20,75, 130), limits =(160,160,155), cancel_event=None) -> bool:
        """startup routine to home, set range and move to start if desired. """

        def cancelled():
            return cancel_event is not None and cancel_event.is_set()

        if cancelled() or self.connection is None:
            return False

        if home:
            if not self.home_stage(cancel_event):
                return False

        if cancelled() or self.connection is None:
            return False

        self.set_rangelimits(limits)

        if cancelled() or self.connection is None:
            return False

        if startloc:
            if not self.move_abs(start):
                return False

        if cancelled() or self.connection is None:
            return False

        device_list = self.connection.detect_devices()
        for device in device_list:
            if cancelled() or self.connection is None:
                return False
            device.all_axes.wait_until_idle(throw_error_on_fault = True)

        return not cancelled() and self.connection is not None


    def disconnect(self) -> bool:
        """close com port connection."""
        self._disconnecting = True
        stopped = False
        connection = self.connection
        try:
            stopped = self.emergency_stop()
            if not self.stop_position_poller():
                print('Stage position poller did not stop before disconnect')
        finally:
            try:
                if connection is not None:
                    connection.close()
            except Exception as e:
                print(f'Closing stage connection failed: {e}')
            finally:
                self.connection = None
                self.axis_x = None
                self.axis_y = None
                self.axis_z = None
                self.devices = []
                self.state = StageState()
                self._jog_velocity = [0.0, 0.0, 0.0]

        return stopped


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
