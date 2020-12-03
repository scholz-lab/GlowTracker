# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 12:11:51 2020

@authors: Hofmann, Scholz, Alvarez
"""
import kivy
kivy.require('1.11.1')
from kivy.app import App
from kivy.core.window import Window
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.popup import Popup
from zaber_motion import Library, Units
from zaber_motion.ascii import Connection
import Zaber_control as stg
import Macroscope_macros as macro

Library.toggle_device_db_store(True)  # This is part of the Zaber initialization. Unclear what it does.
# TODO: separate stage from camera in the programs
# TODO: Structure the GUI such that it shares variables across GUI components
# TODO: Implement new layout

Window.size = (1600, 600)
# MenuBarWidget is the main GUI
class MenuBarWidget(Widget):
    def __init__(self, **kwargs):
        super(MenuBarWidget, self).__init__(**kwargs)

    def home(self):
        stg.home_stage()

    def exit_app(self):
        kivy.base.stopTouchApp()


class AxisLimitsPopup(Popup):
    def __init__(self, **kwargs):
        super(AxisLimitsPopup, self).__init__(**kwargs)
    #TODO!: Add function that changes entered new limit in stage script fct set_rangelimits()

class RecordingSettingsPopup(Popup):
    def __init__(self, **kwargs):
        super(RecordingSettingsPopup, self).__init__(**kwargs)
    #TODO!: Add functions looking up according strings in camera file and changing following values
    #(exposure time, gain)
    #to change number of frames to record: change while loop limit in stg.scripts
    
class FunctionCallsWidget(Widget):
    def __init__(self, **kwargs):
        super(FunctionCallsWidget, self).__init__(**kwargs)
    
    def calibrate_stage(self):
        macro.stageCalibration()

    def center(self):
        stg.center_once()
        
    def autofocus(self):
        macro.zFocus()
    
    def recording(self):
        macro.trackworm()


class ControlsWidget(Widget):
    def __init__(self, **kwargs):
        super(ControlsWidget, self).__init__(**kwargs)


class XControls(Widget):
    def __init__(self, **kwargs):
        super(XControls, self).__init__(**kwargs)
    
    def moveX_bigStepLeft(self):
        with Connection.open_serial_port('COM3') as connection:
            connection.renumber_devices(first_address=1)
            device_list = connection.detect_devices()
            device1 = device_list[0]
            axis_x = device1.get_axis(1)
            axis_x.move_relative(-50, Units.LENGTH_MICROMETRES, wait_until_idle=False)

    def moveX_smallStepLeft(self):
        with Connection.open_serial_port('COM3') as connection:
            connection.renumber_devices(first_address=1)
            device_list = connection.detect_devices()
            device1 = device_list[0]
            axis_x = device1.get_axis(1)
            axis_x.move_relative(-20, Units.LENGTH_MICROMETRES, wait_until_idle=False)

    def moveX_smallStepRight(self):
        with Connection.open_serial_port('COM3') as connection:
            connection.renumber_devices(first_address=1)
            device_list = connection.detect_devices()
            device1 = device_list[0]
            axis_x = device1.get_axis(1)
            axis_x.move_relative(20, Units.LENGTH_MICROMETRES, wait_until_idle=False)

    def moveX_bigStepRight(self):
        with Connection.open_serial_port('COM3') as connection:
            connection.renumber_devices(first_address=1)
            device_list = connection.detect_devices()
            device1 = device_list[0]
            axis_x = device1.get_axis(1)
            axis_x.move_relative(50, Units.LENGTH_MICROMETRES, wait_until_idle=False)

class YControls(Widget):
    def __init__(self, **kwargs):
        super(YControls, self).__init__(**kwargs)

    def moveY_bigStepLeft(self):
        with Connection.open_serial_port('COM3') as connection:
            connection.renumber_devices(first_address=1)
            device_list = connection.detect_devices()
            device2 = device_list[1]
            axis_y = device2.get_axis(1)
            axis_y.move_relative(-50, Units.LENGTH_MICROMETRES, wait_until_idle=False)

    def moveY_smallStepLeft(self):
        with Connection.open_serial_port('COM3') as connection:
            connection.renumber_devices(first_address=1)
            device_list = connection.detect_devices()
            device2 = device_list[1]
            axis_y = device2.get_axis(1)
            axis_y.move_relative(-20, Units.LENGTH_MICROMETRES, wait_until_idle=False)

    def moveY_smallStepRight(self):
        with Connection.open_serial_port('COM3') as connection:
            connection.renumber_devices(first_address=1)
            device_list = connection.detect_devices()
            device2 = device_list[1]
            axis_y = device2.get_axis(1)
            axis_y.move_relative(20, Units.LENGTH_MICROMETRES, wait_until_idle=False)

    def moveY_bigStepRight(self):
        with Connection.open_serial_port('COM3') as connection:
            connection.renumber_devices(first_address=1)
            device_list = connection.detect_devices()
            device2 = device_list[1]
            axis_y = device2.get_axis(1)
            axis_y.move_relative(50, Units.LENGTH_MICROMETRES, wait_until_idle=False)

class ZControls(Widget):
    def __init__(self, **kwargs):
        super(ZControls, self).__init__(**kwargs)

    def moveZ_bigStepLeft(self):
        with Connection.open_serial_port('COM3') as connection:
            connection.renumber_devices(first_address=1)
            device_list = connection.detect_devices()
            if len(device_list) > 2:
                device3 = device_list[2]
                axis_z = device3.get_axis(1)
                axis_z.move_relative(-50, Units.LENGTH_MICROMETRES, wait_until_idle=False)
            else:
                print('No Z axis detected')

    def moveZ_smallStepLeft(self):
        with Connection.open_serial_port('COM3') as connection:
            connection.renumber_devices(first_address=1)
            device_list = connection.detect_devices()
            if len(device_list) > 2:
                device3 = device_list[2]
                axis_z = device3.get_axis(1)
                axis_z.move_relative(-20, Units.LENGTH_MICROMETRES, wait_until_idle=False)
            else:
                print('No Z axis detected')

    def moveZ_smallStepRight(self):
        with Connection.open_serial_port('COM3') as connection:
            connection.renumber_devices(first_address=1)
            device_list = connection.detect_devices()
            if len(device_list) > 2:
                device3 = device_list[2]
                axis_z = device3.get_axis(1)
                axis_z.move_relative(20, Units.LENGTH_MICROMETRES, wait_until_idle=False)
            else:
                print('No Z axis detected')

    def moveZ_bigStepRight(self):
        with Connection.open_serial_port('COM3') as connection:
            connection.renumber_devices(first_address=1)
            device_list = connection.detect_devices()
            if len(device_list) > 2:
                device3 = device_list[2]
                axis_z = device3.get_axis(1)
                axis_z.move_relative(50, Units.LENGTH_MICROMETRES, wait_until_idle=False)
            else:
                print('No Z axis detected')

class FunctionalBodyWidget(BoxLayout):
    def __init__(self, **kwargs):
        super(FunctionalBodyWidget, self).__init__(**kwargs)


class OverallMenu(GridLayout):
    def __init__(self, **kwargs):
        super(OverallMenu, self).__init__(**kwargs)


class CameraWidget(BoxLayout):
    def __init__(self, **kwargs):
        super(CameraWidget, self).__init__(**kwargs)


class MainBoxWidget(GridLayout):
    def __init__(self, **kwargs):
        super(MainBoxWidget, self).__init__(**kwargs)
        

class MacroscopeApp(App):
    # This is used by kivy as the entry point for the definitions of the UI. It is standard. It needs to use the App as
    # input and its name has to end with App. The actual "meat" of the widgets and what do they do and their layout is
    # in another file that must have the same name but low case, without the app at the end and .kv extension
    def build(self):
        return MainBoxWidget()  # This is the actual GUI


def reset():
    # Cleaner for the events in memory
    import kivy.core.window as window
    from kivy.base import EventLoop
    if not EventLoop.event_listeners:
        from kivy.cache import Cache
        window.Window = window.core_select_lib('window', window.window_impl, True)
        Cache.print_usage()
        for cat in Cache._categories:
            Cache._objects[cat] = {}
     
        
if __name__ == '__main__':
    reset()
    # TODO: connection to the decive and error handling is done here
    MacroscopeApp().run()  # This runs the App in an endless loop until it closes. At this point it will execute the code below
    # TODO: disconnect here from the active devices on closure of the GUI
