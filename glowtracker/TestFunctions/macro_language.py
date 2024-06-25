from pyparsing import (
    Group, Suppress, Forward, ZeroOrMore, Keyword, ParserElement, StringEnd, ParseException, common,
    pythonStyleComment, LineEnd
)

import sys
sys.path.append('../glowtracker')

from typing import List

from Zaber_control import Stage

from Basler_control import Camera, saveImage

from pypylon import pylon

import time

def createTextParser() -> ParserElement:

    # 
    # Define Grammar
    # 
    # Arguments
    empty_arg = Suppress('()')
    int_arg = Suppress('(') + common.integer + Suppress(')')
    number_arg = Suppress('(') + common.number + Suppress(')')
    coord_args = Suppress('(') + common.number + Suppress(',') + common.number + Suppress(',') + common.number + Suppress(')')

    # Commands
    move_abs = Group( Keyword('move_abs') + coord_args )
    move_rel = Group( Keyword('move_rel') + coord_args )
    snap = Group( Keyword('snap') + empty_arg )
    record_for = Group( Keyword('record_for') + number_arg )
    start_recording = Group( Keyword('start_recording') + empty_arg )
    stop_recording = Group( Keyword('stop_recording') + empty_arg )
    wait = Group( Keyword('wait') + number_arg )
    comment = pythonStyleComment + LineEnd()

    # Forward declaration for nested loops
    command = Forward()

    # Define loop command with nested commands
    loop = Group( Keyword("loop") + int_arg + Suppress('{') + Group(ZeroOrMore(command)) + Suppress('}') )

    # Finally define command which included loop
    command <<= (
        move_abs | move_rel | snap | record_for | start_recording |
        stop_recording | wait | comment | loop
    )

    # Top-level parser
    parser = ZeroOrMore(command | comment) + StringEnd()

    # Ignore space, tabs, return, newline
    parser.setDefaultWhitespaceChars(' \t')

    return parser


def getMacroScript() -> str:
    # Example script
    script_text = """
    move_abs(1.0, 2.0, 3.0)
    move_rel(0.5, -0.5, 0)
    snap()
    record_for(3.5)
    start_recording()
    wait(2.0)
    # My first loop
    loop(5) {
        # My second loop
        move_abs(1.1, 2.1, 3.1) # My second loop
        wait(1.0)
        loop(3) {
            snap()
            wait(0.5)
        }
    }
    stop_recording()
    """

    return script_text


def executeCommandList(commandList: List, stage: Stage, camera: Camera) -> None:
    
    for command in commandList:
        
        commandName = command[0]

        if commandName == 'move_abs':

            [x, y, z] = command[1:4]
            print(f'Move absolute for {x, y, z}')
            stage.move_abs((x, y, z), 'um', wait_until_idle= True)
            
        elif commandName == 'move_rel':

            [x, y, z] = command[1:4]
            print(f'Move relative for {x, y, z}')
            stage.move_rel((x, y, z), 'um', wait_until_idle= True)
            
        elif commandName == 'snap':
            
            print('Snap an image')
            # Call capture an image
            isSuccess, img = camera.singleTake()

            if isSuccess:
                saveImage(img, 'record', 'image1.tiff')
            
        elif commandName == 'record_for':
            
            recordTime = command[1]
            print(f'Record for {recordTime} sec')
            camera.StartGrabbingMax(int(recordTime), pylon.GrabStrategy_OneByOne)
            
        elif commandName == 'start_recording':

            print('Start recording')
            camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
            
        elif commandName == 'stop_recording':
            
            print('Stop recording')
            camera.StopGrabbing()

        elif commandName == 'wait':

            waitTime = command[1]
            print(f'Wait for {waitTime} sec')
            time.sleep(waitTime)
        
        elif commandName == 'loop':

            numLoop = command[1]
            subCommandList = command[2]

            for i in range(numLoop):
                
                print(f'Loop {i}')
                executeCommandList(subCommandList, stage, camera)


if __name__ == '__main__':

    # Connect to stage
    stage = Stage('COM4')
    if stage.connection is None:
        print("Can't connect to a stage")
        exit
    else:
        print('Successfully connnect to a stage')

    # Connect to camera
    camera = Camera.createAndConnectCamera()
    if camera is None:
        print("Can't connect to a camera")
        exit
    else:
        print('Successfully connect to a stage')

    macroParser = createTextParser()

    macroScript = getMacroScript()

    parsedCommands = []

    try:
        # Parse the script
        parsedCommands = macroParser.parseString(macroScript, parseAll= True)

    except ParseException as pe:
        print(f"Parsing error: {pe}")
        exit
    
    print('Parsed command successful.')

    # Execute the commands
    executeCommandList(parsedCommands, stage, camera)
