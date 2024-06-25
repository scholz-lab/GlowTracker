from pyparsing import (
    Group, Suppress, Forward, ZeroOrMore, Keyword, ParserElement, StringEnd, ParseException, common,
    pythonStyleComment, LineEnd, Word, alphas, alphanums
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
    # Define Syntax
    # 
    
    # Variable
    variable_name = Word(alphas, alphanums)
    intOrVariable = common.integer | variable_name
    numberOrVariable = common.number | variable_name

    # Arguments
    empty_arg = Suppress('()')
    intOrVariable_arg = Suppress('(') + intOrVariable + Suppress(')')
    numberOrVariable_arg = Suppress('(') + numberOrVariable + Suppress(')')
    coord_args = Suppress('(') + numberOrVariable + Suppress(',') + numberOrVariable + Suppress(',') + numberOrVariable + Suppress(')')

    # Commands
    move_abs = Group( Keyword('move_abs') + coord_args )
    move_rel = Group( Keyword('move_rel') + coord_args )
    snap = Group( Keyword('snap') + empty_arg )
    record_for = Group( Keyword('record_for') + numberOrVariable_arg )
    start_recording = Group( Keyword('start_recording') + empty_arg )
    stop_recording = Group( Keyword('stop_recording') + empty_arg )
    wait = Group( Keyword('wait') + numberOrVariable_arg )
    comment = pythonStyleComment + LineEnd()
    varaible_assignment = Group(
        (variable_name + Suppress('=') + numberOrVariable).setParseAction(
            lambda t: ["varaible_assignment", t[0], t[1]]
        )
    )

    # Forward declaration for nested loops
    command = Forward()

    # Define loop command with nested commands
    loop = Group( 
        Keyword("loop") 
        + ( 
            intOrVariable_arg 
            | 
            ( Suppress('(') + variable_name + Suppress(':') + intOrVariable + Suppress(')') ) 
        )
        + Suppress('{') + Group(ZeroOrMore(command)) + Suppress('}') 
    )

    # Finally define command which included loop
    command <<= (
        move_abs | move_rel | snap | record_for | start_recording |
        stop_recording | wait | comment | loop | varaible_assignment
    )

    # Top-level parser
    parser = ZeroOrMore(command | comment) + StringEnd()

    # Ignore space, tabs, return, newline
    parser.setDefaultWhitespaceChars(' \t')

    return parser


def getMacroScript() -> str:
    # Example script
    script_text = """
    x = 2
    wait(x)
    move_abs(1.0,x, 3.0)
    move_rel(0.5, -0.5, x)
    snap()
    record_for(3.5)
    start_recording()
    wait(2.0)
    # My first loop
    loop(5) {
        # My second loop
        y = 5
        move_abs(1.1, x, 3.1) # My second loop
        wait(y)
        loop(i:3) {
            snap()
            wait(i)
        }
    }
    stop_recording()
    """

    return script_text


def executeCommandList(commandList: List, stage: Stage, camera: Camera, scopeVariableDict: dict | None = None) -> None:
    
    if scopeVariableDict is None:
        scopeVariableDict = {}

    def resolveValue(value: str | int | float) -> int | float:
        if isinstance(value, str):
            if value in scopeVariableDict:
                return scopeVariableDict[value]
            else:
                raise Exception(f"The variable '{value}' is undefined.")
        else:
            return value

    
    # Execute the command

    for command in commandList:

        commandName = command[0]

        if commandName == 'move_abs':

            [x, y, z] = list(map(resolveValue, command[1:4]))
            print(f'Move absolute for {x, y, z}')
            stage.move_abs((x, y, z), 'um', wait_until_idle= True)
            
        elif commandName == 'move_rel':

            [x, y, z] = list(map(resolveValue, command[1:4]))
            print(f'Move relative for {x, y, z}')
            stage.move_rel((x, y, z), 'um', wait_until_idle= True)
            
        elif commandName == 'snap':
            
            print('Snap an image')
            # Call capture an image
            isSuccess, img = camera.singleTake()

            if isSuccess:
                saveImage(img, 'record', 'image1.tiff')
            
        elif commandName == 'record_for':
            
            recordTime = resolveValue(command[1])
            print(f'Record for {recordTime} sec')
            camera.StartGrabbingMax(int(recordTime), pylon.GrabStrategy_OneByOne)
            
        elif commandName == 'start_recording':

            print('Start recording')
            camera.StartGrabbing(pylon.GrabStrategy_OneByOne)
            
        elif commandName == 'stop_recording':
            
            print('Stop recording')
            camera.StopGrabbing()

        elif commandName == 'wait':

            waitTime = resolveValue(command[1])
            print(f'Wait for {waitTime} sec')
            time.sleep(waitTime)
        
        elif commandName == 'loop':

            # Only support integer value. A float value will be cast and rounded to int. This need to be on the document.

            numLoop = 0
            loopVariable: str | None = None
            subCommandList = []

            # Parse the arguments
            if len(command) == 3:
                # One argument loop
                numLoop = int(resolveValue(command[1]))
                subCommandList = command[2]

            elif len(command) == 4:
                # Two argument loop in fashion of a:b
                numLoop = int(resolveValue(command[2]))
                # Create a new scope variable
                loopVariable = command[1]
                scopeVariableDict[loopVariable] = 0

                subCommandList = command[3]

            # Execute looping sub commands
            for i in range(numLoop):
                
                print(f'Loop {i}')
                # Update loop variable
                if loopVariable is not None:
                    scopeVariableDict[loopVariable] = i
                # Execute sub commands
                executeCommandList(subCommandList, stage, camera, scopeVariableDict)
        
        elif commandName == 'varaible_assignment':

            variable_name = command[1]
            variable_value = resolveValue(command[2])
            print(f'Assign {variable_name} = {variable_value}')
            scopeVariableDict[variable_name] = variable_value


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
        exit(0)
    
    print('Parsed command successful.')

    # Execute the commands
    executeCommandList(parsedCommands, stage, camera)
