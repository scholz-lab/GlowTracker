from pyparsing import (
    Group, Suppress, Forward, ZeroOrMore, Keyword, ParserElement, StringEnd, ParseException, common
)

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

    # Forward declaration for nested loops
    command = Forward()

    # Define loop command with nested commands
    loop = Group( Keyword("loop") + int_arg + Suppress('{') + Group(ZeroOrMore(command)) + Suppress('}') )

    # Finally define command which included loop
    command <<= (
        move_abs | move_rel | snap | record_for | start_recording |
        stop_recording | wait | loop
    )

    # Top-level parser
    parser = ZeroOrMore(command) + StringEnd()

    # Ignore space, tabs, return, newline
    parser.setDefaultWhitespaceChars(' \t\r\n')

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
    loop(5) {
        move_abs(1.1, 2.1, 3.1)
        wait(1.0)
        loop(3) {
            snap()
            wait(0.5)
        }
    }
    stop_recording()
    """

    return script_text


def executeCommandList(commandList) -> None:
    
    for command in commandList:
        
        commandName = command[0]

        if commandName == 'move_abs':

            [x, y, z] = command[1:4]
            print(f'Move absolute for {x, y, z}')
            
        elif commandName == 'move_rel':

            [x, y, z] = command[1:4]
            print(f'Move relative for {x, y, z}')
            
        elif commandName == 'snap':
            
            print('Snap an image')
            
        elif commandName == 'record_for':
            
            recordTime = command[1]
            print(f'Record for {recordTime} sec')
            
        elif commandName == 'start_recording':

            print('Start recording')
            
        elif commandName == 'stop_recording':
            
            print('Stop recording')

        elif commandName == 'wait':

            waitTime = command[1]
            print(f'Wait for {waitTime} sec')
        
        elif commandName == 'loop':

            numLoop = command[1]
            subCommandList = command[2]

            for i in range(numLoop):
                
                print(f'Loop {i}')
                executeCommandList(subCommandList)



if __name__ == '__main__':

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

    for command in parsedCommands:
        print(command)

    executeCommandList(parsedCommands)
