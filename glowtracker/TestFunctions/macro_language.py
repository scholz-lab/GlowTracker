from pyparsing import (
    Group, Suppress, Forward, ZeroOrMore, Keyword, ParserElement, StringEnd, ParseException, common,
    pythonStyleComment, LineEnd, Word, alphas, alphanums, Literal, oneOf, 
    infixNotation, opAssoc, ParseResults
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
    # Variable
    # 
    variable_name = Word(alphas, alphanums)
    intOrVariable = common.integer | variable_name
    numberOrVariable = common.number | variable_name

    # 
    # Math
    # 
    operand = numberOrVariable

    signOp = oneOf("+ -")
    multOp = oneOf("* /")
    plusOp = oneOf("+ -")
    expOp = Literal("^")
    factOp = Literal("!")
    modOp = Literal("%")

    arithExpr = infixNotation(
        operand,
        [
            (signOp, 1, opAssoc.RIGHT),
            (multOp, 2, opAssoc.LEFT),
            (plusOp, 2, opAssoc.LEFT),
            (expOp, 2, opAssoc.RIGHT),
            (factOp, 1, opAssoc.LEFT),
            (modOp, 2, opAssoc.LEFT),
        ],
    )

    # 
    # Arguments
    # 
    empty_arg = Suppress('()')
    intOrVariable_arg = Suppress('(') + intOrVariable + Suppress(')')
    numberOrVariable_arg = Suppress('(') + numberOrVariable + Suppress(')')
    coord_arg = Suppress('(') + numberOrVariable + Suppress(',') + numberOrVariable + Suppress(',') + numberOrVariable + Suppress(')')
    loop_arg = intOrVariable_arg | ( Suppress('(') + variable_name + Suppress(':') + intOrVariable + Suppress(')') )

    # 
    # Commands
    # 
    move_abs = Group( Keyword('move_abs') + coord_arg )
    move_rel = Group( Keyword('move_rel') + coord_arg )
    snap = Group( Keyword('snap') + empty_arg )
    record_for = Group( Keyword('record_for') + numberOrVariable_arg )
    start_recording = Group( Keyword('start_recording') + empty_arg )
    stop_recording = Group( Keyword('stop_recording') + empty_arg )
    wait = Group( Keyword('wait') + numberOrVariable_arg )
    comment = pythonStyleComment + LineEnd()

    varaible_assignment = Group(
        (variable_name + Suppress('=') + arithExpr).setParseAction(
            lambda t: ['varaible_assignment', t[0], t[1:]]
        )
    )

    # 
    # Loop
    # 

    #   Forward declaration of command list for nested loops
    command = Forward()

    #   Define loop command with nested commands
    loop = Group( 
        Keyword("loop") + loop_arg
        + Suppress('{') + Group(ZeroOrMore(command)) + Suppress('}') 
    )

    #   Finally define command which included loop
    command <<= (
        move_abs | move_rel | snap | record_for | start_recording |
        stop_recording | wait | comment | loop | varaible_assignment
    )

    # Create a parser
    parser = ZeroOrMore(command | comment) + StringEnd()

    # 
    # Configure
    # 
    
    # Ignore space, tabs, return, newline
    parser.setDefaultWhitespaceChars(' \t')

    # Enable cache
    parser.enablePackrat()

    return parser


def getMacroScript() -> str:
    # Example script
    script_text = """
    x = 2
    a = x + 2
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


def executeCommandList(commandList: List, scopeVariableDict: dict | None = None) -> None:
    
    if scopeVariableDict is None:
        scopeVariableDict = {}
    
    def resolveExpression(expression: str | int | float) -> int | float:

        if isinstance(expression, (int, float)):
            # Integer or floating point
            return expression
        
        elif isinstance(expression, str):
            # Variable
            if expression in scopeVariableDict:
                return scopeVariableDict[expression]
            else:
                raise ValueError(f"The variable '{expression}' is undefined.")

        elif isinstance(expression, (list, ParseResults)):
            # Arithmetic expression

            if len(expression) == 1:
                # Parenthesis with one variable
                return resolveExpression(expression[0])

            elif len(expression) == 2:

                op = expression(0)
                value = resolveExpression(expression[1])

                # Sign operand
                if op == '+':
                    return value
                elif op == '-':
                    return -value
                
                # Factorial
                elif op == '!':
                    return value
                
                else:
                    raise ValueError(f"Expression {expression} is invalid.")
            
            else:

                left = resolveExpression(expression[0])
                op = expression[1]
                right = resolveExpression(expression[2])

                # Plus, Minus
                if op == '+':
                    return left + right
                elif op == '-':
                    return left - right

                # Multiplication, division
                elif op == '*':
                    return left * right                
                elif op == '/':
                    return left / right
                
                # Modulo
                elif op == '%':
                    return left % right
                    
                # Exponent
                elif op == '^':
                    return left ** right
                
        else:
            raise ValueError(f"Expression {expression} is invalid.")
    

    # Execute the command

    for command in commandList:

        commandName = command[0]

        if commandName == 'move_abs':

            [x, y, z] = list(map(resolveExpression, command[1:4]))
            print(f'Move absolute for {x, y, z}')
            
        elif commandName == 'move_rel':

            [x, y, z] = list(map(resolveExpression, command[1:4]))
            print(f'Move relative for {x, y, z}')
            
        elif commandName == 'snap':
            
            print('Snap an image')
            
        elif commandName == 'record_for':
            
            recordTime = resolveExpression(command[1])
            print(f'Record for {recordTime} sec')
            
        elif commandName == 'start_recording':

            print('Start recording')
            
        elif commandName == 'stop_recording':
            
            print('Stop recording')

        elif commandName == 'wait':

            waitTime = resolveExpression(command[1])
            print(f'Wait for {waitTime} sec')
        
        elif commandName == 'loop':

            # Only support integer value. A float value will be cast and rounded to int. This need to be on the document.

            numLoop = 0
            loopVariable: str | None = None
            subCommandList = []

            # Parse the arguments
            if len(command) == 3:
                # One argument loop
                numLoop = int(resolveExpression(command[1]))
                subCommandList = command[2]

            elif len(command) == 4:
                # Two argument loop in fashion of a:b
                numLoop = int(resolveExpression(command[2]))
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
            variable_value = resolveExpression(command[2])
            print(f'Assign {variable_name} = {variable_value}')
            scopeVariableDict[variable_name] = variable_value


if __name__ == '__main__':


    macroParser = createTextParser()

    macroScript = getMacroScript()

    parsedCommands = []

    try:
        # Parse the script
        parsedCommands = macroParser.parseString(macroScript, parseAll= True)

    except ParseException as pe:
        print(f"Parsing error: {pe}")
        exit(0)
    
    print('Parsed command.')

    # Execute the commands
    executeCommandList(parsedCommands)
