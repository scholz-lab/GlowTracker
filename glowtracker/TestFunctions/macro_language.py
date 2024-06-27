from pyparsing import (
    Group, Suppress, Forward, ZeroOrMore, Keyword, ParserElement, StringEnd, ParseException, common,
    pythonStyleComment, LineEnd, Word, alphas, alphanums, Literal, oneOf, 
    infixNotation, opAssoc, ParseResults
)

from typing import List

import time

class MacroScriptExecutor:
    """Parser and executor for a custom glowtracker macro scripts.

    List of supported commands:
    - move_abs
    - move_rel
    - snap
    - record_for
    - start_recording
    - stop_recording
    - wait

    Flow control:
    - loop

    Features:
    - python style comment using '#'
    - variable assignment using '='. e.g. 'x = 10'
    - simple arithmetic e.g. '(x*2 + 10)^2'


    Visit the wiki for more details: https://scholz-lab.github.io/GlowTracker/

    """

    def __init__(self) -> None:
        self.parser = self._createTextParser()
        self.functionHandle: dict[callable] = {}


    def _createTextParser(self) -> ParserElement:

        # 
        # Variable
        # 
        variable = Word(alphas, alphanums)
        numberOrVariable = common.number | variable

        # 
        # Math
        # 
        signOp = oneOf("+ -")
        multOp = oneOf("* /")
        plusOp = oneOf("+ -")
        expOp = Literal("^")
        factOp = Literal("!")
        modOp = Literal("%")

        arithExpr = infixNotation(
            numberOrVariable,
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
        single_arg = Suppress('(') + arithExpr + Suppress(')')
        coord_arg = Suppress('(') + arithExpr + Suppress(',') + arithExpr + Suppress(',') + arithExpr + Suppress(')')
        loop_arg = arithExpr | ( 
            Suppress('(') + variable + Suppress(':') + arithExpr + Suppress(')') 
        )

        # 
        # Commands
        # 
        move_abs = Group( Keyword('move_abs') + coord_arg )
        move_rel = Group( Keyword('move_rel') + coord_arg )
        snap = Group( Keyword('snap') + empty_arg )
        record_for = Group( Keyword('record_for') + single_arg )
        start_recording = Group( Keyword('start_recording') + empty_arg )
        stop_recording = Group( Keyword('stop_recording') + empty_arg )
        wait = Group( Keyword('wait') + single_arg )
        comment = pythonStyleComment + LineEnd()

        varaible_assignment = Group(
            (variable + Suppress('=') + arithExpr).setParseAction(
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


    def registerFunctionHandler(self,
            move_abs_handle: callable,
            move_rel_handle: callable,
            snap_handle: callable,
            record_for_handle: callable,
            start_recording_handle: callable,
            stop_recording_handle: callable
        ) -> None:
        """Register function handler from the GlowTracker into the MacroScriptExecutor

        Args:
            move_abs_handle (callable): move stage by absolute coordinates
            move_rel_handle (callable): move stage by relative coordinates
            snap_handle (callable): snap an image
            record_for_handle (callable): record an image for a certain amount of time
            start_recording_handle (callable): start camera recording
            stop_recording_handle (callable): stop camera recording
        """
        self.functionHandle['move_abs'] = move_abs_handle
        self.functionHandle['move_rel'] = move_rel_handle
        self.functionHandle['snap'] = snap_handle
        self.functionHandle['record_for'] = record_for_handle
        self.functionHandle['start_recording'] = start_recording_handle
        self.functionHandle['stop_recording'] = stop_recording_handle
    

    def executeScript(self, script: str) -> None:

        parsedCommands = []

        print('Parsing command.')

        try:
            # Parse the script
            parsedCommands = self.parser.parseString(script, parseAll= True)

        except ParseException as e:
            print(f"Parsing error: {e}")
            raise e
        
        # Execute the commands

        print('Executing commands.')
        
        self._executeCommandList(parsedCommands)


    def _executeCommandList(self, commandList: List | ParseResults, scopeVariableDict: dict | None = None) -> None:
        
        if scopeVariableDict is None:
            scopeVariableDict = {}
        
        # Execute the command

        for command in commandList:

            commandName = command[0]

            if commandName == 'move_abs' or commandName == 'move_rel':

                x = self._resolveExpression(scopeVariableDict, command[1])
                y = self._resolveExpression(scopeVariableDict, command[2])
                z = self._resolveExpression(scopeVariableDict, command[3])

                self.functionHandle[commandName](x, y, z)
                
            elif commandName == 'snap' or commandName == 'start_recording' or commandName == 'stop_recording':
                
                self.functionHandle[commandName]()
                
            elif commandName == 'record_for':
                
                recordTime = self._resolveExpression(scopeVariableDict, command[1])
                self.functionHandle[commandName](recordTime)

            elif commandName == 'wait':

                waitTime = self._resolveExpression(scopeVariableDict, command[1])
                time.sleep(waitTime)
            
            elif commandName == 'loop':

                # Only support integer value. A float value will be cast and rounded to int. This need to be on the document.

                numLoop = 0
                loopVariable: str | None = None
                subCommandList = []

                # Parse the arguments
                if len(command) == 3:
                    # One argument loop
                    numLoop = int(self._resolveExpression(scopeVariableDict, command[1]))
                    subCommandList = command[2]

                elif len(command) == 4:
                    # Two argument loop in fashion of a:b
                    numLoop = int(self._resolveExpression(scopeVariableDict, command[2]))
                    # Create a new scope variable
                    loopVariable = command[1]
                    scopeVariableDict[loopVariable] = 0

                    subCommandList = command[3]

                # Execute looping sub commands
                for i in range(numLoop):
                    
                    # Update loop variable
                    if loopVariable is not None:
                        scopeVariableDict[loopVariable] = i
                    # Execute sub commands
                    self._executeCommandList(subCommandList, scopeVariableDict)
            
            elif commandName == 'varaible_assignment':

                variable_name = command[1]
                variable_value = self._resolveExpression(scopeVariableDict, command[2])
                scopeVariableDict[variable_name] = variable_value


    def _resolveExpression(self, scopeVariableDict, expression: str | int | float | ParseResults) -> int | float:

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
                return self._resolveExpression(scopeVariableDict, expression[0])

            elif len(expression) == 2:

                op = expression[0]
                value = self._resolveExpression(scopeVariableDict, expression[1])

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

                left = self._resolveExpression(scopeVariableDict, expression[0])
                op = expression[1]
                right = self._resolveExpression(scopeVariableDict, expression[2])

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


def getMacroScript() -> str:
    # Example script
    script_text = """
    x = 2
    a = x + 2
    wait(x)
    move_abs(1.0,x, 3.0)
    move_rel(0.5, -0.5, x)
    snap()
    record_for(a + -2)
    start_recording()
    wait(2.0)
    # My first loop
    loop(5 + 2*2) {
        # My second loop
        y = 5
        move_abs(1.1, x, 3.1) # My second loop
        wait(y)
        loop(i:3+x/2) {
            snap()
            move_abs(1.2, x + i/2, 3.2)
            wait(i)
        }
    }
    stop_recording()
    """

    return script_text


if __name__ == '__main__':

    macroScript = getMacroScript()

    macroScriptExecutor = MacroScriptExecutor()

    macroScriptExecutor.registerFunctionHandler(
        move_abs_handle= lambda x, y, z: print(f"move_abs({x}, {y}, {z})"),
        move_rel_handle= lambda x, y, z: print(f"move_rel({x}, {y}, {z})"),
        snap_handle= lambda: print("snap()"),
        record_for_handle= lambda x: print(f"record_for({x})"),
        start_recording_handle= lambda: print("start_recording()"),
        stop_recording_handle= lambda: print("stop_recording()")
    )

    macroScriptExecutor.executeScript(macroScript)
