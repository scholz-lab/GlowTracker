from pyparsing import (
    Word, alphas, alphanums, nums, delimitedList, Group, 
    Suppress, Forward, Optional, oneOf, ZeroOrMore, infixNotation, opAssoc, Keyword, Combine, ParserElement, StringEnd, ParseException, common
)

# Ignore space, tabs, return, newline
ParserElement.setDefaultWhitespaceChars(' \t\r\n')

# Define arguments
empty_arg = Suppress('()')
int_arg = Suppress('(') + common.integer + Suppress(')')
float_arg = Suppress('(') + common.real + Suppress(')')
number_arg = Suppress('(') + common.number + Suppress(')')
coord_args = Suppress('(') + delimitedList(common.number) + Suppress(')')

# Define commands
move_abs = Keyword('move_abs') + coord_args
move_rel = Keyword('move_rel') + coord_args
snap = Keyword('snap') + empty_arg
record_for = Keyword('record_for') + number_arg
start_recording = Keyword('start_recording') + empty_arg
stop_recording = Keyword('stop_recording') + empty_arg
wait = Keyword('wait') + number_arg

# Forward declaration for nested loops
command = Forward()

# Define loop command with nested commands
loop = Keyword("loop") + int_arg + Suppress('{') + Group(ZeroOrMore(command)) + Suppress('}')

# Finally define command which included loop
command <<= (
    move_abs | move_rel | snap | record_for | start_recording |
    stop_recording | wait | loop
)

# Top-level parser
script = ZeroOrMore(command) + StringEnd()

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

try:
    # Parse the script
    parsed_script = script.parseString(script_text, parseAll= True)

    # Print parsed result
    for cmd in parsed_script:
        print(cmd)

except ParseException as pe:
    print(f"Parsing error: {pe}")