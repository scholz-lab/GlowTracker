import nidaqmx
import numpy as np
import time


def write_command(ref_clock, ref_signal, ref_out, command_in):
    """
    Sends a command to the objective and reads back the reply from the objective to the previous command.

    :param ref_clock: reference to the DAQ channel used to generate clock pulses
    :param ref_signal: reference to the DAQ channel used to send commands
    :param ref_out: reference to the DAQ channel used to read from the objective
    :param command_in: hexadecimal command to execute by the objective (8 bits as a boolean array)
    :return: response from the objective to the previous command (8 bit as a boolean array)

    """
    command = translate_command(command_in)
    # initialize the two channels
    response = np.zeros(8, dtype=bool)
    ref_signal.write(command[0], auto_start=True)

    # send bits new command and read from previous
    for bit in range(8):
        response[bit] = ref_out.read(1)[0]  # reading response to previous command
        ref_clock.write(True, auto_start=True)
        ref_signal.write(command[bit + 1], auto_start=True)  # preparing bit for command to execute
        ref_clock.write(False, auto_start=True)  # Send command
    ref_clock.write(True, auto_start=True)  # clock to idle state
    return response


def translate_command(hex_string):
    """
    Transforms the hexadecimal string input into a boolean array of 8 bit

    :param hex_string: hexadecimal command to execute by the objective
    :return: output_command: boolean array

    """
    boolean_command = bin(int(hex_string, base=16))  # taking reverse as it requires MSB first
    output_command = np.zeros((1, 8 - len(boolean_command) + 2), dtype=bool)
    for number in boolean_command[2:]:
        output_command = np.append(output_command, [bool(int(number))])
    output_command = np.append(output_command, False)  # stop bit
    return output_command


def full_protocol(ref_clock, ref_signal, ref_out, command_in):
    """
    Communication protocol consisting of one command followed by two additional commands 'zero'
    The command '0x00' is used to read out the reply to the objective. Usually, replies from the objective consists
    of a single 8 bit reply, but there are exceptions for some commands. This protocol is longer but is more robust

    :param ref_clock: reference to the DAQ channel used to generate clock pulses
    :param ref_signal: reference to the DAQ channel used to send commands
    :param ref_out: reference to the DAQ channel used to read from the objective
    :param command_in: hexadecimal command to execute by the objective (8 bits as a boolean array)

    :return: response_out: this contains the 2x 8 bit reply from the objective
    """

    command_sequence = command_in.split()  # This is in case the command is complex and is constituted by several bytes
    for command in command_sequence:  # Execute each subcommand before asking for the readout
        response1 = write_command(ref_clock, ref_signal, ref_out, command)
    response2 = write_command(ref_clock, ref_signal, ref_out, '0')
    response3 = write_command(ref_clock, ref_signal, ref_out, '0')
    response_out = np.append(response2, response3)
    return response_out


def align_communication(ref_clock, ref_signal, ref_out):
    """
    Sends the initialization command "0x0A" to the objective and reads the reply. if this is not aligned, generates
    clock pulses until the reply from the objective is the correct one.

    :param ref_clock: reference to the DAQ channel used to generate clock pulses
    :param ref_signal: reference to the DAQ channel used to send commands
    :param ref_out: reference to the DAQ channel used to read from the objective
    :return:
    """
    right_response = False
    expected_response = np.array([True, False, True, False, True, False, True, False,  # First byte
                                  False, False, False, False, False, False, False, False], dtype=bool)  # Second Byte
    while not right_response:
        response = full_protocol(ref_clock, ref_signal, ref_out, 'A')
        print(response)
        if np.all(response == expected_response):
            right_response = True
        else:
            print('wrong response. Shifting one bit')
            # sending one click of the clock
            ref_clock.write(False, auto_start=True)  # Shifting a bit
            ref_clock.write(True, auto_start=True)  # clock to idle state


#%% Initialization of the hardware

clock = nidaqmx.Task()
clock.do_channels.add_do_chan('Dev1/port1/line3')

signal = nidaqmx.Task()
signal.do_channels.add_do_chan('Dev1/port1/line2')

out = nidaqmx.Task()
out.di_channels.add_di_chan('Dev1/port1/line1')

align_communication(clock, signal, out)
#
# command_hex = '6'
# print('For the command:' + command_hex)
# response = full_protocol(clock, signal, out, command_hex)
# print(response)
# time.sleep(0.5)
#
# command_hex = '5'
# print('For the command:' + command_hex)
#
# print(response)
# time.sleep(0.5)
#
#
response = full_protocol(clock, signal, out, '6')
time.sleep(0.5)
complex_command = '44 00 FF'
response = full_protocol(clock, signal, out, complex_command)

clock.close()
signal.close()
out.close()