import cannon_communication_protocol as canon
import time

clock, signal, out = canon.initialize_DAQ()
canon.align_communication(clock, signal, out)

canon.full_protocol(clock, signal, out, '6')
time.sleep(0.4)

canon.full_protocol(clock, signal, out, '5')
time.sleep(0.4)

canon.full_protocol(clock, signal, out, '6')
time.sleep(0.4)

complex_command = '44 00 22'
for Index in range(25):  # Move full range in 25 steps
    canon.full_protocol(clock, signal, out, complex_command)
    time.sleep(0.1)

canon.close_DAQ(clock, signal, out)







