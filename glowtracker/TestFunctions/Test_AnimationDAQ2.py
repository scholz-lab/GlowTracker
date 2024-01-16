import numpy as np
import matplotlib.pyplot as plt
import nidaqmx
import time
import sys

def press(event):
    global stop_acquisition
    print('press', event.key)
    sys.stdout.flush()
    if event.key == 'x':
        stop_acquisition = True
        print('Hola')


def update_line(hl, elapsedTime, voltage):
    hl.set_xdata(np.append(hl.get_xdata(), elapsedTime))
    hl.set_ydata(np.append(hl.get_ydata(), voltage))
    plt.draw()


task = nidaqmx.Task()
task.ai_channels.add_ai_voltage_chan("Dev1/ai0")
fig, ax = plt.subplots()
hl, = plt.plot([], [])
fig.canvas.mpl_connect('key_press_event', press)

t = time.time()
Counter = 0
MinData = 0
MaxData = 0

# Global scope
stop_acquisition = False
while Counter < 200 and not stop_acquisition:
    Counter += 1
    print(Counter)
    sample = task.read(number_of_samples_per_channel=1)
    if sample[0] < MinData:
        MinData = sample[0]
    if sample[0] > MaxData:
        MaxData = sample[0]
    elapsed = time.time() - t
    update_line(hl, elapsed, sample[0])
    ax.set_ylim(MinData, MaxData)
    ax.set_xlim(0, elapsed)
    # press 'x' to exit
    fig.canvas.flush_events()

task.close()





