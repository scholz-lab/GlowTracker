import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import nidaqmx
import cv2

# def data_gen(task,t=0):
#    cnt = 0
#    while cnt < 1000:
#        cnt += 1
#        t += 0.1
#        yield t, np.sin(2*np.pi*t) * np.exp(-t/10.)
def data_gen(task,t=0,MinData=0,MaxData=0,Counter=0):
    while Counter < 100:
        Counter += 1
        t += 0.1
        sample = task.read(number_of_samples_per_channel=1)
        print(sample[0])
        if sample[0] < MinData:
            MinData = sample[0]
        if sample[0] > MaxData:
            MaxData = sample[0]
        k = cv2.waitKey(1) & 0xFF
        # press 'q' to exit
        if k == ord('q'):
            break
        ax.set_ylim(MinData,MaxData)
        yield t, sample[0]

def init():
    del xdata[:]
    del ydata[:]
    line.set_data(xdata, ydata)
    return line,task

def run(data):
    # update the data
    t, y = data
    xdata.append(t)
    ydata.append(y)
    xmin, xmax = ax.get_xlim()

    if t >= xmax:
        ax.set_xlim(xmin, 2*xmax)
        ax.figure.canvas.draw()
    line.set_data(xdata, ydata)

    return line,

fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.grid()
xdata, ydata = [], []
task = nidaqmx.Task()
task.ai_channels.add_ai_voltage_chan("Dev1/ai0")
ani = animation.FuncAnimation(fig, run, data_gen(task), blit=False, interval=10,
                            repeat=False, init_func=init)
plt.show()
