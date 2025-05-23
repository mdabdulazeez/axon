

from pylsl import StreamInlet, resolve_byprop
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import style
from collections import deque

last_print = time.time()
fps_counter =  deque(maxlen=150)

# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
# streams = resolve_stream('type', 'EEG')
streams = resolve_byprop('type', 'EEG', minimum=1, timeout=5.0)
#create a new inlet to read from the stream
inlet = StreamInlet(streams[0])

channel_data = {}


for i in range(500): #how many iterations. Eventually this would be a while True

    for i in range(8): #each of the 16 channels here
        sample, timestamp = inlet.pull_sample()
        if i not in channel_data:
            channel_data[i] = sample
        else:
            channel_data[i].append(sample)
    
    fps_counter.append(time.time() - last_print)
    last_print = time.time()
    cur_raw_hz = 1/(sum(fps_counter)/len(fps_counter))
    print(cur_raw_hz)



for chan in channel_data:
    plt.plot(channel_data[chan][:60])
plt.show()