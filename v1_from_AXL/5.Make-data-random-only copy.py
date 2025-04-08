"""Program to show how to read a multi-channel time series from LSL and save it to a file."""

from pylsl import StreamInlet, resolve_byprop #resolve_stream
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import style
from collections import deque
import cv2
import os
import random


ACTION = 'left'
FFT_MAX_HZ = 60

HM_SECONDS = 30  # this is approximate. Not 100% lol.
TOTAL_ITERS = HM_SECONDS * 25  # ~25 iters/sec
BOX_MOVE = "random"  # or model


last_print = time.time()
fps_counter = deque(maxlen=150)

# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_byprop('type', 'EEG', minimum=1, timeout=5.0)# streams = resolve_stream('type', 'EEG')
# create a new inlet to read from the stream
inlet = StreamInlet(streams[0])

WIDTH = 500
HEIGHT = 500
SQ_SIZE = 50
MOVE_SPEED = 1

square = {'x1': int(int(WIDTH) / 2 - int(SQ_SIZE / 2)),
          'x2': int(int(WIDTH) / 2 + int(SQ_SIZE / 2)),
          'y1': int(int(HEIGHT) / 2 - int(SQ_SIZE / 2)),
          'y2': int(int(HEIGHT) / 2 + int(SQ_SIZE / 2))}

box = np.ones((square['y2'] - square['y1'], square['x2'] - square['x1'], 3)) * np.random.uniform(size=(3,))
horizontal_line = np.ones((HEIGHT, 10, 3)) * np.random.uniform(size=(3,))
vertical_line = np.ones((10, WIDTH, 3)) * np.random.uniform(size=(3,))





channel_datas = []


for i in range(TOTAL_ITERS):  # how many iterations. Eventually this would be a while True
    channel_data = []
    for i in range(8):  # each of the 8 channels here
        sample, timestamp = inlet.pull_sample()
        channel_data.append(sample[:FFT_MAX_HZ])

    fps_counter.append(time.time() - last_print)
    last_print = time.time()
    cur_raw_hz = 1 / (sum(fps_counter) / len(fps_counter))
    print(cur_raw_hz)

    env = np.zeros((WIDTH, HEIGHT, 3))

    env[:, HEIGHT // 2 - 5:HEIGHT // 2 + 5, :] = horizontal_line
    env[WIDTH // 2 - 5:WIDTH // 2 + 5, :, :] = vertical_line
    env[square['y1']:square['y2'], square['x1']:square['x2']] = box


    if BOX_MOVE == "random":
        move = random.choice([-1, 0, 1])
        square['x1'] += move
        square['x2'] += move

    cv2.imshow('', env)
    cv2.waitKey(1)

    channel_datas.append(channel_data)


# plt.plot(channel_datas[0][0])
# plt.show()

datadir = "val_data"
if not os.path.exists(datadir):
    os.mkdir(datadir)

actiondir = f"{datadir}/{ACTION}"
if not os.path.exists(actiondir):
    os.mkdir(actiondir)

print(len(channel_datas))

print(f"saving {ACTION} data...")
np.save(os.path.join(actiondir, f"{int(time.time())}.npy"), np.array(channel_datas))
print("done.")