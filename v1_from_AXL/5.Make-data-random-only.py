"""Program to read a multi-channel time series from LSL and save it to a file with cycle-based actions."""

from pylsl import StreamInlet, resolve_byprop
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib import style
from collections import deque
import cv2
import os
import random
import datetime  # Added for better date/time formatting

# Actions and timing
ACTIONS = ["left", "right", "none"]
ACTION_TIMES = [10, 10, 20]  # seconds for each action
ITERATIONS_PER_SECOND = 25  # ~25 iters/sec

FFT_MAX_HZ = 60
NUM_CHANNELS = 8  # Number of channels received from the EEG stream
TARGET_CHANNELS = 16  # Number of channels expected by the model

last_print = time.time()
fps_counter = deque(maxlen=150)

# first resolve an EEG stream on the lab network
print("looking for an EEG stream...")
streams = resolve_byprop('type', 'EEG', minimum=1, timeout=5.0)
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

# Fixed colors: white background, black box, grey lines
white = np.ones((1, 1, 3))  # White (1,1,1)
black = np.zeros((square['y2'] - square['y1'], square['x2'] - square['x1'], 3))  # Black (0,0,0)
grey = np.ones((1, 1, 3)) * 0.5  # Grey (0.5,0.5,0.5)

# Create grey lines
horizontal_line = np.ones((HEIGHT, 10, 3)) * 0.5  # Grey
vertical_line = np.ones((10, WIDTH, 3)) * 0.5  # Grey

# Dictionary to store data for each action
action_data = {"left": [], "right": [], "none": []}

# Run the action cycle
for action_idx, action in enumerate(ACTIONS):
    print(f"Starting action: {action} for {ACTION_TIMES[action_idx]} seconds")
    start_time = time.time()
    action_end_time = start_time + ACTION_TIMES[action_idx]
    
    # Reset box position when we start a new action
    square = {'x1': int(int(WIDTH) / 2 - int(SQ_SIZE / 2)),
              'x2': int(int(WIDTH) / 2 + int(SQ_SIZE / 2)),
              'y1': int(int(HEIGHT) / 2 - int(SQ_SIZE / 2)),
              'y2': int(int(HEIGHT) / 2 + int(SQ_SIZE / 2))}
    
    while time.time() < action_end_time:
        channel_data = []
        for i in range(NUM_CHANNELS):  # each of the channels from the EEG stream
            sample, timestamp = inlet.pull_sample()
            channel_data.append(sample[:FFT_MAX_HZ])

        # For compatibility with 16-channel model, we'll pad with zeros
        # if we need to use 16 channels for the model later
        # Uncomment this if you want to store 16 channels directly
        # while len(channel_data) < TARGET_CHANNELS:
        #     channel_data.append(np.zeros(FFT_MAX_HZ))

        # FPS calculation
        fps_counter.append(time.time() - last_print)
        last_print = time.time()
        cur_raw_hz = 1 / (sum(fps_counter) / len(fps_counter))
        print(f"Action: {action}, FPS: {cur_raw_hz:.2f}")

        # White environment
        env = np.ones((WIDTH, HEIGHT, 3))

        # Grey lines
        env[:, HEIGHT // 2 - 5:HEIGHT // 2 + 5, :] = horizontal_line
        env[WIDTH // 2 - 5:WIDTH // 2 + 5, :, :] = vertical_line
        
        # Black box
        env[square['y1']:square['y2'], square['x1']:square['x2']] = black

        # Move box based on the current action
        if action == "left":
            square['x1'] -= MOVE_SPEED
            square['x2'] -= MOVE_SPEED
        elif action == "right":
            square['x1'] += MOVE_SPEED
            square['x2'] += MOVE_SPEED
        # For "none", the box stays still
        
        # Keep the box within bounds
        if square['x1'] < 0:
            diff = 0 - square['x1']
            square['x1'] += diff
            square['x2'] += diff
        elif square['x2'] > WIDTH:
            diff = square['x2'] - WIDTH
            square['x1'] -= diff
            square['x2'] -= diff

        cv2.imshow('', env)
        cv2.waitKey(1)

        # Store data for the current action
        action_data[action].append(channel_data)

# Save data for each action
datadir = "val_data"
if not os.path.exists(datadir):
    os.mkdir(datadir)

for action in ACTIONS:
    if len(action_data[action]) > 0:
        actiondir = f"{datadir}/{action}"
        if not os.path.exists(actiondir):
            os.mkdir(actiondir)
        
        # Format timestamp as YYYY-MM-DD-HHMMSS
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
        print(f"Saving {action} data ({len(action_data[action])} samples)...")
        np.save(os.path.join(actiondir, f"{timestamp}.npy"), np.array(action_data[action]))
        print(f"Saved {action} data.")

print("Data collection complete.")