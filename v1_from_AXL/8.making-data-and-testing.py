"""
REQUIRES the OpenBCI GUI, Networking Tab, and choose
***125 channels*** for the FFT
"""

from pylsl import StreamInlet, resolve_byprop
import numpy as np
import time
from matplotlib import style
from collections import deque
import cv2
import os
import random
import tensorflow as tf

MODEL_NAME = "/home/zee/Documents/AXL/old_models/32x1-5epoch-1738402862-acc-1.0-loss-0.0.keras"
# "/home/zee/Documents/AXL/models/32x1-10epoch-1742193697-acc-1.0-loss-0.0.keras"
# r"C:\Users\Chat Noir\Desktop\AXL\models\32x1-5epoch-1738393074-acc-1.0-loss-0.0.keras"
# "C:\Users\Chat Noir\Desktop\AXL\models\32x1-5epoch-1738391525-acc-1.0-loss-0.01.keras"

# Load and warm up the model
model = tf.keras.models.load_model(MODEL_NAME)
# Modified reshape to match the model's expected input
reshape = (-1, 16, 60)
# Warm up with correct shape
model.predict(np.zeros((1, 16, 60)))

ACTION = 'right'
FFT_MAX_HZ = 60
HM_SECONDS = 10
TOTAL_ITERS = HM_SECONDS * 25
BOX_MOVE = "model"

last_print = time.time()
fps_counter = deque(maxlen=150)

print("looking for an EEG stream...")
streams = resolve_byprop('type', 'EEG', minimum=1, timeout=5.0)
inlet = StreamInlet(streams[0])

WIDTH = 800
HEIGHT = 800
SQ_SIZE = 50
MOVE_SPEED = 1

square = {
    'x1': int(WIDTH / 2 - SQ_SIZE / 2),
    'x2': int(WIDTH / 2 + SQ_SIZE / 2),
    'y1': int(HEIGHT / 2 - SQ_SIZE / 2),
    'y2': int(HEIGHT / 2 + SQ_SIZE / 2)
}

box = np.ones((square['y2'] - square['y1'], square['x2'] - square['x1'], 3)) * np.random.uniform(size=(3,))
horizontal_line = np.ones((HEIGHT, 10, 3)) * np.random.uniform(size=(3,))
vertical_line = np.ones((10, WIDTH, 3)) * np.random.uniform(size=(3,))

total = 0
left = 0
right = 0
none = 0
correct = 0

channel_datas = []

# Buffer to store FFT data
fft_buffer = deque(maxlen=60)
# Initialize buffer with zeros
for _ in range(60):
    fft_buffer.append(np.zeros(16))

for i in range(TOTAL_ITERS):
    channel_data = []
    for _ in range(16):
        sample, timestamp = inlet.pull_sample()
        channel_data.append(sample[FFT_MAX_HZ])
    
    # Update buffer with new data
    fft_buffer.append(np.array(channel_data))
    
    fps_counter.append(time.time() - last_print)
    last_print = time.time()
    cur_raw_hz = 1 / (sum(fps_counter) / len(fps_counter))
    print(f"Current Hz: {cur_raw_hz:.2f}")

    # Create visualization
    env = np.zeros((WIDTH, HEIGHT, 3))
    env[:, HEIGHT//2-5:HEIGHT//2+5, :] = horizontal_line
    env[WIDTH//2-5:WIDTH//2+5, :, :] = vertical_line
    env[square['y1']:square['y2'], square['x1']:square['x2'], :] = box
    
    cv2.imshow("EEG Visualization", env)
    cv2.waitKey(1)
    
    # Prepare input for model
    # Stack the buffer into the correct shape (1, 16, 60)
    network_input = np.array(list(fft_buffer)).T.reshape(1, 16, 60)
    out = model.predict(network_input, verbose=0)
    print(f"Model output: {out[0]}")
    
    if BOX_MOVE == "random":
        move = random.choice([-1, 0, 1])
        square['x1'] += move
        square['x2'] += move
        
    elif BOX_MOVE == "model":
        choice = np.argmax(out)
        if choice == 0:
            if ACTION == "left":
                correct += 1
            square['x1'] -= MOVE_SPEED
            square['x2'] -= MOVE_SPEED
            left += 1
        elif choice == 1:
            if ACTION == "right":
                correct += 1
            square['x1'] += MOVE_SPEED
            square['x2'] += MOVE_SPEED
            right += 1
        else:
            if ACTION == "none":
                correct += 1
            none += 1
    
    total += 1
    channel_datas.append(channel_data)

# Save the collected data
datadir = "data"
os.makedirs(datadir, exist_ok=True)

actiondir = os.path.join(datadir, ACTION)
os.makedirs(actiondir, exist_ok=True)

print(f"\nTotal recordings: {len(channel_datas)}")
print(f'Saving {ACTION} data...')
np.save(os.path.join(actiondir, f"{int(time.time())}.npy"), np.array(channel_datas))
print("Done.")

# Print statistics
for action in ['left', 'right', 'none']:
    action_dir = os.path.join('data', action)
    if os.path.exists(action_dir):
        total_size = sum(os.path.getsize(os.path.join(action_dir, f)) for f in os.listdir(action_dir))
        print(f"{action}: {total_size/1_000_000:.2f} MB")

accuracy = correct/total if total > 0 else 0
print(f"\nAccuracy for {ACTION}: {accuracy:.2%}")

# Save results to CSV
with open("accuracies.csv", "a") as f:
    f.write(f"{int(time.time())},{ACTION},{accuracy},{MODEL_NAME},{left/total},{right/total},{none/total}\n")

cv2.destroyAllWindows()

"""
######################################################################################################################################################################

                                                                        Version 1

######################################################################################################################################################################
"""




# """
# REQUIRES the OpenBCI GUI, Networking Tab, and choose
# ***125 channels*** for the FFT
# """

# from pylsl import StreamInlet,resolve_byprop #resolve_stream
# import numpy as np
# import time
# from matplotlib import style
# from collections import deque
# import cv2
# import os
# import random
# import tensorflow as tf

# #MODEL_NAME =""

# MODEL_NAME = r"C:\Users\Chat Noir\Desktop\AXL\models\32x1-5epoch-1738391525-acc-1.0-loss-0.01.keras"


# model = tf.keras.models.load_model(MODEL_NAME)
# reshape =(-1, 16, 60)
# model.perdict(np.zeros((32,16,60)).reshape(reshape))


# ACTION = 'left'

# FFT_MAX_HZ = 60

# HM_SECONDS = 10 #this is approximate. Not 1000%
# TOTAL_ITERS = HM_SECONDS *25 #~25 iters/sec
# BOX_MOVE = "model"


# last_print = time.time()
# fps_counter = deque(maxlen=150)

# # first resolve an EEG stream on the lab network
# print("looking for an EEG stream...")
# streams = resolve_byprop('type', 'EEG', minimum=1, timeout=5.0)
# # create a new inlet to read from the stream
# inlet = StreamInlet(streams[0])

# WIDTH = 800
# HEIGHT = 800
# SQ_SIZE = 50
# MOVE_SPEED = 1

# square = {'x1': int(int(WIDTH) / 2 - int(SQ_SIZE / 2)),
#           'x2': int(int(WIDTH) / 2 + int(SQ_SIZE / 2)),
#           'y1': int(int(HEIGHT) / 2 - int(SQ_SIZE / 2)),
#           'y2': int(int(HEIGHT) / 2 + int(SQ_SIZE / 2))}


# box = np.ones((square['y2'] - square['y1'], square['x2'] - square['x1'], 3)) * np.random.uniform(size=(3,))
# horizontal_line = np.ones((HEIGHT, 10, 3)) * np.random.uniform(size=(3,))
# vertical_line = np.ones((10, WIDTH, 3)) * np.random.uniform(size=(3,))



# total = 0
# left = 0
# right = 0
# none = 0
# correct = 0

# channel_datas = []


# for i in range(TOTAL_ITERS):  # how many iterations. Eventually this should be a while True
#     channel_data = []
#     for i in range(16):  # each of the 16 channels here
#         sample, timestamp = inlet.pull_sample()
#         channel_data.append(sample[FFT_MAX_HZ])
    
#     fps_counter.append(time.time() - last_print)
#     last_print = time.time()
#     cur_raw_hz = 1 / (sum(fps_counter) / len(fps_counter))
#     print(cur_raw_hz)

#     env = np.zeros((WIDTH, HEIGHT, 3))
    
#     env[:, HEIGHT//2-5:HEIGHT//2+5, :] = horizontal_line
#     env[WIDTH//2-5:WIDTH//2+5, :, :] = vertical_line
#     env[square['y1']:square['y2'], square['x1']:square['x2'], :] = box
    
    
    
#     cv2.imshow("", env)
#     cv2.waitKey(1)
    
#     network_input = np.array(channel_data).reshape(reshape)
#     out = model.predict(network_input)
#     print(out[0])
    
    
#     if BOX_MOVE == "random":
#         move = random.choice([-1, 0, 1])
#         square['x1'] += move
#         square['x2'] += move
        
#     elif BOX_MOVE == "model":
#         choice = np.argmax(out)
#         if choice == 0:
#             if ACTION == "left": # ACTION "Left":
#                 correct += 1
#             square['x1'] -= MOVE_SPEED
#             square['x2'] -= MOVE_SPEED
#             left += 1

#         elif choice == 2:
#             if ACTION == "right":  # ACTION "right":
#                 correct += 1
#             square['x1'] += MOVE_SPEED
#             square['x2'] += MOVE_SPEED
#             right += 1
        
#         else:
#             if ACTION == "none":
#                 correct += 1
#             none += 1
    
#     total += 1


#     channel_datas.append(channel_data)


# #plt.plot(channel_datas[0][0])
# #plt.show()

# datadir = "data"
# if not os.path.exists(datadir):
#     os.mkdir(datadir)

# actiondir = f"(datadir)/{ACTION}"
# if not os.path.exists(actiondir):
#     os.mkdir(actiondir)

# print(len(channel_datas))

# print(f'saving {ACTION} data...')
# np.save(os.path.join(actiondir, f"{int(time.time())}.npy"), np.array(channel_datas))
# print("done.")

# for action in ['left' , 'right', 'none']:
#     # print(f"{action}: {len(os.listdir(f'data/{action}'))}")
#     print(action, sum(os.path.getsize(f'data/{action}/{f}') for f in os.listdir(f'data/{action}'))/1_000_000, "MB")

# print(ACTION, correct/total)


# with open("accuracies.cvs", "a") as f:
#     f.write(f"{int(time.time())}, {ACTION}, {correct/total}.{MODEL_NAME}, {left/total},{right/total},{none/total}\n")