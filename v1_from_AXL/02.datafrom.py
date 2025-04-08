################################################################################

            # USING LSL FFT from BCI GUI to plot the FFT

#################################################################################

from pyOpenBCI import OpenBCICyton
import time
from collections import deque
import numpy as np
# import cv2

# Initialize variables
last_print = time.time()
fps_counter = deque(maxlen=50)
sequence = np.zeros((5000, 16))
counter = 0

def print_raw(sample):
    global last_print
    global sequence
    global counter

    # Roll the sequence and add new data
    sequence = np.roll(sequence, -1, axis=0)
    sequence[-1, ...] = sample.channels_data

    # Calculate FPS
    fps_counter.append(time.time() - last_print)
    last_print = time.time()
    if len(fps_counter) > 0:
        fps = 1 / (sum(fps_counter) / len(fps_counter))
    else:
        fps = 0

    # Print FPS and counter
    print(f'FPS: {fps:.2f}, Sequence Length: {len(sequence)}, Counter: {counter}')
    print(sample.channels_data)

    # print('\n')
    # print(sample.channels_data)
    # print('\n')
    # print('\n')
    
    # Increment counter
    counter += 1

    # Save data every 30,000 samples
    if counter == 5000:
        np.save("seq.npy", sequence)
        print("Data saved to seq.npy")

# Initialize OpenBCI board
board = OpenBCICyton(port='/dev/ttyUSB0', daisy=True)

try:
    # Start streaming
    board.start_stream(print_raw)
    
except KeyboardInterrupt:
    # Gracefully stop the stream
    print("Stream stopped by user.")
finally:
    # Save the final sequence if needed
    np.save("seq_final_5000.npy", sequence)
    print("Final data saved to seq_final.npy")