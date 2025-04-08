# import numpy as np
# import time
# from collections import deque
# import matplotlib
# matplotlib.use('TkAgg')  # Set an interactive backend
# import matplotlib.pyplot as plt
# from matplotlib import style

# # Use ggplot style for plotting
# style.use("ggplot")

# # Initialize FPS counter
# fps_counter = deque(maxlen=100)

# # Parameters
# FPS = 125  # Sampling rate (samples per second)
# HM_SECONDS_SLICE = 180  # Length of the slice in seconds

# # Load data
# data = np.load("seq_final.npy")
# print(f"Length of data: {len(data)}")  # Print the number of samples
# print(f"Shape of data: {data.shape}")  # Print the shape of the data (samples x channels)
# print(f"               :{FPS * HM_SECONDS_SLICE}")
# # Check if the data is large enough for the specified slice size
# if (FPS * HM_SECONDS_SLICE) < len(data):
#     # Iterate through the data
#     for i in range(0, len(data) - FPS * HM_SECONDS_SLICE):
#         print("Plot") 
#         # Extract a slice of data (1 second of data)
#         new_data = data[i:i + FPS * HM_SECONDS_SLICE]

#         # Extract the c8 channel (index 3)
#         c8 = new_data[:, 15]

#         # Print the c8 channel data
#         print(c8)

#         # Plot the c8 channel data
#         plt.figure(figsize=(10, 4))  # Set figure size
#         plt.plot(c8, label="c8 Channel")
#         plt.title("c8 Channel Data (1 Second Slice)")
#         plt.xlabel("Time (samples)")
#         plt.ylabel("Amplitude")
#         plt.legend()
#         plt.grid(True)
#         plt.show(block=True)  # Ensure the plot window opens

#         # Simulate real-time delay (optional)
#         time.sleep(1 / FPS) 
#         # Break after the first plot (remove this if you want to plot all slices)
#         break
# else:
#     print("Data is too small for the specified slice size.")


import numpy as np
import time
from collections import deque
import matplotlib
matplotlib.use('TkAgg')  # Set an interactive backend
import matplotlib.pyplot as plt
from matplotlib import style

# Use ggplot style for plotting
style.use("ggplot")

# Initialize FPS counter
fps_counter = deque(maxlen=100)

# Parameters
FPS = 125  # Sampling rate (samples per second)
HM_SECONDS_SLICE = 10  # Length of the slice in seconds
ch = 9
# Load data from CSV
data = np.load('seq_final_5000.npy')
print(f"Length of data: {len(data)}")  # Print the number of samples
print(f"Shape of data: {data.shape}")  # Print the shape of the data (samples x channels)
print(f"               :{FPS * HM_SECONDS_SLICE}")

# Check if the data is large enough for the specified slice size
if (FPS * HM_SECONDS_SLICE) < len(data):
    # Iterate through the data
    for i in range(0, len(data) - FPS * HM_SECONDS_SLICE):
        print("Plot") 
        # Extract a slice of data (1 second of data)
        new_data = data[i:i + FPS * HM_SECONDS_SLICE]

        # Extract the c8 channel (index 15, since Python uses 0-based indexing)
        ch_data = new_data[:, ch-1]

        # Print the c8 channel data
        print(ch_data)

        # Plot the c8 channel data
        plt.figure(figsize=(10, 4))  # Set figure size
        plt.plot(ch_data, label=f"{ch} Channel")
        plt.title(f"{ch} Channel Data ({HM_SECONDS_SLICE} Second Slice)")
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.show(block=True)  # Ensure the plot window opens

        # Simulate real-time delay (optional)
        time.sleep(1 / FPS) 
        # Break after the first plot (remove this if you want to plot all slices)
        break
else:
    print("Data is too small for the specified slice size.")