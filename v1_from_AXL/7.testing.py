from pylsl import StreamInlet, resolve_byprop
import numpy as np
import time
from collections import deque
import cv2
import tensorflow as tf

MODEL_NAME = "/home/zee/Documents/AXL/models/32x1-10epoch-1742202030-acc-1.0-loss-0.0.keras"  #r"C:\Users\Chat Noir\Desktop\AXL\models\32x1-10epoch-1738403977-acc-1.0-loss-0.0.keras"
# Load and warm up the model
model = tf.keras.models.load_model(MODEL_NAME)
reshape = (-1, 8, 60)
model.predict(np.zeros((1, 8, 60)))

ACTION = 'none'
FFT_MAX_HZ = 60
HM_SECONDS = 10
TOTAL_ITERS = HM_SECONDS * 25
BOX_MOVE = "model"

last_print = time.time()
fps_counter = deque(maxlen=150)

print("looking for an EEG stream...")
streams = resolve_byprop('type', 'EEG', minimum=1, timeout=5.0)
inlet = StreamInlet(streams[0])

WIDTH, HEIGHT, SQ_SIZE, MOVE_SPEED = 800, 800, 50, 1
square = {
    'x1': int(WIDTH / 2 - SQ_SIZE / 2),
    'x2': int(WIDTH / 2 + SQ_SIZE / 2),
    'y1': int(HEIGHT / 2 - SQ_SIZE / 2),
    'y2': int(HEIGHT / 2 + SQ_SIZE / 2)
}

box = np.ones((square['y2'] - square['y1'], square['x2'] - square['x1'], 3)) * np.random.uniform(size=(3,))
horizontal_line = np.ones((HEIGHT, 10, 3)) * np.random.uniform(size=(3,))
vertical_line = np.ones((10, WIDTH, 3)) * np.random.uniform(size=(3,))

total, left, right, none, correct = 0, 0, 0, 0, 0

# Buffer to store FFT data
fft_buffer = deque(maxlen=60)
for _ in range(60):
    fft_buffer.append(np.zeros(8))

for _ in range(TOTAL_ITERS):
    channel_data = [inlet.pull_sample()[0][FFT_MAX_HZ] for _ in range(8)]
    fft_buffer.append(np.array(channel_data))
    
    fps_counter.append(time.time() - last_print)
    last_print = time.time()
    print(f"Current Hz: {1 / (sum(fps_counter) / len(fps_counter)):.2f}")
    
    env = np.zeros((WIDTH, HEIGHT, 3))
    env[:, HEIGHT//2-5:HEIGHT//2+5, :] = horizontal_line
    env[WIDTH//2-5:WIDTH//2+5, :, :] = vertical_line
    env[square['y1']:square['y2'], square['x1']:square['x2'], :] = box
    
    cv2.imshow("EEG Visualization", env)
    cv2.waitKey(1)
    
    network_input = np.array(list(fft_buffer)).T.reshape(1, 8, 60)
    out = model.predict(network_input, verbose=0)
    print(f"Model output: {out[0]}")
    
    choice = np.argmax(out)
    if choice == 0:
        if ACTION == "left": correct += 1
        square['x1'] -= MOVE_SPEED
        square['x2'] -= MOVE_SPEED
        left += 1
    elif choice == 1:
        if ACTION == "right": correct += 1
        square['x1'] += MOVE_SPEED
        square['x2'] += MOVE_SPEED
        right += 1
    else:
        if ACTION == "none": correct += 1
        none += 1
    
    total += 1
    
cv2.destroyAllWindows()

accuracy = correct / total if total > 0 else 0
print(f"\nAccuracy for {ACTION}: {accuracy:.2%}")

with open("accuracies.csv", "a") as f:
    f.write(f"{int(time.time())},{ACTION},{accuracy},{MODEL_NAME},{left/total},{right/total},{none/total}\n")
