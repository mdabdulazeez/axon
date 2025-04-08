import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Activation, Flatten, Dense 
from sklearn.preprocessing import LabelEncoder
import time
import os

# Data Loading (Replace with your actual data loading code)
ACTIONS = ["left", "right", "none"]
reshape = (-1, 8, 60)  # 8 channels, 60 time steps
TEST_PCT = 0.1

training_data = {}
starting_dir = f"data_"

for action in ACTIONS:
    if action not in training_data:
        training_data[action] = []

    data_dir = os.path.join(starting_dir, action)
    for item in os.listdir(data_dir):
        data = np.load(os.path.join(data_dir, item))
        for example in data:
            training_data[action].append(example)

# Balance the dataset
lengths = [len(training_data[action]) for action in ACTIONS]
print("Lengths of each class:", lengths)

min_len = min(lengths)  # Find minimum length for balancing
for action in ACTIONS:
    np.random.shuffle(training_data[action])
    training_data[action] = training_data[action][:min_len] # Truncate to min length

# Combine and shuffle
combined_data = []

for action in ACTIONS:
    for data in training_data[action]:
        if action == "left":
            combined_data.append([data, [1, 0, 0]])
        elif action == "right":
            combined_data.append([data, [0, 1, 0]])
        elif action == "none":
            combined_data.append([data, [0, 0, 1]])

np.random.shuffle(combined_data)
test_size = int(len(combined_data) * TEST_PCT)  # Split into train and test
print("Train Size:", len(combined_data)-test_size)
print("Test Size:", test_size)

# Split into training and testing sets
train_X = []
train_y =[]
for X, y in combined_data[: -test_size]:
    train_X.append(X)
    train_y.append(y)

test_X = []
test_y =[]
for X, y in combined_data[-test_size:]:  # Fixed test set selection
    test_X.append(X)
    test_y.append(y)

print("Training Size of X", len(train_X))
print("Test Size of X", len(test_X))

print(np.array(train_X).shape)
train_X = np.array(train_X).reshape(reshape)
test_X = np.array(test_X).reshape(reshape)

train_y = np.array(train_y)
test_y = np.array(test_y)

# Model Definition with padding to prevent dimension issues
model = Sequential()

model.add(Conv1D(32, 3, padding='same', input_shape=train_X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(32, 2, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(32, 2, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())

model.add(Dense(64))

model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', 
              optimizer='adam',
              metrics=['accuracy'])

# Print model summary to check dimensions
model.summary()

model.fit(train_X, train_y, batch_size=32, epochs=10, validation_split=0.15)
score = model.evaluate(test_X, test_y, batch_size=128, verbose=0)
print(f"Test Loss: {score[0]:.4f}")
print(f"Test Accuracy: {score[1]:.4f}")

# Ensure the 'models' directory exists
if not os.path.exists("models"):
    os.makedirs("models")

# Model Saving
MODEL_NAME = f"models/32x1-10epoch-{int(time.time())}-acc-{round(score[1], 2)}-loss-{round(score[0], 2)}.keras"
model.save(MODEL_NAME)
print(f"Model saved to: {MODEL_NAME}")