import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization
import os
import random
import time

# Ensure the directory exists
os.makedirs("new_models", exist_ok=True)

ACTIONS = ["left", "right", "none"]
# Modified reshape to preserve the correct number of samples
# The reshape should transform (samples, 16, 60) to (samples, 60, 16) 
# and not double the number of samples
reshape = None  # We'll calculate this dynamically based on our data

def create_data(starting_dir="data"):
    training_data = {}
    for action in ACTIONS:
        if action not in training_data:
            training_data[action] = []

        data_dir = os.path.join(starting_dir, action)
        for item in os.listdir(data_dir):
            data = np.load(os.path.join(data_dir, item))
            for item in data:
                training_data[action].append(item)

    lengths = [len(training_data[action]) for action in ACTIONS]
    print(lengths)

    for action in ACTIONS:
        np.random.shuffle(training_data[action])
        training_data[action] = training_data[action][:min(lengths)]

    lengths = [len(training_data[action]) for action in ACTIONS]
    print(lengths)
    
    combined_data = []
    for action in ACTIONS:
        for data in training_data[action]:
            if action == "left":
                combined_data.append([data, [1, 0, 0]])
            elif action == "right":
                combined_data.append([data, [0, 0, 1]])
            elif action == "none":
                combined_data.append([data, [0, 1, 0]])

    np.random.shuffle(combined_data)
    print("length:", len(combined_data))
    return combined_data

print("creating training data")
traindata = create_data(starting_dir="data")
train_X = []
train_y = []
for X, y in traindata:
    train_X.append(X)
    train_y.append(y)

print("creating testing data")
testdata = create_data(starting_dir="validation_data")
test_X = []
test_y = []
for X, y in testdata:
    test_X.append(X)
    test_y.append(y)

print(len(train_X))
print(len(test_X))

# Add shape debugging
print("train_X shape before reshape:", np.array(train_X).shape)
print("test_X shape before reshape:", np.array(test_X).shape)

# Convert to numpy arrays without reshaping
train_X = np.array(train_X)
test_X = np.array(test_X)

# Now reshape properly to (samples, 60, 16) by transposing the last two dimensions
train_X = np.transpose(train_X, (0, 2, 1))
test_X = np.transpose(test_X, (0, 2, 1))

# Add shape debugging after reshape
print("train_X shape after reshape:", train_X.shape)
print("test_X shape after reshape:", test_X.shape)

train_y = np.array(train_y)
test_y = np.array(test_y)
print("train_y shape:", train_y.shape)
print("test_y shape:", test_y.shape)

model = Sequential()

# Updated model with padding='same'
model.add(Conv1D(64, 3, padding='same', input_shape=train_X.shape[1:]))
model.add(Activation('relu'))

model.add(Conv1D(64, 2, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Conv1D(64, 2, padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))

model.add(Flatten())

model.add(Dense(512))
model.add(Activation('relu'))  # Added missing activation

model.add(Dense(3))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

epochs = 10
batch_size = 32
for epoch in range(epochs):
    model.fit(train_X, train_y, batch_size=batch_size, epochs=1, validation_data=(test_X, test_y))
    score = model.evaluate(test_X, test_y, batch_size=batch_size)
    # Use .h5 extension for valid model saving
    MODEL_NAME = f"new_models/{round(score[1]*100,2)}-acc-64x3-batch-norm-{epoch}epoch-{int(time.time())}-loss-{round(score[0],2)}.keras"
    model.save(MODEL_NAME)
print("saved:")
print(MODEL_NAME)





















# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
# from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization
# import os
# import random
# import time



# ACTIONS = ["left", "right", "none"]
# reshape = (-1, 8, 60)
# # TEST_PCT = 0.1

# def create_data(starting_dir="data"):
#     training_data = {}
#     for action in ACTIONS:
#         if action not in training_data:
#             training_data[action] = []

#         data_dir = os.path.join(starting_dir,action)
#         for item in os.listdir(data_dir):
#             #print(action, item)
#             data = np.load(os.path.join(data_dir, item))
#             for item in data:
#                 training_data[action].append(item)

#     lengths = [len(training_data[action]) for action in ACTIONS]
#     print(lengths)

#     for action in ACTIONS:
#         np.random.shuffle(training_data[action])  # note that regular shuffle is GOOF af
#         training_data[action] = training_data[action][:min(lengths)]

#     lengths = [len(training_data[action]) for action in ACTIONS]
#     print(lengths)
#     # creating X, y 
#     combined_data = []
#     for action in ACTIONS:
#         for data in training_data[action]:

#             if action == "left":
#                 combined_data.append([data, [1, 0, 0]])

#             elif action == "right":
#                 #np.append(combined_data, np.array([data, [1, 0]]))
#                 combined_data.append([data, [0, 0, 1]])

#             elif action == "none":
#                 combined_data.append([data, [0, 1, 0]])

#     np.random.shuffle(combined_data)
#     print("length:",len(combined_data))
#     return combined_data


# print("creating training data")
# traindata = create_data(starting_dir="data")
# train_X = []
# train_y = []
# for X, y in traindata:
#     train_X.append(X)
#     train_y.append(y)

# print("creating testing data")
# testdata = create_data(starting_dir="validation_data")
# test_X = []
# test_y = []
# for X, y in testdata:
#     test_X.append(X)
#     test_y.append(y)

# print(len(train_X))
# print(len(test_X))


# print(np.array(train_X).shape)
# train_X = np.array(train_X).reshape(reshape)
# test_X = np.array(test_X).reshape(reshape)

# train_y = np.array(train_y)
# test_y = np.array(test_y)

# model = Sequential()

# model.add(Conv1D(64, (3), input_shape=train_X.shape[1:]))
# model.add(Activation('relu'))

# model.add(Conv1D(64, (2)))
# model.add(Activation('relu'))
# model.add(MaxPooling1D(pool_size=(2)))

# model.add(Conv1D(64, (2)))
# model.add(Activation('relu'))
# model.add(MaxPooling1D(pool_size=(2)))

# model.add(Flatten())

# model.add(Dense(512))

# model.add(Dense(3))
# model.add(Activation('softmax'))

# model.compile(loss='categorical_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

# epochs = 10
# batch_size = 32
# for epoch in range(epochs):
#     model.fit(train_X, train_y, batch_size=batch_size, epochs=1, validation_data=(test_X, test_y))
#     score = model.evaluate(test_X, test_y, batch_size=batch_size)
#     #print(score)
#     MODEL_NAME = f"new_models/{round(score[1]*100,2)}-acc-64x3-batch-norm-{epoch}epoch-{int(time.time())}-loss-{round(score[0],2)}.model"
#     model.save(MODEL_NAME)
# print("saved:")
# print(MODEL_NAME)
