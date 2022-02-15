import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# The model
model = keras.Sequential()
model.add(layers.Embedding(input_dim=1000, output_dim=64))
model.add(layers.LSTM(128))
model.add(layers.Dense(10))

print(model.summary())

model.compile('rmsprop', 'mse')

input_array = np.random.randint(1000, size=(32,15))
output_array = model.predict(input_array)

# Should be (32, 10)
print(output_array.shape)

