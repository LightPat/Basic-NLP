# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 11:58:37 2022

@author: patse
"""

import csv
import time
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

train_list = []
test_list = []
with open('Train_Set.csv', encoding="utf8") as csv_file:
    csv_reader = csv.reader(csv_file)
    train_list = list(csv_reader)
    
with open('Test_Set.csv', encoding="utf8") as csv_file:
    csv_reader = csv.reader(csv_file)
    test_list = list(csv_reader)

random.shuffle(train_list)
random.shuffle(test_list)

# Separate data and labels
train_data = []
train_labels = []
for row in train_list:
    if row[2] == "Irrelevant":
        continue
    train_data.append(row[3])
    train_labels.append(row[2])

test_data = []
test_labels = []
for row in test_list:
    if row[2] == "Irrelevant":
        continue
    test_data.append(row[3])
    test_labels.append(row[2])
    
    
train_data = np.array(train_data)
test_data = np.array(test_data)

# One hot encode the labels
outputCategories = ["Positive", "Neutral", "Negative"]

train_indexes = []
for label in train_labels:
    train_indexes.append(outputCategories.index(label))

train_labels = np.array(train_indexes)
train_labels = keras.utils.to_categorical(train_labels, len(outputCategories))


test_indexes = []
for label in test_labels:
    test_indexes.append(outputCategories.index(label))

test_labels = np.array(test_indexes)
test_labels = keras.utils.to_categorical(test_labels, len(outputCategories))


# Hyperparameteres
vocab_size = 20000
oov_token = "<OOV>"
padding_type = "post"
truncating_type = "post"
embedding_dim = 16
max_length = 25

# Tokenize the training data
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(train_data)
word_index = tokenizer.word_index

# Pad the sequences so that they are all the same length
train_sequences = tokenizer.texts_to_sequences(train_data)
train_padded = pad_sequences(train_sequences, padding=padding_type, truncating=truncating_type, maxlen=max_length)

test_sequences = tokenizer.texts_to_sequences(test_data)
test_padded = pad_sequences(test_sequences, padding=padding_type, truncating=truncating_type, maxlen=max_length)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(3, activation='sigmoid')
])

with tf.device('/device:CPU:0'):
    start = time.time()
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    history = model.fit(train_padded, train_labels, epochs=10,
                        validation_data=(test_padded, test_labels), verbose=1)

    end = time.time()
    print("Time elapsed for CPU", end-start)



model.save('Twitter-Sentiment-Model')