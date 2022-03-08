# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 11:58:37 2022

@author: patse
"""

import os
import csv
import time
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

"""
serializeDataSet = False

if serializeDataSet == True:
    # Import input data
    print("Importing data from csv")
    input_data = []
    with open('Twitter Dataset.csv', encoding="utf8") as csv_file:   
        csv_reader = csv.reader(csv_file)
        input_data = list(csv_reader)
    
    # Separate data into training and test sets
    test_list = []
    
    count = 0
    # 20% of my data should be in my test set
    max_test_count = int(len(input_data) * 0.2)
    
    for row in input_data:
        count += 1
        index = random.randint(0, len(input_data)-1)
        
        test_list.append(input_data[index])
        if (count == max_test_count):
            break
    
    train_list = []
    for row in input_data:
        if not row in test_list:
            train_list.append(row)

    print(len(train_list), len(test_list), len(train_list) + len(test_list))
    
    # Save lists as csv files
    with open('Test_Set.csv', 'w', encoding="utf8", newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(test_list)
        
    with open('Train_Set.csv', 'w', encoding="utf8", newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerows(train_list)
        
    print("Completed serialization of training and testing datasets.")
"""







# tf.debugging.set_log_device_placement(True)
print("\n", tf.config.list_physical_devices(), "\n", tf.config.list_logical_devices(), "\n")

with tf.device('/device:CPU:0'):
    start = time.time()
    
    end = time.time()
    print("Time elapsed for CPU", end - start, "\n")
    
with tf.device('/device:GPU:0'):
    start = time.time()
    
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
        train_data.append([row[3]])
        train_labels.append(row[2])
    
    test_data = []
    test_labels = []
    for row in test_list:
        test_data.append([row[3]])
        test_labels.append(row[2])
    
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    
    # One hot encode the labels
    outputCategories = ["Positive", "Neutral", "Negative", "Irrelevant"]
    
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
    
    print(train_data.shape, train_labels.shape, test_data.shape, test_labels.shape)
    
    # The model
    model = keras.Sequential()    
    # model.add(keras.Input(shape=(1,)))
    model.add(layers.Embedding(1000, 64))
    model.add(layers.Dense(4))
    
    print(model.summary())
    
    
    
    # model = keras.Sequential()
    # model.add()
    # model.add(layers.Embedding(input_dim=1000, output_dim=64))
    # model.add(layers.LSTM(128))
    # model.add(layers.Dense(4))

    # print(model.summary())

    model.compile('rmsprop', 'mse')

    input_array = train_data[0]
    output_array = model.predict(input_array)
    print(output_array)
    
    # end = time.time()
    # print("Time elapsed for GPU", end - start)