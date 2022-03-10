# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 12:50:04 2022

@author: patse
"""

import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

stringToPredict = "Couldn’t hit shit this game rip"

# Preprocessing
tokenizer = None
with open('tokenizer', 'rb') as f:
    tokenizer = pickle.load(f)

batchedString = np.expand_dims(stringToPredict, axis=0)
tokenizedString = tokenizer.texts_to_sequences(batchedString)
paddedString = np.array(pad_sequences(tokenizedString, padding="post", truncating="post", maxlen=25))

# Load the model
model = tf.keras.models.load_model('Twitter-Sentiment-Model')

# Feed preprocessed input into model
prediction = model.predict(paddedString)

# Decode prediction
outputCategories = ["Positive", "Neutral", "Negative"]
predictionDict = dict(zip(outputCategories, prediction[0]))
max_key = max(predictionDict, key=predictionDict.get)

print(max_key)