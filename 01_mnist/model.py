from __future__ import absolute_import, division, print_function
import os
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, Lambda
from keras import backend as K


# Defining simple model
def by_pass(layer):
  return layer

def create_model(temp = 1.0):
#  model = tf.keras.models.Sequential([
#    keras.layers.Flatten(input_shape=(28, 28)),
#    keras.layers.Dense(128, activation=tf.nn.relu),
#    keras.layers.Dense(10, activation=tf.nn.softmax)

#model.add(Lambda(lambda x: x / temp))
#  ])  

  model = Sequential()
  model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
  model.add(Conv2D(32, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2, 2)))
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(256, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Lambda(lambda x: x / temp))
  model.add(Dense(10, activation='softmax'))

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  return model


























