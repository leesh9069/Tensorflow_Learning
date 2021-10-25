import os
import json
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten, Dense, Activation
from tensorflow.keras.layers import InputLayer

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# model build

model = Sequential()
#model.add(InputLayer(input_shape=(28, 28, 1))) # batch size를 안 넣어줌
model.add(Flatten())
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=2))
model.add(Activation('softmax'))

model.build(input_shape=(None, 28, 28, 1))
# model.build()
model.summary()

# tf.keras.backend.clear_session()

###################################################################
# subclassing 은 inputlayer를 사용하지 않는다.
class TestModel(Model):
    def __init__(self):
        super(TestModel, self).__init__()

        self.flatten = Flatten()
        self.d1 = Dense(units=10)
        self.d1_act = Activation('relu')
        self.d2 = Dense(units=2)
        self.d2_act = Activation('softmax')

    def call(self, x):
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d1_act(x)
        x = self.d2(x)
        x = self.d2_act(x)

        return x

model = TestModel()
model.build(input_shape=(None, 28, 28, 1))

model.summary()

###########################################################
model = Sequential()
model.add(Flatten())
model.add(Dense(units=10, activation='relu'))
model.add(Dense(units=2))
model.add(Activation('softmax'))

print(model.built)
test_img = tf.random.normal(shape=(1, 28, 28, 1))
model(test_img)
print(model.built)