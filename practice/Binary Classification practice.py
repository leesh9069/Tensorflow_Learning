import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from termcolor import colored

plt.style.use('seaborn')

n_sample = 100
x_train = np.random.normal(0,1,size=(n_sample,1)).astype(np.float32)
y_train = (x_train >=0).astype(np.float32)

fig,ax = plt.subplots(figsize=(20,10))
ax.scatter(x_train, y_train)
ax.tick_params(labelsize=20)
plt.show()

class Classifier(tf.keras.Model):
    def __init__(self):
        super(Classifier, self).__init__()

        self.d1 = tf.keras.layers.Dense(units=1,
                                        activation='sigmoid')

    def call(self,x):
        predictions = self.d1(x)
        return predictions


EPOCHS = 100
LR = 0.01

model = Classifier()
loss_object = tf.keras.losses.BinaryCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate=LR)
