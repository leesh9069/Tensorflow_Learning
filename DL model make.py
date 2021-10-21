import tensorflow as tf
import matplotlib.pyplot as plt
# how to make tensorflow model : Sequential / Functional / Model Subclassing

x_train = tf.random.normal(shape=(1000, ), dtype=tf.float32)
y_train = 3*x_train + 1 + 0.2*tf.random.normal(shape=(1000,),dtype=tf.float32)

x_test = tf.random.normal(shape=(300, ), dtype=tf.float32)
y_test = 3*x_test + 1 + 0.2*tf.random.normal(shape=(300,),dtype=tf.float32)
# fig, ax = plt.subplots(figsize=(10,10))
# ax.scatter(x_train.numpy(),
#            y_train.numpy())
# ax.tick_params(labelsize=20)
# ax.grid()
# plt.show()

# Sequential modeling method - simple model(Convolution)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1,
                          activation='linear')
    ])

model.compile(loss='mean_squared_error',
              optimizer='SGD')

model.fit(x_train, y_train, epochs=50, verbose=2)
model.evaluate(x_test, y_test, verbose=2)

# Model subclassing modeling method
from termcolor import colored
class LinearPredictor(tf.keras.Model):
    def __init__(self):
        super(LinearPredictor, self).__init__()

        self.d1 = tf.keras.layers.Dense(units=1,
                                        activation='linear')
    # self.d1을 실행시키는 def
    def call(self, x):
        x = self.d1(x)
        return x

EPOCHS = 10
LR = 0.01

# instantiation learning objects
model = LinearPredictor()

loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.SGD(learning_rate=LR)

# learning
for epoch in range(EPOCHS):
    for x,y in zip(x_train,y_train):
        x = tf.reshape(x, (1,1))
        with tf.GradientTape() as tape:
            predictions = model(x)
            loss = loss_object(y, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    print(colored('Epoch: ', 'red', 'on_white'), epoch+1)

    template = 'Train Loss: {:.4f}\n'
    print(template.format(loss))
