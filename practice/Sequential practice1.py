import tensorflow as tf

# dataset
x_train = tf.random.normal(shape=(1000,),dtype=tf.float32)
y_train = 3*x_train + 1 + 0.2*tf.random.normal(shape=(1000,),dtype=tf.float32)

x_test = tf.random.normal(shape=(300,),dtype=tf.float32)
y_test = 3*x_test + 1 + 0.2*tf.random.normal(shape=(300,),dtype=tf.float32)

# modeling
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=1,
                          activation='linear')
])

#model compile
model.compile(loss='mean_squared_error',
              optimizer = 'SGD')

model.fit(x_train, y_train, batch_size=64, epochs=100)