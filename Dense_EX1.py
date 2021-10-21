import tensorflow as tf
import matplotlib.pyplot as plt
#
# t1 = tf.Variable([1,2,3], dtype=tf.float32)
# t2 = tf.Variable([10,20,30], dtype=tf.float32)
#
# with tf.GradientTape() as tape:
#     t3 = t1 * t2
#     t4 = t3 + t2
#
# gradients = tape.gradient(t4, [t1, t2, t3])
# print(gradients[0])
# print(gradients[1])
# print(gradients[2])

x_data = tf.random.normal(shape=(1000, ), dtype=tf.float32)
y_data = 3*x_data + 1

w = tf.Variable(-1.)
b = tf.Variable(-1.)

LR = 0.01
EPOCHS = 10
w_trace, b_trace = [], []
for epoch in range(EPOCHS):
    for x,y in zip(x_data, y_data):
        with tf.GradientTape() as tape:
            prediction = w*x + b
            loss = (prediction - y) ** 2

        gradients = tape.gradient(loss, [w,b])

        w_trace.append(w.numpy())
        b_trace.append(b.numpy())

        w = tf.Variable(w - LR*gradients[0])
        b = tf.Variable(b - LR*gradients[1])

fig, ax  = plt.subplots(figsize = (20, 10))

ax.plot(w_trace, label = 'weight')
ax.plot(b_trace, label='bias')
ax.tick_params(labelsize=20)
ax.legend(fontsize=30)
ax.grid()
plt.show()

