import tensorflow as tf
import numpy as np


# train_x = np.arange(1000).astype(np.float32).reshape(-1, 1)
# train_y = 3*train_x + 1
#
# train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
# train_ds = train_ds.shuffle(100).batch(32)
#
# for x, y in train_ds:
#     print(x.shape)
#     print(y.shape, '\n')

import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist
from tensorflow.data import Dataset

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# print(type(train_images))
# print(train_images.shape)
# print(train_labels.shape)
# print(test_images.shape)
# print(test_labels.shape)

# import sys
# print(sys.getsizeof(train_images)/1024/1024) # 데이터 크기가 44MB

train_ds = Dataset.from_tensor_slices((train_images, train_labels))
train_ds = train_ds.shuffle(60000).batch(9)

test_ds = Dataset.from_tensor_slices((test_images, test_labels))
test_ds = test_ds.batch(9)

train_ds_iter = iter(train_ds)
images, labels = next(train_ds_iter)

print(images.shape)
print(labels.shape)

fig, axes = plt.subplots(3, 3, figsize=(10, 10))

for ax_idx, ax in enumerate(axes.flat):
    image = images[ax_idx, ...]
    label = labels[ax_idx]

    print(image.shape)
    print(label.shape)

    ax.imshow(image.numpy(), 'gray')
    ax.set_title(label.numpy(), fontsize=20)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.show()
