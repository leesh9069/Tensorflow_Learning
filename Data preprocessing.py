import tensorflow as tf
import tensorflow_datasets as tfds

train_ds = tfds.load(name='mnist',
                     shuffle_files=True,
                     as_supervised=True,
                     split='train',
                     batch_size=4)

for images, labels in train_ds:
    print(images.shape)
    print(images.dtype)

    # 255 - normalization을 해주지 않으면 weight 변동이 크다 - 학습이 거의 일어나지 않다
    # 0-255 를 0 - 1로 바꿔준다
    print(tf.reduce_max(images))
    # print(labels.shape)
    # print(labels.dtype)
    break

################################################################
# map
a = [1,2,3,4,5]
def double(in_val):
    return 2* in_val

doubled = list(map(double, a))
print(doubled)

#lambda
lambda x : 2*x
doubled2 = list(map(lambda x:2*x, a))
print(doubled2)

# tf.cast
t1 = tf.constant([1,2,3,4,5])
print(t1.dtype)
t2 = tf.cast(t1, tf.float32)
print(t2.dtype)

#############################################################
# 0-255 -> 0-11 / uint8 -> float32
def standardization(images, labels):
    images = tf.cast(images, tf.float32) / 255.
    return [images, labels]

train_ds_iter = iter(train_ds)
images, labels = next(train_ds_iter)
print('images.dtype :', images.dtype)
print(tf.reduce_max(images))

train_ds = train_ds.map(standardization)
train_ds_iter = iter(train_ds)
images, labels = next(train_ds_iter)
print('images.dtype :', images.dtype)
print(tf.reduce_max(images))
##################################################################
# data load function
def mnist_data_loader():

    def standardization(images, labels):
        images = tf.cast(images, tf.float32) / 255.
        return [images, labels]

    train_ds, test_ds = tfds.load(name='mnist',
                         shuffle_files=True,
                         as_supervised=True,
                         split=['train', 'test'],
                         batch_size=4)

    train_ds = train_ds.map(standardization)
    test_ds = test_ds.map(standardization)

    return train_ds, test_ds

train_ds, test_ds = mnist_data_loader()
