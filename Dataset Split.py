# train data를 쪼개서 validation 으로 바꾸는 방법
# test data는 wild field 에서 진행해야 한다. 완전히 처음 적용하는 data

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

train_x = np.arange(100).reshape(-1, 1)
train_y = 3*train_x + 1

train_validation_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))

# train_ds 앞에 있는 10개의 데이터를 가져온다
# tmp_ds = train_ds.take(10)
#
# for x, y in tmp_ds:
#     print(x)
#     print(y, '\n')

n_train_validation = 100
n_train_ratio = 0.8
n_train = int(n_train_ratio * n_train_validation)

print(n_train_validation)
print(n_train)

train_ds = train_validation_ds.take(n_train)

###############################################################
train_x = np.arange(100).reshape(-1, 1)
train_y = 3*train_x + 1
n_train = int(n_train_validation * n_train_ratio)
n_validation = n_train_validation - n_train

train_validation_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))

remaining_ds = train_validation_ds.skip(n_train)
validation_ds = remaining_ds.take(n_validation)

for x, y in remaining_ds:
    print(x)
    print(y, '\n')
    break

##################################################################3
(train_validation_ds, test_ds), ds_info = tfds.load(name='mnist',
                                                    shuffle_files=True,
                                                    as_supervised=True,
                                                    split=['train','test'],
                                                    with_info=True)

#print(ds_info.splits['train'].num_examples)
n_train_validation = ds_info.splits['train'].num_examples
train_ratio = 0.8
n_train = int(n_train_validation * train_ratio)
n_validation = n_train_validation - n_train

train_ds = train_validation_ds.take(n_train)
remaining_ds = train_validation_ds.skip(n_train)
validation_ds = remaining_ds.take(n_validation)

train_ds = train_ds.shuffle(100).batch(32)
validation_ds = validation_ds.batch(32)
test_ds = test_ds.batch(32)
