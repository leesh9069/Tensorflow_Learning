# tensorflow data 받아오기

import tensorflow_datasets as tfds

# dataset, ds_info = tfds.load(name='mnist',
#                              shuffle_files=True,
#                              with_info=True)

# n_train = ds_info.splits['train'].num_examples
# n_test = ds_info.splits['test'].num_examples

dataset = tfds.load(name='mnist',
                    shuffle_files=True,
                    as_supervised=True)  # as_supervised - type(tmp)가 dict에서 tuple 로 변경

# print(type(dataset))
# print(dataset.keys(), '\n')
# print(dataset.values())

train_ds = dataset['train'].batch(32)
#test_ds  = dataset['test']

# print(type(train_ds))
# print(type(test_ds))

## as_supervised = True가 아닐 때
# for tmp in train_ds:
#     # print(type(tmp))
#     #
#     # print(tmp.keys())
#     images = tmp['image']
#     labels = tmp['label']
#
#     print(images.shape)
#     print(labels.shape)
#     break

## as_supervised = True (Tuple로 tmp 변환, 코드가 더 간단해진다)
for images, labels in train_ds:

    print(images.shape)
    print(labels.shape)
    break

(train_ds, test_ds), ds_info = tfds.load(name='mnist',
                                         shuffle_files=True,
                                         as_supervised=True,  # as_supervised - type(tmp)가 dict에서 tuple 로 변경
                                         split=['train', 'test'],  # data를 바로 분리해준다
                                         with_info=True) # ds_info 를 불러와줌

train_ds = train_ds.batch(32)

for images, labels in train_ds:
    print(images.shape)
    print(labels.shape)
    break

(train_ds, validation_ds, test_ds), ds_info = tfds.load(name='patch_camelyon',
                                                        shuffle_files=True,
                                                        as_supervised=True,  # as_supervised - type(tmp)가 dict에서 tuple 로 변경
                                                        split=['train', 'validation', 'test'],
                                                        with_info=True,
                                                        batch_size=16)  # ds_info 를 불러와줌

# print(ds_info.features, '\n')
# print(ds_info.splits)
#
# train_ds = train_ds.batch(9)


train_ds_iter = iter(train_ds)
images, labels = next(train_ds_iter)
images = images.numpy()
labels = labels.numpy()

#print(images.shape)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 3, figsize=(15,15))

for ax_idx, ax in enumerate(axes.flat):
    ax.imshow(images[ax_idx,...])
    ax.set_title(labels[ax_idx], fontsize=30)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)