from termcolor import colored

import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import Flatten, Dense, Activation

model = Sequential()
model.add(Conv2D(filters=10, kernel_size=(3, 3), padding='valid', name='conv_1'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, name='conv_1_maxpool'))
model.add(Activation('relu', name='conv_1_act'))

model.add(Conv2D(filters=10, kernel_size=(3, 3), padding='valid', name='conv_2'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=2, name='conv_2_maxpool'))
model.add(Activation('relu', name='softmax'))

model.add(Flatten())
model.add(Dense(units=32, activation='relu', name='dense_1'))
model.add(Dense(units=10, activation='softmax', name='dense_2'))

model.build(input_shape=(None, 28, 28, 1))

# print(colored(" ", 'cyan'), '\n', , '\n')
print(colored("model.layers", 'cyan'), '\n', model.layers, '\n')

print(len(model.layers))

final_layer = model.layers[0]
final_layer_config = final_layer.get_config()
#print(final_layer_config)
print(json.dumps(final_layer_config, indent=2))

print(colored("type(final_layer_config)", 'cyan'), '\n',
      type(final_layer_config), '\n')
print(colored("final_layer_config.keys()", 'cyan'), '\n',
      final_layer_config.keys(), '\n')
print(colored("final_layer_config.values(", 'cyan'), '\n',
      final_layer_config.values, '\n')

for layer in model.layers:
    layer_config = layer.get_config()

    layer_name = layer_config['name']
    if layer_name.startswith('conv') and len(layer_name.split('_')) <=2:
        print(colored('Layer name:', 'cyan'), layer_name)
        print('n filters: ', layer_config['filters'])
        print('kernel size: ', layer_config['kernel_size'])
        print('padding: ', layer_config['padding'])

#########################################################################
final_layer = model.layers[-1]

final_layer.get_weights() # weight 와 bias 모두 가지고 있는 list