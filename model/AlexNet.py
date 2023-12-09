import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D

# import keras.backend.tensorflow_backend as K
# K.set_image_dim_ordering('th')
def AlexNet(num_classses=1000):
    model = Sequential()
    model.add(ZeroPadding2D((2, 2), input_shape=(227, 227, 3)))
    model.add(Convolution2D(64, (11, 11), strides=(4, 4), activation='relu'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))

    model.add(ZeroPadding2D((2, 2)))
    model.add(Convolution2D(192, (5, 5), activation='relu'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(384, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((3, 3), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(num_classses, activation='softmax'))
    model.add(Dense(num_classses, activation='softmax'))

    return model

model = AlexNet(num_classses=1000)
model.summary()
# model.save_weights('alexnet_weights.h5')
model.load_weights('../weight/pretrain/alexnet_weights_pytorch.h5')
for layer in model.layers:
    for weight in layer.weights:
        print(weight.name, weight.shape)
