# pre-trained vgg

import numpy as np
import tensorflow as tf

def VGG16(IMG_SIZE):
    preprocess_input = tf.keras.applications.vgg16.preprocess_input
    IMG_SHAPE = IMG_SIZE + (3,)
    base_model = tf.keras.applications.VGG16(input_shape=IMG_SHAPE,
                                             include_top=False,
                                             weights='imagenet')
    base_model.summary()

def VGG19(IMG_SIZE):
    preprocess_input = tf.keras.applications.vgg19.preprocess_input
    IMG_SHAPE = IMG_SIZE + (3,)
    base_model = tf.keras.applications.VGG19(input_shape=IMG_SHAPE,
                                             include_top=False,
                                             weights='imagenet')
    base_model.summary()

if __name__ == '__main__':
    IMG_SIZE = (224, 224)
    VGG16(IMG_SIZE)
    VGG19(IMG_SIZE)
