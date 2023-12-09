# pre-trained resnet

import numpy as np
import tensorflow as tf

def ResNet50(IMG_SIZE):
    preprocess_input = tf.keras.applications.resnet50.preprocess_input
    IMG_SHAPE = IMG_SIZE + (3,)
    base_model = tf.keras.applications.ResNet50(input_shape=IMG_SHAPE,
                                                include_top=False,
                                                weights='imagenet')
    base_model.summary()

if __name__ == '__main__':
    IMG_SIZE = (224, 224)
    ResNet50(IMG_SIZE)