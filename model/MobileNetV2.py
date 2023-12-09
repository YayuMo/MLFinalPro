# pre-trained MobileNetV2
import numpy as np
import tensorflow as tf

def MobileNetV2(IMG_SIZE):
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    # scale = tf.keras.layers.Rescaling(1./ 127.5, offset=-1)
    IMG_SHAPE = IMG_SIZE + (3,)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')
    base_model.trainable = False # freezing training
    base_model.summary()
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    # feature_batch_average = global_average_layer(feature_batch)




if __name__ == '__main__':
    IMG_SIZE =  (160, 160)
    # IMG_SIZE = (224, 224)
    MobileNetV2(IMG_SIZE)