# pre-trained MobileNetV2
import numpy as np
import tensorflow as tf
from dataGenerator import dataGeneration
from constants import *

def MobileNetV2(image_size,train_ds,learning_rate, freeze):
    # Normalizing
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    # scale = tf.keras.layers.Rescaling(1./ 127.5, offset=-1)
    image_shape = image_size + (3,)

    # data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2)
    ])

    # add pre-trained model
    base_model = tf.keras.applications.MobileNetV2(input_shape=image_shape,
                                                   include_top=False,
                                                   weights='imagenet')

    # Feature Extraction
    image_batch, label_batch = next(iter(train_ds))
    feature_batch = base_model(image_batch)

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)

    base_model.trainable = freeze # freezing training
    # base_model.summary()

    # build classifier
    prediction_layer = tf.keras.layers.Dense(1, 'sigmoid')
    # prediction_batch = prediction_layer(feature_batch_average)

    # build model structure
    inputs = tf.keras.Input(shape=image_shape)
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = tf.keras.Model(inputs, outputs)

    if(freeze == False):
        # compile file
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
    else:
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=learning_rate / 10),
            metrics=['accuracy']
        )

    return model


if __name__ == '__main__':
    train_ds, val_ds, test_ds = dataGeneration(
        data_dir=DATA_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        data_split=DATA_SPLIT
    )
    # IMG_SIZE = (224, 224)
    model = MobileNetV2(
        image_size=IMG_SIZE,
        train_ds=train_ds,
        learning_rate=Learning_Rate,
        freeze=False
    )
    # model summary
    model.summary()