import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from constants import *
from dataGenerator import dataGeneration

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

def alexPreModel(image_size, train_ds, learning_rate, freeze):
    # Normalizing
    image_shape = image_size + (3,)
    rescale_layer = tf.keras.layers.Rescaling(1./255)

    # data augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2)
    ])
    # add pre-trained model
    base_model = AlexNet(num_classses=1000)
    base_model.load_weights('weight/pretrain/alexnet_weights_pytorch.h5')

    # Feature Extraction
    image_batch, label_batch = next(iter(train_ds))
    feature_batch = base_model(image_batch)

    # global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    # feature_batch_average = global_average_layer(feature_batch)

    base_model.trainable = freeze  # freezing training
    # base_model.summary()

    # build classifier
    prediction_layer = tf.keras.layers.Dense(1, 'sigmoid')
    # prediction_batch = prediction_layer(feature_batch_average)

    # build model structure
    inputs = tf.keras.Input(shape=image_shape)
    x = data_augmentation(inputs)
    x = rescale_layer(x)
    x = base_model(x, training=False)
    # x = global_average_layer(x)
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
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate / 10),
            metrics=['accuracy']
        )

    return model


# model = AlexNet(num_classses=1000)
# model.summary()
# # model.save_weights('alexnet_weights.h5')
# model.load_weights('../weight/pretrain/alexnet_weights_pytorch.h5')
# for layer in model.layers:
#     for weight in layer.weights:
#         print(weight.name, weight.shape)

if __name__ == '__main__':
    train_ds, val_ds, test_ds = dataGeneration(
        data_dir=DATA_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        data_split=DATA_SPLIT
    )
    # IMG_SIZE = (224, 224)
    model = alexPreModel(
        image_size=IMG_SIZE,
        train_ds=train_ds,
        learning_rate=Learning_Rate,
        freeze=False
    )
    # model summary
    model.summary()