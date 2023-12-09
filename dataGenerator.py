import tensorflow as tf
from constants import *

def dataGeneration(data_dir, img_size, batch_size, data_split):

    # generate 20 batches with specific size
    # train dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=data_split,
        subset="training",
        shuffle=True,
        seed=123,
        batch_size=batch_size,
        image_size=img_size
    )
    # validation dataset
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=data_split,
        subset="validation",
        shuffle=True,
        seed=123,
        batch_size=batch_size,
        image_size=img_size
    )

    # create test set from validation set
    val_batches = tf.data.experimental.cardinality(val_ds)
    test_ds = val_ds.take(val_batches // 5)
    val_ds = val_ds.skip(val_batches // 5)

    # AUTO TUNE prevent I/O block
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds

# test
if __name__ == '__main__':
    train_ds, val_ds, test_ds = dataGeneration(
        data_dir=DATA_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        data_split=DATA_SPLIT
    )

    # print(len(train_ds))
    # print(len(val_ds))
    # print(len(test_ds))