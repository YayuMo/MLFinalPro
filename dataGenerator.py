import tensorflow as tf

def dataGeneration(DATA_DIR, IMG_SIZE, BATCH_SIZE, DATA_SPLIT):

    # generate 20 batches with specific size
    # train dataset
    train_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=DATA_SPLIT,
        subset="training",
        shuffle=True,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE
    )
    # validation dataset
    val_ds = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        validation_split=DATA_SPLIT,
        subset="validation",
        shuffle=True,
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE
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


if __name__ == '__main__':
    pass