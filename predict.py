from dataGenerator import dataGeneration
import tensorflow as tf
from constants  import *

# load data
train_ds, val_ds, test_ds = dataGeneration(
    data_dir=DATA_DIR,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    data_split=DATA_SPLIT
)
# load model
new_model = tf.keras.models.load_model('weight/mobilenetv2/mobilenetv2.h5')

loss, accuracy = new_model.evaluate(test_ds)