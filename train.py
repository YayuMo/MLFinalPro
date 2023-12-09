from model.MobileNetV2 import MobileNetV2
from model.VGG import VGG16, VGG19
from model.ResNet import ResNet50
from dataGenerator import dataGeneration
from constants import *
import tensorflow as tf
from util import plotAccandLoss

def train(pre_train_model, epoch):
    # set GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    # load data
    train_ds, val_ds, test_ds = dataGeneration(
        data_dir=DATA_DIR,
        img_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        data_split=DATA_SPLIT
    )

    if(pre_train_model == 'MobileNetV2'):
        # freezed training
        modelFreezed = MobileNetV2(
            image_size=IMG_SIZE,
            train_ds=train_ds,
            learning_rate=Learning_Rate,
            freeze=False
        )
        history = modelFreezed.fit(train_ds,
                                   epochs=int(epoch / 2),
                                   validation_data=val_ds)
        # accuracy
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        # loss
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        # Fine Tune
        model = MobileNetV2(
            image_size=IMG_SIZE,
            train_ds=train_ds,
            learning_rate=Learning_Rate,
            freeze=True
        )

        history_fine = model.fit(train_ds,
                                 epochs=epoch,
                                 initial_epoch=history.epoch[-1],
                                 validation_data=val_ds)
        # total acc
        acc += history_fine.history['accuracy']
        val_acc += history_fine.history['val_accuracy']
        # total loss
        loss += history_fine.history['loss']
        val_loss += history_fine.history['val_loss']

        return acc, val_acc, loss, val_loss
    elif(pre_train_model == 'VGG16'):
        # freezed training
        modelFreezed = VGG16(
            image_size=IMG_SIZE,
            train_ds=train_ds,
            learning_rate=Learning_Rate,
            freeze=False
        )
        history = modelFreezed.fit(train_ds,
                                   epochs=int(epoch / 2),
                                   validation_data=val_ds)
        # accuracy
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        # loss
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        # Fine Tune
        model = VGG16(
            image_size=IMG_SIZE,
            train_ds=train_ds,
            learning_rate=Learning_Rate,
            freeze=True
        )

        history_fine = model.fit(train_ds,
                                 epochs=epoch,
                                 initial_epoch=history.epoch[-1],
                                 validation_data=val_ds)
        # total acc
        acc += history_fine.history['accuracy']
        val_acc += history_fine.history['val_accuracy']
        # total loss
        loss += history_fine.history['loss']
        val_loss += history_fine.history['val_loss']

        return acc, val_acc, loss, val_loss
    elif(pre_train_model == 'VGG19'):
        # freezed training
        modelFreezed = VGG19(
            image_size=IMG_SIZE,
            train_ds=train_ds,
            learning_rate=Learning_Rate,
            freeze=False
        )
        history = modelFreezed.fit(train_ds,
                                   epochs=int(epoch / 2),
                                   validation_data=val_ds)
        # accuracy
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        # loss
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        # Fine Tune
        model = VGG19(
            image_size=IMG_SIZE,
            train_ds=train_ds,
            learning_rate=Learning_Rate,
            freeze=True
        )

        history_fine = model.fit(train_ds,
                                 epochs=epoch,
                                 initial_epoch=history.epoch[-1],
                                 validation_data=val_ds)
        # total acc
        acc += history_fine.history['accuracy']
        val_acc += history_fine.history['val_accuracy']
        # total loss
        loss += history_fine.history['loss']
        val_loss += history_fine.history['val_loss']

        return acc, val_acc, loss, val_loss
    elif(pre_train_model == 'ResNet50'):
        # freezed training
        modelFreezed = ResNet50(
            image_size=IMG_SIZE,
            train_ds=train_ds,
            learning_rate=Learning_Rate,
            freeze=False
        )
        history = modelFreezed.fit(train_ds,
                                   epochs=int(epoch / 2),
                                   validation_data=val_ds)
        # accuracy
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        # loss
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        # Fine Tune
        model = ResNet50(
            image_size=IMG_SIZE,
            train_ds=train_ds,
            learning_rate=Learning_Rate,
            freeze=True
        )

        history_fine = model.fit(train_ds,
                                 epochs=epoch,
                                 initial_epoch=history.epoch[-1],
                                 validation_data=val_ds)
        # total acc
        acc += history_fine.history['accuracy']
        val_acc += history_fine.history['val_accuracy']
        # total loss
        loss += history_fine.history['loss']
        val_loss += history_fine.history['val_loss']

        return acc, val_acc, loss, val_loss


if __name__ == '__main__':
    acc1, val_acc1, loss1, val_loss1 = train('ResNet50', epoch=Epoch)
    acc2, val_acc2, loss2, val_loss2 = train('VGG19', epoch=Epoch)
    plotAccandLoss(acc1, val_acc1, loss1, val_loss1, epoch=Epoch)
    plotAccandLoss(acc2, val_acc2, loss2, val_loss2, epoch=Epoch)