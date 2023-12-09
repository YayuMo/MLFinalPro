import random

import matplotlib.pyplot as plt

def plotAccandLoss(acc, val_acc, loss, val_loss, epoch):
    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    # plt.ylim([0.8, 1])
    plt.plot([int(epoch / 2), int(epoch / 2)],
             plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    # plt.ylim([0, 1.0])
    plt.plot([int(epoch / 2), int(epoch / 2)],
             plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()