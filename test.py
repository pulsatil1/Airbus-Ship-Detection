from matplotlib import pyplot as plt
from tensorflow import keras
import config
from functions import generator, losses

test_imgs = ['00dc34840.jpg', '00c3db267.jpg',
             '00aa79c47.jpg', '00a3a9d72.jpg']

seg_model = keras.models.load_model(
    'seg_model.h5', custom_objects={"FocalLoss": losses.FocalLoss, "dice_coef": losses.dice_coef})

rows = 1
columns = 2
for i in range(len(test_imgs)):
    img, pred = generator.gen_pred(
        config.TEST_DIR, test_imgs[i], seg_model)
    fig = plt.figure(figsize=(10, 7))
    fig.add_subplot(rows, columns, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title("Image")
    fig.add_subplot(rows, columns, 2)
    plt.imshow(pred, interpolation=None)
    plt.axis('off')
    plt.title("Prediction")
    plt.show()
