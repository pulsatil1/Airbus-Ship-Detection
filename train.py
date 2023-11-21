import gc
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from keras import models, layers
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import config
from functions import losses, generator

train = os.listdir(config.TRAIN_DIR)
test = os.listdir(config.TEST_DIR)

# Data Preparation
masks = pd.read_csv(os.path.join(
    config.BASE_DIR, 'train_ship_segmentations_v2.csv'))

masks['ships'] = masks['EncodedPixels'].map(
    lambda c_row: 1 if isinstance(c_row, str) else 0)
unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
unique_img_ids['has_ship'] = unique_img_ids['ships'].map(
    lambda x: 1.0 if x > 0 else 0.0)
masks.drop(['ships'], axis=1, inplace=True)

# Undersample Empty Images
balanced_train_df = unique_img_ids.groupby('ships').apply(
    lambda x: x.sample(config.SAMPLES_PER_GROUP) if len(x) > config.SAMPLES_PER_GROUP else x)

# Split & Image generators
train_ids, valid_ids = train_test_split(balanced_train_df,
                                        test_size=0.2,
                                        stratify=balanced_train_df['ships'])
train_df = pd.merge(masks, train_ids)
valid_df = pd.merge(masks, valid_ids)

train_gen = generator.make_image_gen(train_df)
valid_x, valid_y = next(generator.make_image_gen(
    valid_df, config.VALID_IMG_COUNT))

# Data Augmentation
dg_args = dict(featurewise_center=False,
               samplewise_center=False,
               rotation_range=45,
               width_shift_range=0.1,
               height_shift_range=0.1,
               shear_range=0.01,
               zoom_range=[0.9, 1.25],
               horizontal_flip=True,
               vertical_flip=True,
               fill_mode='reflect',
               data_format='channels_last')

gc.enable()
gc.collect()

# Base Model


def upsample_conv(filters, kernel_size, strides, padding):
    return layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)


def upsample_simple(filters, kernel_size, strides, padding):
    return layers.UpSampling2D(strides)


if config.UPSAMPLE_MODE == 'DECONV':
    upsample = upsample_conv
else:
    upsample = upsample_simple


def unet(pretrained_weights=None, input_size=(256, 256, 3), NET_SCALING=config.NET_SCALING):
    inputs = layers.Input(input_size)

    c1 = layers.Conv2D(16, (3, 3), activation='relu',
                       kernel_initializer='he_normal', padding='same')(inputs)
    c1 = layers.Dropout(0.1)(c1)
    c1 = layers.Conv2D(16, (3, 3), activation='relu',
                       kernel_initializer='he_normal', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(32, (3, 3), activation='relu',
                       kernel_initializer='he_normal', padding='same')(p1)
    c2 = layers.Dropout(0.1)(c2)
    c2 = layers.Conv2D(32, (3, 3), activation='relu',
                       kernel_initializer='he_normal', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(64, (3, 3), activation='relu',
                       kernel_initializer='he_normal', padding='same')(p2)
    c3 = layers.Dropout(0.2)(c3)
    c3 = layers.Conv2D(64, (3, 3), activation='relu',
                       kernel_initializer='he_normal', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(128, (3, 3), activation='relu',
                       kernel_initializer='he_normal', padding='same')(p3)
    c4 = layers.Dropout(0.2)(c4)
    c4 = layers.Conv2D(128, (3, 3), activation='relu',
                       kernel_initializer='he_normal', padding='same')(c4)
    p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = layers.Conv2D(256, (3, 3), activation='relu',
                       kernel_initializer='he_normal', padding='same')(p4)
    c5 = layers.Dropout(0.3)(c5)
    c5 = layers.Conv2D(256, (3, 3), activation='relu',
                       kernel_initializer='he_normal', padding='same')(c5)

    u6 = layers.Conv2DTranspose(
        128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(128, (3, 3), activation='relu',
                       kernel_initializer='he_normal', padding='same')(u6)
    c6 = layers.Dropout(0.2)(c6)
    c6 = layers.Conv2D(128, (3, 3), activation='relu',
                       kernel_initializer='he_normal', padding='same')(c6)

    u7 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(64, (3, 3), activation='relu',
                       kernel_initializer='he_normal', padding='same')(u7)
    c7 = layers.Dropout(0.2)(c7)
    c7 = layers.Conv2D(64, (3, 3), activation='relu',
                       kernel_initializer='he_normal', padding='same')(c7)

    u8 = layers.Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(32, (3, 3), activation='relu',
                       kernel_initializer='he_normal', padding='same')(u8)
    c8 = layers.Dropout(0.1)(c8)
    c8 = layers.Conv2D(32, (3, 3), activation='relu',
                       kernel_initializer='he_normal', padding='same')(c8)

    u9 = layers.Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = layers.Conv2D(16, (3, 3), activation='relu',
                       kernel_initializer='he_normal', padding='same')(u9)
    c9 = layers.Dropout(0.1)(c9)
    c9 = layers.Conv2D(16, (3, 3), activation='relu',
                       kernel_initializer='he_normal', padding='same')(c9)

    d = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    seg_model = models.Model(inputs=[inputs], outputs=[d])

    return seg_model


seg_model = unet()

# callbacks setting
weight_path = "fullres_model & weights/{}_weights.best.hdf5".format(
    'seg_model')

checkpoint = ModelCheckpoint(weight_path, monitor='val_dice_coef', verbose=1,
                             save_best_only=True, mode='max', save_weights_only=True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_dice_coef', factor=0.5,
                                   patience=3,
                                   verbose=1, mode='max', epsilon=0.0001, cooldown=2, min_lr=1e-6)
early = EarlyStopping(monitor="val_dice_coef",
                      mode="max",
                      patience=15)
callbacks_list = [checkpoint, early, reduceLROnPlat]


seg_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=losses.FocalLoss,
                  metrics=[losses.dice_coef, 'binary_accuracy'])

step_count = min(config.MAX_TRAIN_STEPS,
                 train_df.shape[0]//config.BATCH_SIZE)
aug_gen = generator.create_aug_gen(generator.make_image_gen(train_df))
loss_history = [seg_model.fit(aug_gen,
                              steps_per_epoch=step_count,
                              epochs=config.MAX_TRAIN_EPOCHS,
                              validation_data=(valid_x, valid_y),
                              callbacks=callbacks_list,
                              workers=1
                              )]

seg_model.save('seg_model.h5')
