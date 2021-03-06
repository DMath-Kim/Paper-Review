import os
import keras
import keras.backend as K
import cv2
import numpy as np
import math
from keras.models import Model
from keras.layers import *
from keras.datasets import cifar10
from keras.utils import np_utils
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from keras.callbacks import LearningRateScheduler

num_classes = 10

def load_cifar10_data(img_rows, img_cols):

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    x_train = x_train[0:5000, :, :, :]
    y_train = y_train[0:5000]

    x_test = x_test[0:500, :, :, :]
    y_test = y_test[0:500]

    # resize training images
    x_train = np.array([cv2.resize(img, (img_rows, img_cols)) for img in x_train[:, :, :, :]])
    x_test = np.array([cv2.resize(img, (img_rows, img_cols)) for img in x_test[:, :, :, :]])

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')

    x_train = x_train / 255.0
    x_test = x_test / 255.0

    return x_train, y_train, x_test, y_test

def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj, name=None, kernel_init='glorot_uniform', bias_init='zeros'):

    conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_3x3_reduce = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3_reduce)
    conv_5x5_reduce = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_5x5_reduce)
    max_pool = MaxPool2D((3, 3), strides=(1, 1), padding='same')(x)
    pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(max_pool)

    output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)

    return output

def decay(epoch, steps=100):
    initial_lrate = 0.01
    drop = 0.96
    epoch_drop = 8
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch) / epoch_drop))

    return lrate

def googlenet():

    x_train, y_train, x_valid, y_valid = load_cifar10_data(224, 224)
    kernel_init = keras.initializers.glorot_uniform()
    bias_init = keras.initializers.Constant(value=0.2)
    input_layer = Input(shape=(224, 224, 3))

    # Layer 1
    x = Conv2D(64, (7, 7), padding='same', strides=(2, 2), activation='relu', name='conv_1_7x7/2', kernel_initializer=kernel_init, bias_initializer=bias_init)(input_layer)
    x = MaxPool2D((3, 3), strides=(2, 2), name='max_pool_1_3x3/2', padding='same')(x)

    # Layer 2
    x = Conv2D(192, (3, 3), padding='same', strides=(1, 1), activation='relu', name='conv_2_3x3/1', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
    x = MaxPool2D((3, 3), strides=(2, 2), name='max_pool_2_3x3/2', padding='same')(x)

    # Layer 3
    x = inception_module(x, 64, 96, 128, 16, 32, 32, name='inception_3a', kernel_init=kernel_init, bias_init=bias_init)
    x = inception_module(x, 128, 128, 192, 32, 96, 64, name='inception_3b', kernel_init=kernel_init, bias_init=bias_init)
    x = MaxPool2D((3, 3), strides=(2, 2), name='max_pool_3_3x3/2')(x)

    # Layer 4
    x = inception_module(x, 192, 96, 208, 16, 48, 64, name='inception_4a')

    # Layer 4 - Auxiliary Learning 1
    x1 = AveragePooling2D((5, 5), strides=3, name='avg_pool_aux_1')(x)
    x1 = Conv2D(128, (1, 1), padding='same', activation='relu', name='conv_aux_1')(x1)
    x1 = Flatten()(x1)
    x1 = Dense(1024, activation='relu', name='dense_aux_1')(x1)
    x1 = Dropout(0.7)(x1)
    x1 = Dense(10, activation='softmax', name='aux_output_1')(x1)

    x = inception_module(x, 160, 112, 224, 24, 64, 64, name='inception_4b', kernel_init=kernel_init, bias_init=bias_init)
    x = inception_module(x, 128, 128, 256, 24, 64, 64, name='inception_4c', kernel_init=kernel_init, bias_init=bias_init)
    x = inception_module(x, 112, 144, 288, 32, 64, 64, name='inception_4d', kernel_init=kernel_init, bias_init=bias_init)

    # Layer 4 - Auxiliary Learning 2
    x2 = AveragePooling2D((5, 5), strides=3, name='avg_pool_aux_2')(x)
    x2 = Conv2D(128, (1, 1), padding='same', activation='relu', name='conv_aux_2')(x2)
    x2 = Flatten()(x2)
    x2 = Dense(1024, activation='relu', name='dense_aux_2')(x2)
    x2 = Dropout(0.7)(x2)
    x2 = Dense(10, activation='softmax', name='aux_output_2')(x2)

    x = inception_module(x, 256, 160, 320, 32, 128, 128, name='inception_4e', kernel_init=kernel_init, bias_init=bias_init)
    x = MaxPool2D((3, 3), strides=(2, 2), name='max_pool_4_3x3/2')(x)

    # Layer 5
    x = inception_module(x, 256, 160, 320, 32, 128, 128, name='inception_5a', kernel_init=kernel_init, bias_init=bias_init)
    x = inception_module(x, 384, 192, 384, 48, 128, 128, name='inception_5b', kernel_init=kernel_init, bias_init=bias_init)
    x = GlobalAveragePooling2D(name='global_avg_pool_5_3x3/1')(x)
    x = Dropout(0.4)(x)
    x = Dense(10, activation='softmax', name='output')(x)

    model = Model(input_layer, [x, x1, x2], name='inception_v1')

    model.summary()

    epoch = 25
    initial_lrate = 0.01

    sgd = SGD(lr=initial_lrate, momentum=0.9, nesterov=False)
    lr_sc = LearningRateScheduler(decay, verbose=1)

    model.compile(loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
                  loss_weights=[1, 0.3, 0.3], optimizer=sgd, metrics=['accuracy'])

    history = model.fit(x_train, [y_train, y_train, y_train], validation_data=(x_valid, [y_valid, y_valid, y_valid]),
                        epochs=epoch, batch_size=20, callbacks=[lr_sc])


googlenet()
