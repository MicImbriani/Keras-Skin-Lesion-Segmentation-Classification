import numpy as np
import matplotlib as plt

from tqdm import tqdm_notebook

from keras.layers import Input, Lambda, Conv2D, Dropout, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, Concatenate
from keras.models import Model, load_model, model_from_json


import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import tensorflow as tf
import metrics


def conv_block(neurons, block_input, batch_norm=False, middle=False):
    conv1 = Conv2D(neurons, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(block_input)
    if batch_norm:
        conv1 = BatchNormalization()(conv1)
    conv2 = Conv2D(neurons, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    if batch_norm:
        conv2 = BatchNormalization()(conv2)
    if middle:
        conv2 = Dropout(0.5)(conv2)
        return conv2
    pool = MaxPooling2D(pool_size=(2,2))(conv2)
    return pool, conv2

def deconv_block(neurons, block_input, shortcut, batch_norm=False, dropout=None):
    deconv = Conv2D(neurons, (2,2), activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(block_input))
    upconv = Concatenate(axis=3)([deconv, shortcut])
    upconv = Conv2D(neurons, (3, 3), activation='relu', padding="same", kernel_initializer='he_normal')(upconv)
    if batch_norm:
        upconv = BatchNormalization()(upconv)
    upconv = Conv2D(neurons, (3, 3), activation='relu', padding="same", kernel_initializer='he_normal')(upconv)
    if batch_norm:
        upconv = BatchNormalization()(upconv)
    return upconv
    
def unet(input_size, batch_norm=False):
    input_layer = Input(input_size)

    # Down
    conv1, shortcut1 = conv_block(64, input_layer, batch_norm)
    conv2, shortcut2 = conv_block(128, conv1, batch_norm)
    conv3, shortcut3 = conv_block(256, conv2, batch_norm)
    conv4, shortcut4 = conv_block(512, conv3, batch_norm, dropout=True)
    
    # Middle
    convm = conv_block(1024, conv4, batch_norm, dropout=True, middle=True)
    
    # Up
    deconv4 = deconv_block(512, convm, shortcut4, batch_norm)
    deconv3 = deconv_block(256, deconv4, shortcut3, batch_norm)
    deconv2 = deconv_block(128, deconv3, shortcut2, batch_norm)
    deconv1 = deconv_block(64, deconv2, shortcut1, batch_norm)
    
    final_conv = Conv2D(2, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(deconv1)
    output_layer = Conv2D(1, 1, activation="sigmoid")(final_conv)
    
    model = Model(input_layer, output_layer)

    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9, decay=1e-6, nesterov=True),
                  loss=metrics.jaccard_coef,
                  metrics=[metrics.dice_coef_loss,
                           metrics.jaccard_coef_loss,
                           metrics.true_positive,
                           metrics.true_negative
                           ]
                  )
                  
    model.summary()


    return model