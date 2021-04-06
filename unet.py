import numpy as np
import matplotlib as plt

from tqdm import tqdm_notebook

from keras.layers import Input, Lambda, Conv2D, SpatialDropout2D, BatchNormalization, Activation, MaxPooling2D, Conv2DTranspose, concatenate
from keras.models import Model, load_model, model_from_json


def conv_block(neurons, block_input, batch_norm=False, dropout=None, middle=False):
    conv1 = Conv2D(neurons, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(block_input)
    if batch_norm:
        conv1 = BatchNormalization()(conv1)
    if dropout is not None:
        conv1 = SpatialDropout2D(dropout)(conv1)
    conv2 = Conv2D(neurons, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    if batch_norm:
        conv2 = BatchNormalization()(conv2)
    if dropout is not None:
        conv2 = SpatialDropout2D(dropout)(conv2)
    if middle is True:
        return conv2
    pool = MaxPooling2D((2,2))(conv2)
    return pool, conv2

def deconv_block(neurons, block_input, shortcut, batch_norm=False, dropout=None):
    deconv = Conv2DTranspose(neurons, (3, 3), activation='relu', strides=(2, 2), padding="same")(block_input)
    uconv = concatenate([deconv, shortcut])
    uconv = Conv2D(neurons, (2, 2), padding="same", kernel_initializer='he_normal')(uconv)
    if batch_norm:
        uconv = BatchNormalization()(uconv)
    #uconv = Activation('relu')(uconv)
    if dropout is not None:
        uconv = SpatialDropout2D(dropout)(uconv)
    uconv = Conv2D(neurons, (3, 3), activation='relu', padding="same", kernel_initializer='he_normal')(uconv)
    if batch_norm:
        uconv = BatchNormalization()(uconv)
    #uconv = Activation('relu')(uconv)
    if dropout is not None:
        uconv = SpatialDropout2D(dropout)(uconv)
    return uconv
    
    
def build_model(input_size, batch_norm=False):
    #size = (64, 64, 1)
    input_layer = Input(input_size)

    # Down
    conv1, shortcut1 = conv_block(64, input_layer, batch_norm)
    conv2, shortcut2 = conv_block(128, conv1, batch_norm)
    conv3, shortcut3 = conv_block(256, conv2, batch_norm)
    conv4, shortcut4 = conv_block(512, conv3, batch_norm, dropout=0.5)
    
    # Middle
    convm = conv_block(1024, conv4, batch_norm, middle=True)
    
    # Up
    deconv4 = deconv_block(512, convm, shortcut4, batch_norm)
    deconv3 = deconv_block(256, deconv4, shortcut3, batch_norm)
    deconv2 = deconv_block(128, deconv3, shortcut2, batch_norm)
    deconv1 = deconv_block(64, deconv2, shortcut1, batch_norm)
    #final_conv = Conv2D(2, (3,3), activation='relu', padding='same', kernel_initializer='he_normal')(deconv1)

    output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(deconv1)
    
    model = Model(input_layer, output_layer)
    return model