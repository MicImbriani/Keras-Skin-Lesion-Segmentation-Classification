import keras.backend as K
from keras import losses
import tensorflow as tf

def float_flatten(x):
    x = K.flatten(x)
    return K.cast(x, K.floatx())



def true_positive(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pos = K.round(K.clip(y_true, 0, 1))
    tp = (K.sum(y_pos * y_pred_pos) + smooth)/ (K.sum(y_pos) + smooth) 
    return tp 

def true_negative(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos 
    tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth )
    return tn 

def dice_coef(y_true, y_pred, smooth=1):
    y_true = float_flatten(y_true)
    y_pred = float_flatten(y_pred)
    intersection = K.sum(y_true * y_pred, axis=[0,1,2])
    union = K.sum(y_true, axis=[0,1,2]) + K.sum(y_pred, axis=[0,1,2])
    return K.mean( (2. * intersection + smooth) / (union + smooth))

def dice_coef_loss(y_true, y_pred):
    dice = dice_coef(y_true, y_pred)
    return (1 - dice)

def jaccard_coef(y_true, y_pred, smooth=1):
    y_true = float_flatten(y_true)
    y_pred = float_flatten(y_pred)
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    summation = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (summation - intersection + smooth)
    return jac

def jaccard_coef_loss(y_true, y_pred, smooth=1):
    jac = jaccard_coef_c(y_true, y_pred)
    return (1 - jac)