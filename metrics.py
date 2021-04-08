import keras.backend as K
from keras import losses
import tensorflow as tf

def toFloat(x):
    return K.cast(x, K.floatx())

def toBool(x):
    return K.cast(x, bool)

    """ Intersection over union functions inspired from:
    https://stackoverflow.com/questions/65974208/intersection-over-union-iou-metric-for-multi-class-semantic-segmentation-task
    """
def intersection_over_union(y_true, y_pred, smooth=1e-07):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou

def iou_loss(y_true, y_pred):
    return 1 - intersection_over_union(y_true, y_pred)

def iou_bce_loss(y_true, y_pred):
    return losses.binary_crossentropy(y_true, y_pred) + 5 * iou_loss(y_true, y_pred)


    """Dice coefficient inspired from:
    https://github.com/keras-team/keras/issues/3611 ,
    but with smooth coefficient with value 100, as per
    https://github.com/juliandewit/kaggle_ndsb2017/issues/21 .
    
    I created two different functions, one for obtaining the score
    itself, and one for the loss.
    """
def dice_coef(y_true, y_pred):
    smooth = 100
    y_true = K.cast(y_true, 'float32')
    y_true_flatten = K.cast(K.flatten(y_true), 'float32')
    y_pred = K.cast(y_pred, 'float32')
    y_pred_flatten = K.cast(K.flatten(y_pred), 'float32')
    #y_pred_flatten = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = K.sum(y_true_flatten * y_pred_flatten)
    score = (2. * intersection + smooth) / (K.sum(y_true_flatten) + K.sum(y_pred_flatten) + smooth)
    return score

def dice_loss(y_true, y_pred):
    smooth = 100
    y_true = K.cast(y_true, 'float32')
    y_true_flatten = K.cast(K.flatten(y_true), 'float32')
    y_pred = K.cast(y_pred, 'float32')
    y_pred_flatten = K.cast(K.flatten(y_pred), 'float32')
    #y_pred_flatten = K.cast(K.greater(K.flatten(y_pred), 0.5), 'float32')
    intersection = K.sum(y_true_flatten * y_pred_flatten)
    score = (2. * intersection + smooth) / (K.sum(y_true_flatten) + K.sum(y_pred_flatten) + smooth)
    return 1. - score



    """Jaccard index is more optimal for unbalanced datasets, taken from:
    https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96
    """
def jaccard_distance_loss(y_true, y_pred, smooth=100):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac)



    """Taken from:
    https://github.com/Atomwh/FocalLoss_Keras/blob/master/focalloss.py
    """
def focal_loss(y_true, y_pred):
    gamma=0.5
    alpha=0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

def competition_metric(true, pred): #any shape can go

    tresholds = [0.5 + (i*.05)  for i in range(10)]

    #flattened images (batch, pixels)
    true = K.batch_flatten(true)
    pred = K.batch_flatten(pred)
    pred = toFloat(K.greater(pred, 0.5))

    #total white pixels - (batch,)
    trueSum = K.sum(true, axis=-1)
    predSum = K.sum(pred, axis=-1)

    #has mask or not per image - (batch,)
    true1 = toFloat(K.greater(trueSum, 1))    
    pred1 = toFloat(K.greater(predSum, 1))

    #to get images that have mask in both true and pred
    truePositiveMask = toBool(true1 * pred1)

    #separating only the possible true positives to check iou
    testTrue = tf.boolean_mask(true, truePositiveMask)
    testPred = tf.boolean_mask(pred, truePositiveMask)

    #getting iou and threshold comparisons
    iou = intersection_over_union(testTrue,testPred) 
    truePositives = [toFloat(K.greater(iou, tres)) for tres in tresholds]

    #mean of thressholds for true positives and total sum
    truePositives = K.mean(K.stack(truePositives, axis=-1), axis=-1)
    truePositives = K.sum(truePositives)

    #to get images that don't have mask in both true and pred
    trueNegatives = (1-true1) * (1 - pred1) # = 1 -true1 - pred1 + true1*pred1
    trueNegatives = K.sum(trueNegatives) 

    return (truePositives + trueNegatives) / toFloat(K.shape(true)[0])

# For threshold determination
def faster_iou_metric_batch(A, B):
    batch_size = A.shape[0]
    metric = []
    for batch in range(batch_size):
        t, p = A[batch], B[batch]
        if np.count_nonzero(t) == 0 and np.count_nonzero(p) > 0:
            metric.append(0)
            continue
        if np.count_nonzero(t) >= 1 and np.count_nonzero(p) == 0:
            metric.append(0)
            continue
        if np.count_nonzero(t) == 0 and np.count_nonzero(p) == 0:
            metric.append(1)
            continue

        intersection = np.logical_and(t, p)
        union = np.logical_or(t, p)
        iou = np.sum(intersection > 0) / np.sum(union > 0)
        thresholds = np.arange(0.5, 1, 0.05)
        s = []
        for thresh in thresholds:
            s.append(iou > thresh)
        metric.append(np.mean(s))

    return np.mean(metric)