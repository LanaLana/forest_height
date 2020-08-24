from keras import backend as K
import tensorflow as tf
import numpy as np
import math

#---------------------------------------------------------------------------------------
# binary segmentation
#---------------------------------------------------------------------------------------
def weighted_binary_crossentropy(y_true, y_pred):
    eps = 1e-12
    y_pred = K.clip(y_pred, eps, 1. - eps)
    weights = tf.where(tf.equal(y_true, 0.), tf.ones_like(y_true), y_true) 
    weights = tf.where(tf.equal(weights, 2.2), tf.ones_like(weights)*1.2, weights)
    weights = tf.where(tf.equal(weights, 1.2), tf.ones_like(weights)*1.2, weights)
    y_true = tf.where(tf.equal(y_true, 2.2), tf.zeros_like(y_true), y_true)
    y_true = tf.where(tf.equal(y_true, 1.2), tf.ones_like(y_true), y_true)
    y_true = K.clip(y_true, 0., 1.)
    return -K.mean((y_true * K.log(y_pred + 1e-15) + (1. - y_true) * K.log(1. - y_pred + 1e-15 ))*weights)

#---------------------------------------------------------------------------------------
# regression
#---------------------------------------------------------------------------------------
def rmse(y_true, y_pred):
    non_zero = tf.where(tf.equal(y_true, 0), tf.zeros_like(y_true), tf.ones_like(y_true))
    return K.sqrt(tf.reduce_sum(K.square(y_pred*non_zero - y_true)) / K.sum(non_zero)) 

def weighted_rmse(values):
    def loss(y_true, y_pred):
        weights = tf.ones_like(y_true)
        for i in range(29):
            weights += tf.where(tf.greater(y_true*30, i + 0.5), tf.ones_like(y_true), tf.zeros_like(y_true)) * \
                tf.where(tf.less_equal(y_true*30, i+1.5), tf.ones_like(y_true), tf.zeros_like(y_true)) * values[i]
        non_zero = tf.where(tf.equal(y_true, 0), tf.zeros_like(y_true), tf.ones_like(y_true))
        _epsilon = tf.convert_to_tensor(K.epsilon(), dtype=y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
        tmp_sum = tf.reduce_sum(K.square(y_pred*non_zero - y_true)*weights) / K.sum(non_zero)
        return K.sqrt(tmp_sum)
    return loss

def mae(y_true, y_pred):
    non_zero = tf.where(tf.equal(y_true, 0), tf.zeros_like(y_true), tf.ones_like(y_true))
    return tf.reduce_sum(K.abs(y_pred*non_zero - y_true)) / K.sum(non_zero)    


#---------------------------------------------------------------------------------------
# classification
#---------------------------------------------------------------------------------------
def weighted_categorical_crossentropy(weights, batch_size=20, IMG_ROW=256, IMG_COL=256):
    def loss(target,output,from_logits=False):
        output /= tf.reduce_sum(output,
                                len(output.get_shape()) - 1,
                                True)
        non_zero_pixels = tf.reduce_sum(target, axis=-1)
        _epsilon = tf.convert_to_tensor(K.epsilon(), dtype=output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
        weighted_losses = target * tf.log(output) * weights
        return - tf.reduce_sum(weighted_losses,len(output.get_shape()) - 1) * \
                        (IMG_ROW*IMG_COL*batch_size) / K.sum(non_zero_pixels)
    
    return loss
    

def create_class_weight(labels_dict,mu=0.15):
    total = np.sum(list(labels_dict.values()))
    keys = labels_dict.keys()
    class_weight = dict()
    weights_list = np.zeros((len(keys)))
    for key in keys:
        score = math.log(mu*total/float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0
        weights_list[sorted(keys).index(key)] = class_weight[key]
    return class_weight, weights_list