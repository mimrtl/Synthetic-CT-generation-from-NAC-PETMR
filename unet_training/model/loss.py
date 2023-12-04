#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
from keras import backend as K
from global_dict.w_global import gbl_get_value, gbl_set_value
from keras.models import load_model
import tensorflow as tf
from keras.models import Model
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'

SEED = 314


# define mean square error
def mean_squared_error_12(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=[1, 2])


def mean_squared_error_14(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=[1, 2, 3, 4])


def mse(y_true, y_pred):
    loss = K.mean(K.square(y_pred - y_true), axis=-1)
    return loss * 1e6


def mae(y_true, y_pred):
    loss = K.mean(K.abs(y_pred - y_true), axis=-1)
    return loss * 1e6


def focusMae(y_true, y_pred):
    ones = tf.ones_like(y_true)
    zeros = tf.zeros_like(y_true)
    label = tf.where(y_true > 0.3433333333333333333334, zeros, y_true)
    label = tf.where(label < 0.33, zeros, ones)
    loss = K.mean(K.abs(y_pred - y_true) * label)
    return loss * 1e7 * 3


def maeWithFocus(y_true, y_pred):
    loss1 = mae(y_true, y_pred)
    loss2 = focusMae(y_true, y_pred)
    return loss1 + loss2


# compute the gram matrix which is required to compute the style loss
def gram_matrix(x):
    if K.ndim(x) == 5:
        x = K.permute_dimensions(x, (0, 4, 1, 2, 3))
        shape = K.shape(x)
        B, C, H, W, D = shape[0], shape[1], shape[2], shape[3], shape[4]
        features = K.reshape(x, K.stack([B, C, H*W*D]))
        gram = K.batch_dot(features, features, axes=2)
        denominator = C * H * W
        gram = gram / K.cast(denominator, x.dtype)
        return gram
    else:
        raise ValueError('the dimension is not correct!')


# Description: compute style loss
def style(y_pred, y_true):
    style_weight = gbl_get_value('style_weight')
    batch_size = gbl_get_value("batch_size")
    vgg_path = gbl_get_value('vgg_path')
    size_x = gbl_get_value('size_x')
    size_y = gbl_get_value('size_y')
    size_z = gbl_get_value('size_z')
    color_mode = gbl_get_value('color_mode')

    data_format = K.image_data_format()
    if data_format == 'channels_last':
        y_true = K.reshape(y_true, (-1, size_x, size_y, size_z, color_mode))
        y_pred = K.reshape(y_pred, (-1, size_x, size_y, size_z, color_mode))
    else:
        y_true = K.reshape(y_true, (-1, color_mode, size_x, size_y, size_z))
        y_pred = K.reshape(y_pred, (-1, color_mode, size_x, size_y, size_z))

    print('+++++++++++++++++++++++++++++++++++++')
    print('style_loss')
    print(y_true.shape)
    print(y_pred.shape)
    print('+++++++++++++++++++++++++++++++++++++')

    # load the model
    model_path = vgg_path
    model = load_model(model_path, compile=False)
    for layer in model.layers:
        layer.trainable = False

    selected_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'predictions']
    selected_output = [model.get_layer(name).output for name in selected_layers]
    layer_model = Model(inputs=model.input, outputs=selected_output)

    feature_pred = layer_model(y_pred)
    feature_true = layer_model(y_true)

    # feature_gram
    gram_pred = []
    gram_true = []
    for i in range(len(feature_pred) - 1):
        gram_pred.append(gram_matrix(feature_pred[i]))
        gram_true.append(gram_matrix(feature_true[i]))

    style_loss = 0
    for i in range(len(selected_layers) - 1):
        temp = mean_squared_error_12(gram_true[i], gram_pred[i][:batch_size])
        style_loss += temp
    style_loss = style_weight * style_loss

    return style_loss


# Description: compute content loss
def content(y_true, y_pred):
    content_weight = gbl_get_value('content_weight')
    vgg_path = gbl_get_value('vgg_path')
    size_x = gbl_get_value('size_x')
    size_y = gbl_get_value('size_y')
    size_z = gbl_get_value('size_z')
    color_mode = gbl_get_value('color_mode')

    data_format = K.image_data_format()
    if data_format == 'channels_last':
        y_true = K.reshape(y_true, (-1, size_x, size_y, size_z, color_mode))
        y_pred = K.reshape(y_pred, (-1, size_x, size_y, size_z, color_mode))
    else:
        y_true = K.reshape(y_true, (-1, color_mode, size_x, size_y, size_z))
        y_pred = K.reshape(y_pred, (-1, color_mode, size_x, size_y, size_z))

    print('+++++++++++++++++++++++++++++++++++++')
    print('content_loss')
    print(y_true.shape)
    print(y_pred.shape)
    print('+++++++++++++++++++++++++++++++++++++')

    # load the model
    model_path = vgg_path
    model = load_model(model_path, compile=False)
    for layer in model.layers:
        layer.trainable = False

    selected_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'predictions']
    selected_output = [model.get_layer(name).output for name in selected_layers]
    layer_model = Model(inputs=model.input, outputs=selected_output)

    feature_pred = layer_model(y_pred)
    feature_true = layer_model(y_true)

    content_loss = mean_squared_error_14(feature_true[2], feature_pred[2])
    content_loss = content_weight * content_loss

    return content_loss


def perceptual_loss(y_true, y_pred):
    batch_size = gbl_get_value("batch_size")
    style_weight = gbl_get_value('style_weight')
    content_weight = gbl_get_value('content_weight')
    vgg_path = gbl_get_value('vgg_path')
    size_x = gbl_get_value('size_x')
    size_y = gbl_get_value('size_y')
    size_z = gbl_get_value('size_z')
    color_mode = gbl_get_value('color_mode')

    data_format = K.image_data_format()
    if data_format == 'channels_last':
        y_true = K.reshape(y_true, (-1, size_x, size_y, size_z, color_mode))
        y_pred = K.reshape(y_pred, (-1, size_x, size_y, size_z, color_mode))
    else:
        y_true = K.reshape(y_true, (-1, color_mode, size_x, size_y, size_z))
        y_pred = K.reshape(y_pred, (-1, color_mode, size_x, size_y, size_z))

    print('+++++++++++++++++++++++++++++++++++++')
    print('perceptual_loss')
    print(y_true.shape)
    print(y_pred.shape)
    print('+++++++++++++++++++++++++++++++++++++')

    # load the model
    print(vgg_path)
    model_path = vgg_path
    model = load_model(model_path, compile=False)
    for layer in model.layers:
        layer.trainable = False

    selected_layers = ['block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'predictions']
    selected_output = [model.get_layer(name).output for name in selected_layers]
    layer_model = Model(inputs=model.input, outputs=selected_output)

    feature_pred = layer_model(y_pred)
    feature_true = layer_model(y_true)

    # feature_gram
    gram_pred = []
    gram_true = []
    for i in range(len(feature_pred)-1):
        gram_pred.append(gram_matrix(feature_pred[i]))
        gram_true.append(gram_matrix(feature_true[i]))

    style_loss = 0
    for i in range(len(selected_layers)-1):
        temp = mean_squared_error_12(gram_true[i], gram_pred[i][:batch_size])
        style_loss += temp
    style_loss = style_weight * style_loss

    content_loss = mean_squared_error_14(feature_true[2], feature_pred[2])
    content_loss = content_weight * content_loss

    final_percep_loss = style_loss + content_loss
    return final_percep_loss
