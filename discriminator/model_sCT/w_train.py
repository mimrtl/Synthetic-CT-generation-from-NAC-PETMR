#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import math
import numpy as np
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from global_dict.w_global import gbl_get_value
from model_sCT.vgg_3d import vgg
from skimage.util import random_noise
from keras.models import load_model
from model_sCT.nii_generator_aug import *
SEED = 314

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.45
sess = tf.Session(config=config)
KTF.set_session(sess)


def train_a_vgg(pretrained_model=None):
    gp_id = gbl_get_value("gp_id")
    train_path = gbl_get_value("train_path")
    val_path = gbl_get_value("val_path")
    method = gbl_get_value("method")
    rounds = gbl_get_value("rounds")

    slice_x = gbl_get_value("slice_x")
    color_mode = gbl_get_value('color_mode')
    model_id = gbl_get_value("model_id")
    dir_model = gbl_get_value('dir_model')

    epochs = gbl_get_value("n_epoch")
    batch_size = gbl_get_value("batch_size")
    flag_save = True

    size_x = gbl_get_value('size_x')
    size_y = gbl_get_value('size_y')
    size_z = gbl_get_value('size_z')

    stride_x = gbl_get_value('stride_x')
    stride_y = gbl_get_value('stride_y')
    stride_z = gbl_get_value('stride_z')

    flag_save = True

    # ----------------------------------------------Configurations----------------------------------------------#

    # logs
    log_path = '../logs/gp_' + gp_id + '/' + str(size_x) + '-' + str(size_y) + '-' + str(size_z) + '/' + str(method) + '/' + str(rounds) + '/' + model_id + '/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    tensorboard = TensorBoard(log_dir=log_path, batch_size=batch_size,
                              write_graph=True, write_grads=True,
                              write_images=True)

    # set traininig configurations the learning rate must be 1e5
    conf = {"input_shape": (size_x, size_y, size_z, slice_x), "learning_rate": 1e-5,
            "decay": 0.0, "epsilon": 1e-8, "beta_1": 0.9, "beta_2": 0.999,
            "validation_split": 0.25, "batch_size": batch_size, "epochs": epochs, "model_id": model_id}
    np.save(log_path + model_id + '_info.npy', conf)

    # set augmentation configurations
    conf_a = {"rotation_range": 15, "shear_range": 10,
              "width_shift_range": 0.33, "height_shift_range": 0.33, "zoom_range": 0.33,
              "horizontal_flip": True, "vertical_flip": True, "fill_mode": 'nearest',
              "seed": 314, "batch_size": conf["batch_size"], "color_mode": color_mode}
    np.save(log_path + model_id + '__aug.npy', conf_a)

    chunk_info = {'size_x': size_x, 'size_y': size_y, 'size_z': size_z,
                  'stride_x': stride_x, 'stride_y': stride_y, 'stride_z': stride_z}

    if flag_save:
        check_path1 = dir_model+'loss_model.hdf5'
        checkpoint1 = ModelCheckpoint(check_path1, monitor='val_loss',
                                      verbose=1, save_best_only=True, mode='min')
        check_path2 = dir_model+'acc_model.hdf5'
        checkpoint2 = ModelCheckpoint(check_path2, monitor='val_acc',
                                      verbose=1, save_best_only=True, mode='max')
        check_path3 = dir_model+'model_{epoch:02d}-{val_loss:.2f}.hdf5'
        checkpoint3 = ModelCheckpoint(check_path3, monitor='val_loss',
                                      verbose=1, save_best_only=False, mode='min', period=500)
        callbacks_list = [checkpoint1, checkpoint2, checkpoint3, tensorboard]
    else:
        callbacks_list = [tensorboard]

    # ----------------------------------------------Create Model----------------------------------------------#

    # build up the model
    print(conf)

    if pretrained_model == 0:
        model = vgg(conf["input_shape"])
    else:
        model_path = gbl_get_value("pretrained_path")
        model = load_model(model_path, compile=False)

    # loss = mean_squared_error_1e6
    opt = Adam(lr=conf["learning_rate"], decay=conf["decay"],
               epsilon=conf["epsilon"], beta_1=conf["beta_1"], beta_2=conf["beta_2"])

    # x_train and y_train are the same one; x_val and y_val are the same one

    dg_train = customImageDataGenerator(rotation_range=conf_a["rotation_range"],
                                        shear_range=conf_a["shear_range"],
                                        width_shift_range=conf_a["width_shift_range"],
                                        height_shift_range=conf_a["height_shift_range"],
                                        zoom_range=conf_a["zoom_range"],
                                        horizontal_flip=conf_a["horizontal_flip"],
                                        vertical_flip=conf_a["vertical_flip"],
                                        fill_mode=conf_a["fill_mode"])
    dg_val = customImageDataGenerator(width_shift_range=conf_a["width_shift_range"],
                                      height_shift_range=conf_a["height_shift_range"],
                                      zoom_range=conf_a["zoom_range"],
                                      horizontal_flip=conf_a["horizontal_flip"],
                                      vertical_flip=conf_a["vertical_flip"],
                                      fill_mode=conf_a["fill_mode"])

    aug_dir = ''

    # zip files
    # if x is not None and y is None: classification; if x and y are not None: x --> y
    data_generator_t = dg_train.flow(x=train_path, y=None,
                                     data_type_x=['NAC_percent', 'CTAC'],
                                     data_type_y=None, batch_size=conf_a["batch_size"],
                                     chunk_info=chunk_info, color_mode=conf_a["color_mode"], shuffle=True,
                                     seed=conf_a["seed"], save_to_dir=None)
    data_generator_v = dg_val.flow(x=val_path, y=None,
                                     data_type_x=['NAC_percent', 'CTAC'],
                                     data_type_y=None, batch_size=conf_a["batch_size"],
                                     chunk_info=chunk_info, color_mode=conf_a["color_mode"], shuffle=True,
                                     seed=conf_a["seed"], save_to_dir=None)

    # ----------------------------------------------Train Model----------------------------------------------#

    # compile
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['acc'])
    model.summary()

    # train
    model.fit_generator(generator=data_generator_t,
                        steps_per_epoch=int(dg_train.chunk_num / conf_a["batch_size"]),  #
                        epochs=conf["epochs"],
                        callbacks=callbacks_list,
                        validation_data=data_generator_v,
                        validation_steps=int(dg_val.chunk_num / conf_a["batch_size"]))  #
    return model
