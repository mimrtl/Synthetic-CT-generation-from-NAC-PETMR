#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from global_dict.w_global import gbl_get_value
from model.unet_3d import unet
from keras.models import load_model
from model.nii_generator_aug import *
from model.loss import *
from keras_radam import RAdam
SEED = 314

import tensorflow as tf


def train_a_unet(pretrained_model=None):
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
    n_fliter = gbl_get_value("n_filter")
    depth = gbl_get_value("depth")
    batch_size = gbl_get_value("batch_size")
    optimizer = 'RAdam'
    flag_save = True

    size_x = gbl_get_value('size_x')
    size_y = gbl_get_value('size_y')
    size_z = gbl_get_value('size_z')

    stride_x = gbl_get_value('stride_x')
    stride_y = gbl_get_value('stride_y')
    stride_z = gbl_get_value('stride_z')

    # ----------------------------------------------Configurations----------------------------------------------#

    # logs
    log_path = '../logs/gp_' + gp_id + '/' + str(size_x) + '-' + str(size_y) + '-' + str(size_z) + '/unet/' + str(method) + '/' + str(rounds) + '/' + model_id + '/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    tensorboard = TensorBoard(log_dir=log_path, batch_size=batch_size,
                              write_graph=True, write_grads=True,
                              write_images=True)

    # set traininig configurations
    conf = {"image_shape": (size_x, size_y, size_z, slice_x), "out_channel": 1, "filter": n_fliter, "depth": depth,
            "inc_rate": 2, "activation": 'relu', "dropout": 0.5, "batchnorm": True, "maxpool": True,
            "upconv": True, "residual": True, "shuffle": True, "augmentation": True,
            "learning_rate": 1e-5, "decay": 0.0, "epsilon": 1e-8, "beta_1": 0.9, "beta_2": 0.999,
            "validation_split": 0.25, "seed": 42, "batch_size": batch_size, "epochs": epochs,
            "loss": "mse1e6", "metric": "mse", "optimizer": optimizer, "model_id": model_id}
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
        check_path = dir_model+'model_{epoch:02d}-{val_loss:.2f}' + '.hdf5'  
        checkpoint1 = ModelCheckpoint(check_path, monitor='val_loss',
                                      verbose=1, save_best_only=False, 
                                      save_weights_only=False, mode='min', period=400)

        check_path4 = dir_model+'model_best.hdf5'
        checkpoint4 = ModelCheckpoint(check_path4, monitor='val_loss',
                                      verbose=1, save_best_only=True,
                                      save_weights_only=False, mode='min')

        check_path2 = dir_model+'model_{epoch:02d}-{val_loss:.2f}' + '.hdf5'
        checkpoint2 = ModelCheckpoint(check_path2, monitor='val_loss',
                                      verbose=1, save_best_only=False,
                                      save_weights_only=False, mode='min', period=450)
        check_path3 = dir_model+'model_{epoch:02d}-{val_loss:.2f}' + '.hdf5'
        checkpoint3 = ModelCheckpoint(check_path3, monitor='val_loss',
                                      verbose=1, save_best_only=False,
                                      save_weights_only=False, mode='min', period=500)

        check_path5 = dir_model+'model_latest' + '.hdf5' 
        checkpoint5 = ModelCheckpoint(check_path5, monitor='val_loss',
                                      verbose=1, save_best_only=False,
                                      save_weights_only=False, mode='min', period=1)
        '''
        check_path4 = dir_model+'model_' + model_id + '_val_maeWithFocus_best.hdf5'
        checkpoint4 = ModelCheckpoint(check_path4, monitor='val_maeWithFocus',
                                      verbose=1, save_best_only=True,
                                      save_weights_only=False, mode='min')
        '''
        callbacks_list = [checkpoint1, checkpoint2, checkpoint3, checkpoint5, tensorboard]
    else:
        callbacks_list = [tensorboard]

    # ----------------------------------------------Create Model----------------------------------------------#

    # build up the model
    print(conf)

    if pretrained_model == 0:
        model = unet(img_shape=conf["image_shape"], out_ch=conf["out_channel"],
                     start_ch=conf["filter"], depth=conf["depth"],
                     inc_rate=conf["inc_rate"], activation=conf["activation"],
                     dropout=conf["dropout"], batchnorm=conf["batchnorm"],
                     maxpool=conf["maxpool"], upconv=conf["upconv"],
                     residual=conf["residual"])
    else:
        model_path = gbl_get_value("pretrained_path")
        model = load_model(model_path, compile=False)

    opt = Adam(lr=conf["learning_rate"], decay=conf["decay"],
               epsilon=conf["epsilon"], beta_1=conf["beta_1"], beta_2=conf["beta_2"])

    # opt = RAdam(lr=conf["learning_rate"], total_steps=10000, warmup_proportion=0.1, min_lr=1e-6)
    # ----------------------------------------------Data Generator----------------------------------------------#
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
    data_generator_t = dg_train.flow(x=train_path, y=train_path,
                                     # data_type_x=['WATER', 'FAT', 'InPhase', 'OutPhase', 'NAC'],
                                     data_type_x=['NAC_percent'],
                                     data_type_y=['CTAC'], batch_size=conf_a["batch_size"],
                                     chunk_info=chunk_info, color_mode=conf_a["color_mode"], shuffle=True,
                                     seed=conf_a["seed"], save_to_dir=None)
    data_generator_v = dg_val.flow(x=val_path, y=val_path,
                                     # data_type_x=['WATER', 'FAT', 'InPhase', 'OutPhase', 'NAC'],
                                     data_type_x=['NAC_percent'],
                                     data_type_y=['CTAC'], batch_size=conf_a["batch_size"],
                                     chunk_info=chunk_info, color_mode=conf_a["color_mode"], shuffle=True,
                                     seed=conf_a["seed"], save_to_dir=None)

    # ----------------------------------------------Train Model----------------------------------------------#

    # loss = mae
    loss = perceptual_loss
    # compile mse, mae, content, style
    model.compile(loss=perceptual_loss, optimizer=opt, metrics=[mse, mae, content, style, perceptual_loss])
    # model.compile(loss=loss, optimizer=opt, metrics=[mse, mae])
    model.summary()

    # train
    model.fit_generator(generator=data_generator_t,
                        steps_per_epoch=int(dg_train.chunk_num / conf_a["batch_size"]),  #
                        epochs=conf["epochs"],
                        callbacks=callbacks_list,
                        validation_data=data_generator_v,
                        validation_steps=int(dg_val.chunk_num / conf_a["batch_size"]))  #
    return model

