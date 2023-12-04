#!/usr/bin/python
# -*- coding: UTF-8 -*-

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys

sys.path.append(os.path.dirname(sys.path[0]))

import glob
import nibabel as nib
import numpy as np
import copy
from keras import backend as K
from keras.models import load_model
from process.normalization import MaxMinNorm, NacNorm


# load nifti file
def load_nii(path, data_type, file=False):
    print(path)
    data_path = glob.glob(path + '*' + data_type + '*.nii')[-1]
    print(data_path)
    data_file = nib.load(data_path)
    data = data_file.get_fdata()
    if file:
        return data_file
    else:
        return data


def save_nii(path, data_type, syn_data, dif_data, model_id, label):
    '''  save data as nifti file
    :param path: the path of subject
    :param data_type: CTAC
    :param syn_data: syn CT
    :param dif_data: dif CT
    :param model_id: mark of n_pixel and n_stride
    :return:
    '''
    data_file = load_nii(path, data_type, True)
    affine = data_file.affine
    header = data_file.header
    nii_file = nib.Nifti1Image(syn_data, affine, header)
    dif_file = nib.Nifti1Image(dif_data, affine, header)
    nib.save(nii_file, path + model_id + '_sCT_' + label + '.nii')
    # nib.save(dif_file, path + model_id + '_dif2.nii')


# percent normalization method
def percentNormalization(data):
    maximum = np.max(data)
    edge = np.percentile(data[data > 0], 95)
    data[data < edge] = (data[data < edge] * 0.95) / edge
    data[data >= edge] = 0.95 + 0.05 * (data[data >= edge] - edge) / (maximum - edge)
    return data


# prediction
def predict_sCT(model, test_path, tag=''):
    # set up patch size
    size_x = 64
    size_y = 64
    size_z = 32

    # set up moving stride for each dimension
    stride_x = 8
    stride_y = 8
    stride_z = 1

    dicts = {'size_x': size_x, 'size_y': size_y, 'size_z': size_z,
             'stride_x': stride_x, 'stride_y': stride_y, 'stride_z': stride_z}
    slice_x = 1
    depth = 4
    cases = 18
    train_round = 'whole'
    batch_size = 8
    start_filter = 32

    # load NAC PET and CT images
    data_X_nac = load_nii(test_path, 'NAC')
    data_Y = load_nii(test_path, 'CTAC')

    # Normalization
    data_input_nac = NacNorm(data_X_nac)
    # data_input_nac = percentNormalization(data_X_nac)
    target_z = 47

    print("data shape", data_input_nac.shape)
    # create template
    new_X_nac = np.zeros((192, 192, target_z))
    new_Y = np.zeros((192, 192, target_z))

    upper_x = new_Y.shape[0]
    upper_y = new_Y.shape[1]
    upper_z = new_Y.shape[2]

    difference = upper_z - data_Y.shape[2]
    low = difference // 2
    up = upper_z - (difference - low)
    print("low: {}; up: {}".format(low, up))

    new_X_nac[:, :, low:up] = data_input_nac
    new_Y[:, :, low:up] = data_Y

    model_id = test_path.split('/')[-2]
    X = np.zeros((1, size_x, size_y, size_z, slice_x))
    y_hat = np.zeros((upper_x, upper_y, upper_z, 1200))
    y_hat[:, :, :, :] = -2000
    max_z = 0

    count_z = 0

    # sliding window
    for temp_z in range(0, upper_z-size_z+1, stride_z):
        print('temp_z:{}'.format(temp_z))
        count_y = 0
        for temp_y in range(0, upper_y-size_y+1, stride_y):
            for temp_x in range(0, upper_x-size_x+1, stride_x):
                X[0, :, :, :, 0] = new_X_nac[temp_x:temp_x + size_x, temp_y:temp_y + size_y, temp_z:temp_z + size_z]
                pred_y = np.squeeze(model.predict(X))
                idx = count_z * 64 + count_y * 8
                num = temp_x // stride_x

                if temp_x < size_x:
                    cur_num = 0
                    for i in range(int(size_x/stride_x-1), -1, -1):
                        y_hat[temp_x + i*stride_x:temp_x + (i+1)*stride_x, temp_y:temp_y + size_y, temp_z:temp_z + size_z, idx+cur_num] \
                            = pred_y[i * stride_x: (i + 1) * stride_x, :, :]
                        max_z = max(max_z, idx+cur_num)
                        if cur_num < num:
                            cur_num += 1
                else:
                    for i in range(0, int(size_x/stride_x)):
                        y_hat[temp_x + i*stride_x:temp_x + (i+1)*stride_x, temp_y:temp_y+size_y, temp_z: temp_z+size_z, idx+num] \
                            = pred_y[i*stride_x: (i+1)*stride_x, :, :]
                        max_z = max(max_z, idx+num)
                        if num > 0:
                            num -= 1
            count_y += 1
        count_z += 1
    print('maximum of z is', max_z)
    y_hat_norm = np.zeros((upper_x, upper_y, upper_z))

    # calculate the medium at the pixel level
    for temp_z in range(upper_z):
        print(temp_z)
        for temp_y in range(upper_y):
            for temp_x in range(upper_x):
                temp = y_hat[temp_x, temp_y, temp_z, :]
                temp = list(temp)
                temp.sort()
                temp = temp[::-1]
                while temp and temp[-1] < -1999:
                    temp.pop()
                if temp:
                    medium = np.median(np.array(temp))
                else:
                    medium = 0
                    print('the location is (' + str(temp_x) + ', ' + str(temp_y) + ', ' + str(temp_z) + ')')
                y_hat_norm[temp_x, temp_y, temp_z] = medium

    del y_hat

    Y_max = 2000
    Y_min = -1000

    # restore norm
    y_hat_norm *= (Y_max-Y_min)
    y_hat_norm += Y_min

    dif = y_hat_norm - new_Y

    ori_y_hat = y_hat_norm[:, :, low:up]
    ori_dif = dif[:, :, low:up]

    save_nii(test_path, 'CTAC', ori_y_hat, ori_dif, model_id, tag)


if __name__ == '__main__':
    # model_path = '/breast_project/training/transfer/result/gp_1/64-64-32/unet/upload_model/mse_model_xue_local_computer.hdf5'
    model_path = '/breast_project/training/result/gp_1/64-64-32/unet/per/1st_round/model_best.hdf5'
    sub_path = '/breast_project/training/transfer/result/gp_1/64-64-32/unet/upload_model/subj004/'
    model = load_model(model_path, compile=False)
    predict_sCT(model, sub_path, tag='model_test_per_nacnorm')
