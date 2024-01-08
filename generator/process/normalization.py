#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import os
import nibabel as nib


def MaxMinNorm(data, type='MR'):
    if type == 'CT':
        data_max = 2000.0
        data_min = -1000.0
        data[data > data_max] = data_max
        data[data < data_min] = data_min
    else:
        data_max = np.amax(data)
        data_min = np.amin(data)
    print("Before:", data_max, data_min)
    data -= data_min
    data /= (data_max-data_min)
    print("After:", np.amax(data), np.amin(data))
    return data


def NacNorm(data):
    return data/1500


# percent normalization method
def percentNormalization(data):
    maximum = np.max(data)
    edge = np.percentile(data[data > 0], 95)
    data[data < edge] = (data[data < edge] * 0.95) / edge
    data[data >= edge] = 0.95 + 0.05 * (data[data >= edge] - edge) / (maximum - edge)
    return data


# norm sCT
if __name__ == '__main__':
    path = '/data/whole/val/nii/'
    folders = os.listdir(path)
    for folder in folders:
        print(folder)
        sCT_path = ''
        sCT_path = path + folder + '/' + folder + '_sCT_1.nii'
        file = nib.load(sCT_path)
        data = file.get_fdata()
        norm_data = MaxMinNorm(data, type='CT')
        affine = file.affine
        header = file.header
        nii_file = nib.Nifti1Image(norm_data, affine, header)
        save_path = ''
        save_path = sCT_path.replace('/nii/', '/norm_nii/')
        nib.save(nii_file, save_path)
