import nibabel as nib
import numpy as np
import math
import os


# Description: compute mae, std, mse, psnr, and NCC
def single_cmp(sCT_path, CT_path):

    sCT_file = nib.load(sCT_path)
    CT_file = nib.load(CT_path)

    sCT_data = sCT_file.get_fdata()
    CT_data = CT_file.get_fdata()

    CT_data[CT_data < -1000] = -1000
    CT_data[CT_data > 2000] = 2000

    mae_loss = np.mean(np.abs(sCT_data - CT_data))
    # print('mae loss:', mae_loss)
    print(mae_loss)

    std_loss = np.std(np.abs(sCT_data - CT_data))
    # print('std:', std_loss)
    print(std_loss)

    mse_loss = np.mean(np.square(sCT_data - CT_data))
    # print('mse loss:', mse_loss)
    print(mse_loss)

    psnr = (10.0 * math.log((np.amax(CT_data) ** 2) / mse_loss)) / math.log(10)
    # print('psnr:', psnr)
    print(psnr)

    mean_CT = np.average(CT_data)
    mean_sCT = np.average(sCT_data)

    new_CT = CT_data - mean_CT
    new_sCT = sCT_data - mean_sCT

    numerator = np.sum(new_CT * new_sCT)
    denominator = np.sqrt(np.sum(new_CT * new_CT) * np.sum((new_sCT * new_sCT)))
    NCC = numerator / denominator

    # print('NCC:', NCC)
    print(NCC)


if __name__ == '__main__':
    sCT_path = './test/subj023/subj023_sCT.nii'
    CT_path = './test/subj023/subj023_CTAC.nii'
    single_cmp(sCT_path, CT_path)
