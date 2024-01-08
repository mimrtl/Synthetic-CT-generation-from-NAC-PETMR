import numpy as np
import nibabel as nib
import os


def clean_ct_data():
    # num 19 20 21 22 23
    # img type CTAC FAT WATER InPhase OutPhase NAC
    num = "18"
    img_type = 'CTAC'
    input_folder = '/FDG-data/ori_data_3/subj0' + num + '/'
    img_path = input_folder + 'subj0' + num + '_' + img_type + '_mask_clip.nii'
    img_file = nib.load(img_path)
    img_data = img_file.get_fdata()

    # remove noise by threshold
    img_data[img_data > -300] = 1
    img_data[img_data <= -300] = 0

    # img_data[437:, :, 60:] = 0
    # img_data[:, :130, :] = 0
    # img_data[:60, :, :] = 0

    affine = img_file.affine
    header = img_file.header
    nii_file = nib.Nifti1Image(img_data, affine, header)
    if not os.path.exists(input_folder + 'target/'):
        os.mkdir(input_folder + 'target/')
    nib.save(nii_file, input_folder + 'target/' + img_type + '_mask.nii')

    ct_path = input_folder + 'subj0' + num + '_' + img_type + '_clip.nii'
    ct_file = nib.load(ct_path)
    ct_data = ct_file.get_fdata()
    ct_data[ct_data < -1000] = -1000
    ct_data[ct_data > 2000] = 2000
    ct_data += 1000

    new_data = np.zeros((ct_data.shape[0], ct_data.shape[1], ct_data.shape[2]))
    new_data = np.multiply(img_data, ct_data)
    # new_data -= 1000
    affine = ct_file.affine
    header = ct_file.header
    nii_file = nib.Nifti1Image(new_data, affine, header)
    nib.save(nii_file, input_folder + 'target/subj0' + num + '_' + img_type + '_target.nii')


def clean_nacpet_data():
    input_folder = '/data/RegNIFTIs/subj022/'
    img_path = input_folder + 'subj022_CTAC_mask_target.nii'
    img_file = nib.load(img_path)
    img_data = img_file.get_fdata()

    # these are the noises which should be removed
    # img_data[img_data <= 30] = 0
    # img_data[img_data > 30] = 1
    #
    # img_data[448:, :233, 102:111] = 0
    # img_data[448:, :, 43:102] = 0
    # img_data[443:, :219, 100:105] = 0
    # img_data[443:, :, 53:87] = 0
    # img_data[443:, :232, 86:101] = 0
    # img_data[438:, :206, 45:102] = 0
    # img_data[444:, :213, 34:50] = 0
    # img_data[431:, :199, 53:70] = 0
    # img_data[433:, :183, 94:98] = 0

    # save the clean data
    # affine = img_file.affine
    # header = img_file.header
    # nii_file = nib.Nifti1Image(img_data, affine, header)
    # nib.save(nii_file, input_folder + 'mr_mask_1.nii')

    pet_path = input_folder + 'subj022_NAC.nii'
    pet_file = nib.load(pet_path)
    pet_data = pet_file.get_fdata()

    # new_data = np.zeros((mr_data.shape[0], mr_data.shape[1], mr_data.shape[2]))
    pet_data = np.multiply(img_data, pet_data)

    affine = pet_file.affine
    header = pet_file.header
    nii_file = nib.Nifti1Image(pet_data, affine, header)
    nib.save(nii_file, input_folder + 'NAC_target_1.nii')



def remove_noise():
    data_path = '/FDG-data/processed/20/subj020_CTAC_target.nii'
    data_file = nib.load(data_path)
    data = data_file.get_fdata()
    data[:, 0:20, :] = 0
    affine = data_file.affine
    header = data_file.header
    nii_file = nib.Nifti1Image(data, affine, header)
    nib.save(nii_file, '/FDG-data/processed/20/subj020_CTAC_target_1.nii')


def load_nii(path, data_type):
    # load nifti data
    dict = {}
    for i in data_type:
        data_path = path + '_' + i + '.nii'
        file = nib.load(data_path)
        data = file.get_fdata()
        dict[i] = []
        dict[i].append(data)
        dict[i].append(file)
    return dict


def save_nii(save_dict, data_dict, path):
    # save nifti data
    for key in save_dict.keys():
        affine = data_dict[key][1].affine
        header = data_dict[key][1].header
        nii_file = nib.Nifti1Image(save_dict[key], affine, header)
        nib.save(nii_file, path + '_' + key + '_target' + '.nii')


def create_mask(path):
    # combine the CT mask and NAC PET mask, and apply it to CT and PET images
    data_type = ['CTAC_mask', 'NAC_mask', 'CTAC', 'NAC']
    data_dict = load_nii(path, data_type)

    ct_mask_data = data_dict['CTAC_mask'][0]
    ct_mask_data[ct_mask_data >= -450] = 1
    ct_mask_data[ct_mask_data < -450] = 0

    nac_mask_data = data_dict['NAC_mask'][0]
    nac_mask_data[nac_mask_data < 90] = 0
    nac_mask_data[nac_mask_data >= 90] = 1
    # nac_mask_data[:, 0:30, :] = 0
    # nac_mask_data[59:75, :42, :] = 0

    final_mask = ct_mask_data * nac_mask_data

    ct_data = data_dict['CTAC'][0]
    ct_data[ct_data < -1024] = -1024
    ct_data[ct_data > 2000] = 2000
    # ct data range(0, 3024)
    ct_data += 1024
    new_ct_data = final_mask * ct_data
    # new_ct_data = ct_mask_data * ct_data
    new_ct_data -= 1024

    nac_data = data_dict['NAC'][0]
    new_nac_data = nac_mask_data * nac_data

    save_dict = {}
    save_dict['CTAC'] = new_ct_data
    save_dict['NAC'] = new_nac_data
    save_dict['CTAC_mask'] = final_mask
    save_dict['NAC_mask'] = nac_mask_data

    save_nii(save_dict, data_dict, path)


def crop_data(path):
    # To remove truncated back and train model more efficiently, the data was cropped.
    dicts = {'subj001': [19, 76], 'subj007': [20, 69], 'subj014': [20, 69], 'subj020': [25, 69],
             'subj002': [17, 74], 'subj008': [31, 85], 'subj015': [40, 81], 'subj021': [21, 76],
             'subj003': [18, 71], 'subj009': [16, 71], 'subj016': [28, 69], 'subj022': [27, 87],
             'subj004': [19, 68], 'subj010': [29, 67], 'subj017': [19, 71], 'subj023': [22, 75],
             'subj005': [25, 74], 'subj011': [16, 70], 'subj018': [22, 74],
             'subj006': [19, 70], 'subj012': [19, 68], 'subj019': [19, 75], }
    folders = os.listdir(path)
    for folder in folders:
        print(folder)
        folder_path = path + '/' + folder + '/'
        types = ['CTAC', 'NAC']
        down = dicts[folder][0]
        up = dicts[folder][1]
        for data_type in types:
            file_path = folder_path + folder + '_' + data_type + '_target.nii'
            file = nib.load(file_path)
            data = file.get_fdata()
            data = data[:, :, down:up]
            if folder == 'subj004':
                if data_type == 'CTAC':
                    data[:, :47, :] = -1000
                else:
                    data[:, :47, :] = 0
            elif folder == 'subj006' or folder == 'subj010':
                if data_type == 'CTAC':
                    data[:, :40, :] = -1000
                else:
                    data[:, :40, :] = 0
            else:
                pass
            affine = file.affine
            header = file.header
            nii_file = nib.Nifti1Image(data, affine, header)
            save_folder = '/data/crop_data/' + folder
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            nib.save(nii_file, save_folder + '/' + folder + '_' + data_type + '.nii')


if __name__ == '__main__':
    clean_ct_data()
    clean_nacpet_data()
    # nums = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    nums = [14, 15, 16, 18, 19, 20, 21, 22, 23]
    for num in nums:
        whole_mask(str(num))
    remove_noise()
    nums = [23]
    for num in nums:
        path = '/data/RegNIFTIs/subj0' + str(num) + '/subj0' + str(num)
        create_mask(path)
    crop_data(path)


