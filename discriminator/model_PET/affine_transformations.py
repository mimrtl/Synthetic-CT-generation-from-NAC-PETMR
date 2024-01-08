"""Utilities for performing affine transformations on image data.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import glob
import nibabel as nib
import numpy as np
import os
import copy
from keras import backend as K

try:
    import scipy
    # scipy.ndimage cannot be accessed until explicitly imported
    from scipy import ndimage
except ImportError:
    scipy = None

try:
    from PIL import ImageEnhance
    from PIL import Image as pil_image
except ImportError:
    pil_image = None
    ImageEnhance = None


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def array_to_img(x, data_format='channels_last', scale=True, dtype='float32'):
    """Converts a 3D Numpy array to a PIL Image instance.

    # Arguments
        x: Input Numpy array.
        data_format: Image data format.
            either "channels_first" or "channels_last".
        scale: Whether to rescale image values
            to be within `[0, 255]`.
        dtype: Dtype to use.

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if invalid `x` or `data_format` is passed.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    x = np.asarray(x, dtype=dtype)
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image). '
                         'Got array with shape: %s' % (x.shape,))

    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Invalid data_format: %s' % data_format)

    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    if data_format == 'channels_first':
        x = x.transpose(1, 2, 0)
    if scale:
        x = x + max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 4:
        # RGBA
        return pil_image.fromarray(x.astype('uint8'), 'RGBA')
    elif x.shape[2] == 3:
        # RGB
        return pil_image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return pil_image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number: %s' % (x.shape[2],))


def img_to_array(img, data_format='channels_last', dtype='float32'):
    """Converts a PIL Image instance to a Numpy array.

    # Arguments
        img: PIL Image instance.
        data_format: Image data format,
            either "channels_first" or "channels_last".
        dtype: Dtype to use for the returned array.

    # Returns
        A 3D Numpy array.

    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: %s' % data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype=dtype)
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: %s' % (x.shape,))
    return x


def apply_affine_transform(x, theta=0, tx=0, ty=0, shear=0, zx=1, zy=1,
                           row_axis=0, col_axis=1, channel_axis=2,
                           fill_mode='nearest', cval=0., order=1):
    """Applies an affine transformation specified by the parameters given.

    # Arguments
        x: 2D numpy array, single image.
        theta: Rotation angle in degrees.
        tx: Width shift.
        ty: Heigh shift.
        shear: Shear angle in degrees.
        zx: Zoom in x direction.
        zy: Zoom in y direction
        row_axis: Index of axis for rows in the input image.
        col_axis: Index of axis for columns in the input image.
        channel_axis: Index of axis for channels in the input image.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
        order: int, order of interpolation

    # Returns
        The transformed version of the input.
    """
    if scipy is None:
        raise ImportError('Image transformations require SciPy. '
                          'Install SciPy.')
    transform_matrix = None
    if theta != 0:
        theta = np.deg2rad(theta)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        transform_matrix = rotation_matrix

    if tx != 0 or ty != 0:
        shift_matrix = np.array([[1, 0, tx],
                                 [0, 1, ty],
                                 [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shift_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shift_matrix)

    if shear != 0:
        shear = np.deg2rad(shear)
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = shear_matrix
        else:
            transform_matrix = np.dot(transform_matrix, shear_matrix)

    if zx != 1 or zy != 1:
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])
        if transform_matrix is None:
            transform_matrix = zoom_matrix
        else:
            transform_matrix = np.dot(transform_matrix, zoom_matrix)

    if transform_matrix is not None:
        h, w = x.shape[row_axis], x.shape[col_axis]
        transform_matrix = transform_matrix_offset_center(
            transform_matrix, h, w)
        x = np.rollaxis(x, channel_axis, 0)
        final_affine_matrix = transform_matrix[:2, :2]
        final_offset = transform_matrix[:2, 2]

        channel_images = [ndimage.interpolation.affine_transform(
            x_channel,
            final_affine_matrix,
            final_offset,
            order=order,
            mode=fill_mode,
            cval=cval) for x_channel in x]
        x = np.stack(channel_images, axis=0)
        x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def apply_brightness_shift(x, brightness):
    """Performs a brightness shift.

    # Arguments
        x: Input tensor. Must be 3D.
        brightness: Float. The new brightness value.
        channel_axis: Index of axis for channels in the input tensor.

    # Returns
        Numpy image tensor.

    # Raises
        ValueError if `brightness_range` isn't a tuple.
    """
    if ImageEnhance is None:
        raise ImportError('Using brightness shifts requires PIL. '
                          'Install PIL or Pillow.')
    x = array_to_img(x)
    x = imgenhancer_Brightness = ImageEnhance.Brightness(x)
    x = imgenhancer_Brightness.enhance(brightness)
    x = img_to_array(x)
    return x


def apply_channel_shift(x, intensity, channel_axis=0):
    """Performs a channel shift.

    # Arguments
        x: Input tensor. Must be 3D.
        intensity: Transformation intensity.
        channel_axis: Index of axis for channels in the input tensor.

    # Returns
        Numpy image tensor.

    """
    x = np.rollaxis(x, channel_axis, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [
        np.clip(x_channel + intensity,
                min_x,
                max_x)
        for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def compute_chunks(cases, data_types, chunk_info):
    temp = []
    size_x = chunk_info['size_x']
    size_y = chunk_info['size_y']
    size_z = chunk_info['size_z']
    stride_x = chunk_info['stride_x']
    stride_y = chunk_info['stride_y']
    stride_z = chunk_info['stride_z']
    whole_data = {}
    # [path of the cases]
    count = 0
    for case in cases:
        whole_data[case] = {}
        # /data/test_nii/train/Case1
        for data_type in data_types:
            whole_data[case][data_type] = []
            path = glob.glob(os.path.join(case, '*'+data_type+'*.nii'))
            if len(path) == 0:
                raise ValueError("no "+data_type+' data was found! try to add one!')
            elif len(path) > 1:
                raise ValueError("multi " + data_type + ' data were found. Please delete extra data!')
            else:
                pass
            file = nib.load(path[0])
            data = file.get_fdata()
            whole_data[case][data_type].append(data)

            upper_x = data.shape[0]
            upper_y = data.shape[1]
            upper_z = data.shape[2]

            count = 0
            z = 0
            flag_z = 0
            while (z <= (upper_z - size_z)) or z < upper_z:
                # print(z)
                # [:, :, z:z+size]
                y = 0
                flag_y = 0
                while y <= upper_y - size_y or y < upper_y:
                    # print(y)
                    # [:, y:y+size, z:z+size]
                    x = 0
                    while x <= upper_x - size_x:
                        # print(x)
                        count += 1
                        # new_data = data[x:x + size, y:y + size, z:z + size]
                        # [[case_name, type_name], x, y, z]
                        temp.append([[case, data_type], x, y, z])
                        x = x + stride_x
                        # print('x:{}'.format(x))
                    if x < upper_x:
                        # print('duoyu:{}'.format(x))
                        count += 1
                        # new_data = data[upper_x - size:upper_x, y:y + size, z:z + size]
                        temp.append([[case, data_type], upper_x-size_x, y, z])
                        x = upper_x
                        # print('the upper_x:{}'.format(x))
                    if flag_y:
                        y = upper_y
                    else:
                        if y + stride_y <= upper_y - size_y:
                            y += stride_y
                        else:
                            flag_y = 1
                            y = upper_y - size_y
                            # print('y:{}'.format(y))
                if flag_z:
                    z = upper_z
                else:
                    if z + stride_z <= upper_z - size_z:
                        z += stride_z
                    else:
                        flag_z = 1
                        z = upper_z - size_z
                        # print('z:{}'.format(z))
        # print('Count:{}'.format(count))
    print(len(temp))
    return temp, whole_data


def load_chunk_data(cur_x, data_x, chunk_info, color_mode, data_format):
    case_name = cur_x[0][0]
    type_name = cur_x[0][1]
    data = data_x[case_name][type_name][0]

    x = cur_x[1]
    y = cur_x[2]
    z = cur_x[3]

    size_x = chunk_info['size_x']
    size_y = chunk_info['size_y']
    size_z = chunk_info['size_z']

    # for medical images, they are grayscale. the color_mode is 1
    # the images are rgb images, this function and  needs to be changed

    # consider the data situation 512x512x30 vs 512x512x30x3
    if data_format == 'channels_last':
        chunk_data = np.zeros((size_x, size_y, size_z, color_mode), dtype=K.floatx())
        if color_mode == 1:
            chunk_data[:, :, :, 0] = data[x:x+size_x, y:y+size_y, z:z+size_z]
        else:
            chunk_data = data[x:x+size_x, y:y+size_y, z:z+size_z, :]
    else:
        chunk_data = np.zeros((color_mode, size_x, size_y, size_z), dtype=K.floatx())
        if color_mode == 1:
            chunk_data[0, :, :, :] = data[x:x+size_x, y:y+size_y, z:z+size_z]
        else:
            chunk_data = data[:, x:x+size_x, y:y+size_y, z:z+size_z]

    return chunk_data


def compute_chunks_multi(cases, data_types, chunk_info):
    temp = []
    size_x = chunk_info['size_x']
    size_y = chunk_info['size_y']
    size_z = chunk_info['size_z']
    stride_x = chunk_info['stride_x']
    stride_y = chunk_info['stride_y']
    stride_z = chunk_info['stride_z']
    whole_data = {}
    # [path of the cases]
    count = 0
    for case in cases:
        whole_data[case] = {}
        # /data/test_nii/train/Case1
        paths = []
        for data_type in data_types:
            single_path = glob.glob(os.path.join(case, '*' + data_type + '*.nii'))
            if len(single_path) == 0:
                raise ValueError("no " + data_type + ' data was found! try to add one!')
            elif len(single_path) > 1:
                raise ValueError("multi " + data_type + ' data were found. Please delete extra data!')
            else:
                paths.append(single_path[0])
                whole_data[case][data_type] = []

                sgl_file = nib.load(single_path[0])
                sgl_data = sgl_file.get_fdata()
                whole_data[case][data_type].append(sgl_data)

        path = paths[0]
        file = nib.load(path)
        data = file.get_fdata()

        upper_x = data.shape[0]
        upper_y = data.shape[1]
        upper_z = data.shape[2]

        count = 0
        z = 0
        flag_z = 0
        while (z <= (upper_z - size_z)) or z < upper_z:
            # print(z)
            # [:, :, z:z+size]
            y = 0
            flag_y = 0
            while y <= upper_y - size_y or y < upper_y:
                # print(y)
                # [:, y:y+size, z:z+size]
                x = 0
                while x <= upper_x - size_x:
                    count += 1
                    # new_data = data[x:x + size, y:y + size, z:z + size]
                    for i in data_types:
                        temp.append([case, x, y, z])
                    x = x + stride_x
                if x < upper_x:
                    count += 1
                    # new_data = data[upper_x - size:upper_x, y:y + size, z:z + size]
                    for i in data_types:
                        temp.append([case, upper_x-size_x, y, z])
                    x = upper_x
                if flag_y:
                    y = upper_y
                else:
                    if y + stride_y <= upper_y - size_y:
                        y += stride_y
                    else:
                        flag_y = 1
                        y = upper_y - size_y
                        # print('y:{}'.format(y))
            if flag_z:
                z = upper_z
            else:
                if z + stride_z <= upper_z - size_z:
                    z += stride_z
                else:
                    flag_z = 1
                    z = upper_z - size_z
                    # print('z:{}'.format(z))
        print('Count:{}'.format(count))
    print(len(temp))
    return temp, whole_data


def load_chunk_data_multi(cur_x, data_x, chunk_info, data_type, color_mode, data_format):
    size_x = chunk_info['size_x']
    size_y = chunk_info['size_y']
    size_z = chunk_info['size_z']

    case_name = cur_x[0]

    if len(data_x[case_name].keys()) < len(data_type):
        raise ValueError('data was missing! the current one is {} while the expected one is {}!'.format(cur_x, data_type))
    elif len(data_x[case_name].keys()) > len(data_type):
        raise ValueError('extra data was found. Please delete extra data!')
    else:
        if data_format == 'channels_last':
            chunk_data_total = np.zeros((size_x, size_y, size_z, len(data_type)*color_mode), dtype=K.floatx())
        else:
            chunk_data_total = np.zeros((len(data_type)*color_mode, size_x, size_y, size_z), dtype=K.floatx())
        x = cur_x[1]
        y = cur_x[2]
        z = cur_x[3]
        for i in range(len(data_type)):
            data = data_x[case_name][data_type[i]][0]
            # for medical images, they are grayscale. the color_mode is 1
            down = i*color_mode
            up = (i+1)*color_mode
            if data_format == 'channels_last':
                if color_mode == 1:
                    chunk_data_total[:, :, :, i] = data[x:x + size_x, y:y + size_y, z:z + size_z]
                else:
                    chunk_data_total[:, :, :, down:up] = data[x:x+size_x, y:y+size_y, z:z+size_z, :]
            else:
                if color_mode == 1:
                    chunk_data_total[i, :, :, :] = data[x:x+size_x, y:y+size_y, z:z+size_z]
                else:
                    chunk_data_total[down:up, :, :, :] = data[:, x:x+size_x, y:y+size_y, z:z+size_z]
    return chunk_data_total


