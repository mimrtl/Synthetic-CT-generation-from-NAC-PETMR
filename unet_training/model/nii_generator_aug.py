"""
From KERAS package
https://github.com/keras-team/keras/blob/cebf2084ebb0a603383ceb6807653921796cd095/keras/preprocessing/image.py#L342

Based on Emadeldeen-24's work on time steps application
https://gist.github.com/Emadeldeen-24/736c33ac2af0c00cc48810ad62e1f54a

Fairly basic set of tools for real-time data augmentation on image data.
Can easily be extended to include new transformations,
new preprocessing methods, etc...
"""

import numpy as np
import os
from six.moves import range
import threading
import warnings
from keras import backend as K
import h5py
from model.affine_transformations import *


class customImageDataGenerator(object):
    """Generate minibatches of image data with real-time data augmentation.
    # Arguments
        featurewise_center: set input mean to 0 over the dataset.
        samplewise_center: set each sample mean to 0.
        featurewise_std_normalization: divide inputs by std of the dataset.
        samplewise_std_normalization: divide each input by its std.
        zca_whitening: apply ZCA whitening.
        zca_epsilon: epsilon for ZCA whitening. Default is 1e-6.
        rotation_range: degrees (0 to 180).
        width_shift_range: fraction of total width.
        height_shift_range: fraction of total height.
        shear_range: shear intensity (shear angle in radians).
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range.
        channel_shift_range: shift range for each channels.
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'.
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally.
        vertical_flip: whether to randomly flip images vertically.
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided. This is
            applied after the `preprocessing_function` (if any provided)
            but before any other transformation.
        preprocessing_function: function that will be implied on each input.
            The function will run before any other modification on it.
            The function should take one argument:
            one image (Numpy tensor with rank 3),
            and should output a Numpy tensor with the same shape.
        data_format: 'channels_first' or 'channels_last'. In 'channels_first' mode, the channels dimension
            (the depth) is at index 1, in 'channels_last' mode it is at index 3.
            It defaults to the `image_data_format` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "channels_last".
    """

    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 brightness_range=None,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format=None,
                 interpolation_order=1,
                 dtype='float32',
                 chunk_num=0):
        if data_format is None:
            data_format = K.image_data_format()
        self.featurewise_center = featurewise_center
        self.samplewise_center = samplewise_center
        self.featurewise_std_normalization = featurewise_std_normalization
        self.samplewise_std_normalization = samplewise_std_normalization
        self.zca_whitening = zca_whitening
        self.zca_epsilon = zca_epsilon
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.brightness_range = brightness_range
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.fill_mode = fill_mode
        self.cval = cval
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rescale = rescale
        self.preprocessing_function = preprocessing_function
        self.dtype = dtype
        self.interpolation_order = interpolation_order
        self.chunk_num = chunk_num

        if data_format not in {'channels_last', 'channels_first'}:
            raise ValueError('`data_format` should be `"channels_last"` (channel after row and '
                             'column) or `"channels_first"` (channel before row and column). '
                             'Received arg: ', data_format)
        self.data_format = data_format
        if data_format == 'channels_first':
            self.channel_axis = 1
            self.row_axis = 2
            self.col_axis = 3
            self.dep_axis = 4
        if data_format == 'channels_last':
            self.channel_axis = 4
            self.row_axis = 1
            self.col_axis = 2
            self.dep_axis = 3

        self.mean = None
        self.std = None
        self.principal_components = None

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise ValueError('`zoom_range` should be a float or '
                             'a tuple or list of two floats. '
                             'Received arg: ', zoom_range)
        if zca_whitening:
            if not featurewise_center:
                self.featurewise_center = True
                warnings.warn('This ImageDataGenerator specifies '
                              '`zca_whitening`, which overrides '
                              'setting of `featurewise_center`.')
            if featurewise_std_normalization:
                self.featurewise_std_normalization = False
                warnings.warn('This ImageDataGenerator specifies '
                              '`zca_whitening` '
                              'which overrides setting of'
                              '`featurewise_std_normalization`.')
        if featurewise_std_normalization:
            if not featurewise_center:
                self.featurewise_center = True
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_std_normalization`, '
                              'which overrides setting of '
                              '`featurewise_center`.')
        if samplewise_std_normalization:
            if not samplewise_center:
                self.samplewise_center = True
                warnings.warn('This ImageDataGenerator specifies '
                              '`samplewise_std_normalization`, '
                              'which overrides setting of '
                              '`samplewise_center`.')
        if brightness_range is not None:
            if (not isinstance(brightness_range, (tuple, list)) or
                    len(brightness_range) != 2):
                raise ValueError(
                    '`brightness_range should be tuple or list of two floats. '
                    'Received: %s' % (brightness_range,))
        self.brightness_range = brightness_range

    def flow(self,
             x,
             y=None,
             data_type_x=None,
             data_type_y=None,
             batch_size=4,
             chunk_info=None,
             color_mode=1,
             shuffle=True,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png'):
        """Takes data & label path or arrays, generates batches of augmented data.

        # Arguments
            x: Input data.
                 array of rank 1 containing all cases' paths or
                 string of folder path containing all cases.
            y: Labels.
            data_type_x: the data type of input data (eg: MR).
            data_type_y: the data type of the target data (eg: CT).
            batch_size: Int (default: 4).
            chunk_size: Int (default: 32).
            stride: Int (default: 16)
                the length of pixels during each move of creating chunk data
            color_mode: the channels of input data.
                it should be 1 for grayscale images and 3 for rgb images
            shuffle: Boolean (default: True).
            seed: Int (default: None).
            save_to_dir: None or str (default: None).
                This allows you to optionally specify a directory
                to which to save the augmented pictures being generated
                (useful for visualizing what you are doing).
            save_prefix: Str (default: `''`).
                Prefix to use for filenames of saved pictures
                (only relevant if `save_to_dir` is set).
            save_format: one of "png", "jpeg"
                (only relevant if `save_to_dir` is set). Default: "png".

        # Returns
            An `Iterator` yielding tuples of `(x, y)`
                where `x` is a numpy array of image data
                (in the case of a single image input) or a list
                of numpy arrays (in the case with
                additional inputs) and `y` is a numpy array
                of corresponding labels.
                If `y` is None, only the numpy array `x` is returned.
        """
        return NumpyArrayIterator(
            x, y, self,
            data_type_x=data_type_x,
            data_type_y=data_type_y,
            batch_size=batch_size,
            chunk_info=chunk_info,
            color_mode=color_mode,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            data_x=None,
            data_y=None)

    def standardize(self, x):
        """Apply the normalization configuration to a batch of inputs.
        # Arguments
            x: batch of inputs to be normalized.
        # Returns
            The inputs, normalized.
        """
        if self.preprocessing_function:
            x = self.preprocessing_function(x)
        if self.rescale:
            x *= self.rescale

        if self.channel_axis == 1:
            img_channel_axis = self.channel_axis - 1  # 0
        else:
            img_channel_axis = self.channel_axis - 2  # 2 process 2d images
        if self.samplewise_center:
            x -= np.mean(x, axis=img_channel_axis, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, axis=img_channel_axis, keepdims=True) + 1e-7)

        if self.featurewise_center:
            if self.mean is not None:
                x -= self.mean
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_center`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.featurewise_std_normalization:
            if self.std is not None:
                x /= (self.std + 1e-7)
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`featurewise_std_normalization`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        if self.zca_whitening:
            if self.principal_components is not None:
                flatx = np.reshape(x, (-1, np.prod(x.shape[-3:])))
                whitex = np.dot(flatx, self.principal_components)
                x = np.reshape(whitex, x.shape)
            else:
                warnings.warn('This ImageDataGenerator specifies '
                              '`zca_whitening`, but it hasn\'t'
                              'been fit on any training data. Fit it '
                              'first by calling `.fit(numpy_data)`.')
        return x

    def get_random_transform(self, img_shape, seed=None):
        """Randomly augment a single image tensor.
        # Arguments
            x: 3D tensor, single image.
            seed: random seed.
        # Returns
            A randomly transformed version of the input (same shape).
        """
        # x is a single image, defined as (32, 32, 1)
        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1

        if seed is not None:
            np.random.seed(seed)

        if self.rotation_range:
            theta = np.random.uniform(-self.rotation_range,
                                      self.rotation_range)
        else:
            theta = 0

        if self.height_shift_range:
            try:
                tx = np.random.choice(self.height_shift_range)
                tx *= np.random.choice([-1, 1])
            except ValueError:
                tx = np.random.uniform(-self.height_shift_range,
                                       self.height_shift_range)
            if np.max(self.height_shift_range) < 1:
                tx *= img_shape[img_row_axis]
        else:
            tx = 0

        if self.width_shift_range:
            try:
                ty = np.random.choice(self.width_shift_range)
                ty *= np.random.choice([-1, 1])
            except ValueError:
                ty = np.random.uniform(-self.width_shift_range,
                                       self.width_shift_range)
            if np.max(self.width_shift_range) < 1:
                ty *= img_shape[img_col_axis]
        else:
            ty = 0

        if self.shear_range:
            shear = np.random.uniform(-self.shear_range,
                                      self.shear_range)
        else:
            shear = 0

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0],
                                       self.zoom_range[1], 2)

        flip_horizontal = (np.random.random() < 0.5) * self.horizontal_flip
        flip_vertical = (np.random.uniform() < 0.5) * self.vertical_flip

        channel_shift_intensity = None
        if self.channel_shift_range != 0:
            channel_shift_intensity = np.random.uniform(-self.channel_shift_range,
                                                        self.channel_shift_range)

        brightness = None
        if self.brightness_range is not None:
            brightness = np.random.uniform(self.brightness_range[0],
                                           self.brightness_range[1])

        transform_parameters = {'theta': theta,
                                'tx': tx,
                                'ty': ty,
                                'shear': shear,
                                'zx': zx,
                                'zy': zy,
                                'flip_horizontal': flip_horizontal,
                                'flip_vertical': flip_vertical,
                                'channel_shift_intensity': channel_shift_intensity,
                                'brightness': brightness}
        return transform_parameters

    def apply_transform(self, x, transform_parameters):
        """Applies a transformation to an image according to given parameters.

        # Arguments
            x: 3D tensor, single image. like 32x32x3 or 32x32x1 or 1x32x32 or 3x32x32
            transform_parameters: Dictionary with string - parameter pairs
                describing the transformation.
                Currently, the following parameters
                from the dictionary are used:
                - `'theta'`: Float. Rotation angle in degrees.
                - `'tx'`: Float. Shift in the x direction.
                - `'ty'`: Float. Shift in the y direction.
                - `'shear'`: Float. Shear angle in degrees.
                - `'zx'`: Float. Zoom in the x direction.
                - `'zy'`: Float. Zoom in the y direction.
                - `'flip_horizontal'`: Boolean. Horizontal flip.
                - `'flip_vertical'`: Boolean. Vertical flip.
                - `'channel_shift_intencity'`: Float. Channel shift intensity.
                - `'brightness'`: Float. Brightness shift intensity.

        # Returns
            A transformed version of the input (same shape).
        """

        img_row_axis = self.row_axis - 1
        img_col_axis = self.col_axis - 1
        if self.channel_axis == 1:
            img_channel_axis = self.channel_axis - 1
        else:
            img_channel_axis = self.channel_axis - 2

        x = apply_affine_transform(x, transform_parameters.get('theta', 0),
                                   transform_parameters.get('tx', 0),
                                   transform_parameters.get('ty', 0),
                                   transform_parameters.get('shear', 0),
                                   transform_parameters.get('zx', 1),
                                   transform_parameters.get('zy', 1),
                                   row_axis=img_row_axis,
                                   col_axis=img_col_axis,
                                   channel_axis=img_channel_axis,
                                   fill_mode=self.fill_mode,
                                   cval=self.cval,
                                   order=self.interpolation_order)

        if transform_parameters.get('channel_shift_intensity') is not None:
            x = apply_channel_shift(x,
                                    transform_parameters['channel_shift_intensity'],
                                    img_channel_axis)

        if transform_parameters.get('flip_horizontal', False):
            x = flip_axis(x, img_col_axis)

        if transform_parameters.get('flip_vertical', False):
            x = flip_axis(x, img_row_axis)

        if transform_parameters.get('brightness') is not None:
            x = apply_brightness_shift(x, transform_parameters['brightness'])
        return x

    def random_transform(self, x, seed=None):
        """Applies a random transformation to an image.

        # Arguments
            x: 3D tensor, single image.
            seed: Random seed.

        # Returns
            A randomly transformed version of the input (same shape).
        """
        params = self.get_random_transform(x.shape, seed)
        return self.apply_transform(x, params)


class Iterator(object):
    """Abstract base class for image data iterators.
    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(n, batch_size, shuffle, seed)

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
        self.reset()
        while True:
            if seed is not None:
                np.random.seed(seed + self.total_batches_seen)
            if self.batch_index == 0:
                index_array = np.arange(n)
                if shuffle:
                    index_array = np.random.permutation(n)
            current_index = (self.batch_index * batch_size) % n
            if n > current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = n - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)


class NumpyArrayIterator(Iterator):
    """Iterator yielding data from a Numpy array.
    # Arguments
        x: Numpy array of input data.
        y: Numpy array of targets data.
        image_data_generator: Instance of `ImageDataGenerator`
            to use for random transformations and normalization.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seed for data shuffling.
        data_format: String, one of `channels_first`, `channels_last`.
        save_to_dir: Optional directory where to save the pictures
            being yielded, in a viewable format. This is useful
            for visualizing the random transformations being
            applied, for debugging purposes.
        save_prefix: String prefix to use for saving sample
            images (if `save_to_dir` is set).
        save_format: Format to use for saving sample images
            (if `save_to_dir` is set).
    """

    def __init__(self,
                 x,
                 y,
                 image_data_generator,
                 data_type_x=None,
                 data_type_y=None,
                 batch_size=4,
                 chunk_info=None,
                 color_mode=1,
                 shuffle=False,
                 seed=None,
                 data_format='channels_last',
                 save_to_dir=None,
                 save_prefix='',
                 save_format='png',
                 dtype='float32',
                 data_x=None,
                 data_y=None):

        # x a group of data used len(x) = 1000; len(y) = 1000
        if data_format is None:
            data_format = K.image_data_format()

        # deal with x
        if data_type_x is None:
            raise ValueError('please enter the data type of x in order to get the data!')
        self.data_type_x = data_type_x
        if isinstance(x, str):
            cases_x = os.listdir(x)
            cases_x = [x+'/'+case for case in cases_x]
        elif isinstance(x, list):
            cases_x = x
        else:
            raise ValueError('the type of input x should be the '
                             'training folder path or list of training data path')
        if y is None:
            paths_x, data_x = compute_chunks(cases_x, self.data_type_x, chunk_info)
        else:
            paths_x, data_x = compute_chunks_multi(cases_x, self.data_type_x, chunk_info)
        self.x = paths_x
        self.data_x = data_x

        # deal with y
        # if y is None: classification; if y is not None: x --> y
        if y is not None:
            if data_type_y is None:
                raise ValueError('please enter the data type of y '
                                 'in order to get the data if y is not None!')
            self.data_type_y = data_type_y
            if isinstance(y, str):
                cases_y = os.listdir(y)
                cases_y = [y+'/'+case for case in cases_y]
            elif isinstance(y, list):
                cases_y = y
            else:
                raise ValueError('the type of input x should be the '
                                 'training folder path or list of training data path')
            paths_y, data_y = compute_chunks_multi(cases_y, self.data_type_y, chunk_info)
            self.y = paths_y
            self.data_y = data_y
        else:
            self.y = None
            self.data_type_y = None
            self.data_y = None

        if self.y is not None and len(self.x) != len(self.y):
            raise ValueError('X (images tensor) and y (labels) '
                             'should have the same length. '
                             'Found: number of X = %s, y.shape = %s' %
                             (len(self.x), len(self.y)))

        self.image_data_generator = image_data_generator
        self.chunk_info = chunk_info
        self.color_mode = color_mode
        self.data_format = data_format
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        self.dtype = dtype
        self.image_data_generator.chunk_num = len(self.x)

        super(NumpyArrayIterator, self).__init__(self.image_data_generator.chunk_num, batch_size, shuffle, seed)

    def augmentation(self, chunk_data, transform_parameters, aug_axis):
        process_type = np.zeros(chunk_data.shape, dtype=K.floatx())
        size_x = self.chunk_info['size_x']
        size_y = self.chunk_info['size_y']
        size_z = self.chunk_info['size_z']

        if aug_axis == 0:
            for n in range(size_x):
                if self.data_format == 'channels_last':
                    single_img = chunk_data[n, :, :, :]
                    single_x = self.image_data_generator.apply_transform(single_img, transform_parameters)
                    process_type[n, :, :, :] = single_x
                else:
                    single_img = chunk_data[:, n, :, :]
                    single_x = self.image_data_generator.apply_transform(single_img, transform_parameters)
                    process_type[:, n, :, :] = single_x
        elif aug_axis == 1:
            for n in range(size_y):
                if self.data_format == 'channels_last':
                    single_img = chunk_data[:, n, :, :]
                    single_x = self.image_data_generator.apply_transform(single_img, transform_parameters)
                    process_type[:, n, :, :] = single_x
                else:
                    single_img = chunk_data[:, :, n, :]
                    single_x = self.image_data_generator.apply_transform(single_img, transform_parameters)
                    process_type[:, :, n, :] = single_x
        else:
            for n in range(size_z):
                if self.data_format == 'channels_last':
                    single_img = chunk_data[:, :, n, :]
                    single_x = self.image_data_generator.apply_transform(single_img, transform_parameters)
                    process_type[:, :, n, :] = single_x
                else:
                    single_img = chunk_data[:, :, :, n]
                    single_x = self.image_data_generator.apply_transform(single_img, transform_parameters)
                    process_type[:, :, :, n] = single_x
        return process_type

    def next(self):
        """For python 3.x.
        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel

        if self.y is None:  # classification
            num_channel_x = self.color_mode
            num_channel_y = len(self.data_type_x)
        else:  # regression
            num_channel_x = len(self.data_type_x)
            num_channel_y = len(self.data_type_y)
        size_x = self.chunk_info['size_x']
        size_y = self.chunk_info['size_y']
        size_z = self.chunk_info['size_z']

        if self.data_format == 'channels_last':
            batch_x = np.zeros(
                tuple([current_batch_size] + [size_x, size_y, size_z, num_channel_x]), dtype=K.floatx())
            if self.y is None:
                batch_y = np.zeros(
                    tuple([current_batch_size] + [num_channel_y]), dtype=K.floatx())
            else:
                batch_y = np.zeros(
                    tuple([current_batch_size] + [size_x, size_y, size_z, num_channel_y]), dtype=K.floatx())
        else:
            batch_x = np.zeros(
                tuple([current_batch_size] + [num_channel_x, size_x, size_y, size_z]), dtype=K.floatx())
            if self.y is None:
                batch_y = np.zeros(
                    tuple([current_batch_size] + [num_channel_y]), dtype=K.floatx())
            else:
                batch_y = np.zeros(
                    tuple([current_batch_size] + [num_channel_y, size_x, size_y, size_z]), dtype=K.floatx())

        # build batch of image data
        for i, j in enumerate(index_array):
            # [[case, data_type], x, y, z] or [case, x, y, z]
            cur_x = self.x[j]
            if self.y is None:
                data_x = load_chunk_data(cur_x, self.data_x, self.chunk_info, self.color_mode, self.data_format)
            else:
                data_x = load_chunk_data_multi(cur_x, self.data_x, self.chunk_info, self.data_type_x, self.color_mode, self.data_format)
            # data augmentation
            aug_axis = j % 3
            if aug_axis == 0:
                if self.data_format == 'channels_last':
                    img_shape = (size_y, size_z, self.color_mode)
                else:
                    img_shape = (self.color_mode, size_y, size_z)
            elif aug_axis == 1:
                if self.data_format == 'channels_last':
                    img_shape = (size_x, size_z, self.color_mode)
                else:
                    img_shape = (self.color_mode, size_x, size_z)
            else:
                if self.data_format == 'channels_last':
                    img_shape = (size_x, size_y, self.color_mode)
                else:
                    img_shape = (self.color_mode, size_x, size_y)
            transform_parameters = self.image_data_generator.get_random_transform(img_shape, j)

            for k in range(num_channel_x):
                if self.data_format == 'channels_last':
                    temp = np.zeros((size_x, size_y, size_z, self.color_mode))
                    if self.color_mode == 1:
                        temp[:, :, :, 0] = data_x[:, :, :, k]
                        temp = self.augmentation(temp, transform_parameters, aug_axis)
                        temp = self.image_data_generator.standardize(temp)
                        data_x[:, :, :, k] = temp[:, :, :, 0]
                    else:
                        down = k * self.color_mode
                        up = (k+1) * self.color_mode
                        temp = data_x[:, :, :, down:up]
                        temp = self.augmentation(temp, transform_parameters, aug_axis)
                        temp = self.image_data_generator.standardize(temp)
                        data_x[:, :, :, down:up] = temp
                else:
                    temp = np.zeros((self.color_mode, size_x, size_y, size_z))
                    if self.color_mode == 1:
                        temp[0, :, :, :] = data_x[k, :, :, :]
                        temp = self.augmentation(temp, transform_parameters, aug_axis)
                        temp = self.image_data_generator.standardize(temp)
                        data_x[k, :, :, :] = temp[0, :, :, :]
                    else:
                        down = k * self.color_mode
                        up = (k+1) * self.color_mode
                        temp = data_x[down:up, :, :, :]
                        temp = self.augmentation(temp, transform_parameters, aug_axis)
                        temp = self.image_data_generator.standardize(temp)
                        data_x[down:up, :, :, :] = temp
            batch_x[i] = data_x

            if self.y:
                cur_y = self.y[j]
                case_x = cur_x[0]
                case_y = cur_y[0]
                if case_x != case_y:
                    raise ValueError('the input case and the target case are not matched!')
                elif cur_x[1] != cur_y[1] or cur_x[2] != cur_y[2] or cur_x[3] != cur_y[3]:
                    raise ValueError('the chunk info of the input and the target is not matched!')
                else:
                    data_y = load_chunk_data_multi(cur_y, self.data_y, self.chunk_info, self.data_type_y,
                                                   self.color_mode, self.data_format)
                    data_y = self.augmentation(data_y, transform_parameters, aug_axis)
                    data_y = self.image_data_generator.standardize(data_y)
                    batch_y[i] = data_y
            else:
                cur_data_type = cur_x[0][1]
                y_labels = {}
                for p in range(len(self.data_type_x)):
                    label = [0 for q in range(len(self.data_type_x))]
                    label[p] = 1
                    y_labels[self.data_type_x[p]] = np.array(label)
                batch_y[i] = y_labels[cur_data_type]

        if self.save_to_dir:
            for i in range(current_batch_size):
                chunk_x = batch_x[i]
                for j in range(self.chunk_info['size_z']):
                    if self.data_format == 'channels_last':
                        # chunk_x[0]: 32x32x1 or 32x32x3
                        img = array_to_img(chunk_x[j, :, :, :], self.data_format, scale=True)
                    else:
                        # chunk_x 1x32x32
                        img = array_to_img(chunk_x[:, :, :, j], self.data_format, scale=True)
                    fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                      index=current_index + i*self.chunk_info['size_z'] + j,
                                                                      hash=np.random.randint(1e4),
                                                                      format=self.save_format)
                    img.save(os.path.join(self.save_to_dir, fname))

        return batch_x, batch_y
        # if self.y:
        #     return batch_x, batch_y
        # else:
        #     return batch_x
