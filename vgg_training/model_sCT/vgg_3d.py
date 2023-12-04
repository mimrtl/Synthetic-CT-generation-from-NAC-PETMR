from keras.models import Sequential
from keras.layers import Dense,Conv3D,MaxPooling3D,Flatten


def vgg(input_shape=(32, 32, 32, 1)):
    model = Sequential()
    model.add(Conv3D(64, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block1_conv1',
                     input_shape=input_shape))
    model.add(Conv3D(64, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block1_conv2'))
    model.add(MaxPooling3D(
            pool_size=(2, 2, 2),
            strides=(2, 2, 2),
            name='block1_pool',
            padding='same'
    ))
    model.add(Conv3D(128, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block2_conv1'))
    model.add(Conv3D(128, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block2_conv2'))

    model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same',
                           name='block2_pool'))

    model.add(Conv3D(256, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block3_conv1'))
    model.add(Conv3D(256, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block3_conv2'))
    model.add(Conv3D(256, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block3_conv3'))
    model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same', name='block3_pool'))
    model.add(Conv3D(512, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block4_conv1'))
    model.add(Conv3D(512, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block4_conv2'))
    model.add(Conv3D(512, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block4_conv3'))
    model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same', name='block4_pool'))
    model.add(Conv3D(512, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block5_conv1'))
    model.add(Conv3D(512, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block5_conv2'))

    model.add(Conv3D(512, (3, 3, 3),
                     activation='relu',
                     padding='same',
                     name='block5_conv3'))
    model.add(MaxPooling3D((2, 2, 2), strides=(2, 2, 2), padding='same', name='block5_pool'))

    model.add(Flatten(name='flatten'))
    model.add(Dense(4096, activation='relu', name='fc1'))

    model.add(Dense(4096, activation='relu', name='fc2'))

    model.add(Dense(2, activation='sigmoid', name='predictions'))
    return model


if __name__ == '__main__':
    vgg()
