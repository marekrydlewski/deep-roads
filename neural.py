import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD, Adam
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

__modelNameBaseNetwork = "base_neural.h5"
__modelNameSpecializedNetwork = "specialized_neural.h5"
__modelNameSpecializedSmallNetwork = "specialized_small_neural.h5"
__windowSize = 20
__windowSizeSmall = 14

IS_ROAD = np.array([1, 0])
IS_NOT_ROAD = np.array([0, 1])
THRESHOLD = 120
WINDOW = 20


# network checking if at least 1px in a window is a road
def get_base_network():
    try:
        return load_model(__modelNameBaseNetwork)
    except:
        print('Create new model of base network')

        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(__windowSize, __windowSize, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
        model.add(Dropout(0.2))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='valid'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu', kernel_constraint=maxnorm(3)))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu', kernel_constraint=maxnorm(3)))
        model.add(Dense(2, activation='softmax'))

        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model


def save_base_network(net):
    net.save(__modelNameBaseNetwork)


# network checking if at least 1px in the centre of a window is a road (2px x 2px)
def get_specialized_network():
    try:
        return load_model(__modelNameSpecializedNetwork)
    except:
        print('Create new model of specialized network')

        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(__windowSize, __windowSize, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
        model.add(Dropout(0.2))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='valid'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu', kernel_constraint=maxnorm(3)))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu', kernel_constraint=maxnorm(3)))
        model.add(Dense(2, activation='softmax'))

        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model


def save_specialized_network(net):
    net.save(__modelNameSpecializedNetwork)


# network checking if at least 1px in the centre of a window is a road (2px x 2px)
def get_specialized_small_network():
    try:
        return load_model(__modelNameSpecializedSmallNetwork)
    except:
        print('Create new model of specialized network')

        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(__windowSizeSmall, __windowSizeSmall, 3), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
        model.add(Dropout(0.2))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='valid', kernel_constraint=maxnorm(3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='valid'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu', kernel_constraint=maxnorm(3)))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu', kernel_constraint=maxnorm(3)))
        model.add(Dense(2, activation='softmax'))

        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model


def save_specialized_small_network(net):
    net.save(__modelNameSpecializedSmallNetwork)