import numpy
from keras.datasets import cifar10
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

__modelNameBaseNetwork = "base_neural.h5"
__modelNameSpecializedNetwork = "base_neural.h5"
__epochs = 25
__lrate = 0.01


def get_base_network():
    try:
        return load_model(__modelNameBaseNetwork)
    except:
        print('Create new model of base network')

        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(3, 20, 20), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
        model.add(Dropout(0.2))
        model.add(Conv2D(26, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(256, activation='relu', kernel_constraint=maxnorm(3)))
        model.add(Dropout(0.5))
        model.add(Dense(2, activation='softmax'))

        # Compile model
        decay = __lrate / __epochs
        sgd = SGD(lr=__lrate, momentum=0.9, decay=decay, nesterov=False)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        return model


def get_specialized_network():
    try:
        return load_model(__modelNameSpecializedNetwork)
    except:
        print('Create new model of specialized network')

        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), activation='relu', padding='same'))
        model.add(Dropout(0.2))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(Dropout(0.2))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dropout(0.2))
        model.add(Dense(256, activation='relu', kernel_constraint=maxnorm(3)))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu', kernel_constraint=maxnorm(3)))
        model.add(Dropout(0.2))
        model.add(Dense(2, activation='softmax'))

        # Compile model
        decay = __lrate / __epochs
        sgd = SGD(lr=__lrate, momentum=0.9, decay=decay, nesterov=False)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        return model

def roads(image):
    pass



if __name__ == "__main__":
    pass