from keras.layers import Dense, Flatten, Dropout, ZeroPadding3D, InputLayer
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import (Conv2D, MaxPooling3D, Conv3D, MaxPooling2D)
from keras.regularizers import l2

import environment


def build_Conv_RNN():
    model = Sequential()

    model.add(TimeDistributed(Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same'),
                              input_shape=environment.VIDEO_FRAME_SHAPE))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))))
    model.add(TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))))
    model.add(TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same')))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides=(2, 2))))
    model.add(TimeDistributed(Flatten()))
    model.add(Dropout(0.5))
    # model.add(LSTM(256, return_sequences=False, dropout=0.5, implementation=0))
    # model.add(LSTM(256, return_sequences=False, recurrent_dropout=0.5, dropout=0.5, implementation=1))
    # model.add(LSTM(256, return_sequences=False, recurrent_dropout=0.5, dropout=0.5, implementation=1, W_regularizer=l2(0.1)))
    model.add(LSTM(256, return_sequences=False, recurrent_dropout=0.5, dropout=0.5, implementation=1,
                   W_regularizer=l2(0.001)))

    # added at 2018-3-19 for trial
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(environment.NB_CLASSES, activation='softmax'))

    # Set the metrics. Only use top k if there's a need.
    metrics = ['accuracy']
    if environment.NB_CLASSES >= 10:
        metrics.append('top_k_categorical_accuracy')

    # Now compile the network.
    optimizer = Adam(lr=1e-5, decay=1e-6)
    # optimizer = 'adadelta'
    model.compile(loss='categorical_crossentropy', optimizer=optimizer,
                  metrics=metrics)

    return model
