from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical
from keras.regularizers import l2
import numpy as np
from collections import deque
import sys
import threading
import time
import os.path
import glob
import random

import environment
import conv_rnn

import tensorflow as tf
from keras import backend as K


class threadsafe_iterator:
    def __init__(self, iterator):
        self.iterator = iterator
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return next(self.iterator)


def threadsafe_generator(func):
    """Decorator"""
    def gen(*a, **kw):
        return threadsafe_iterator(func(*a, **kw))
    return gen


@threadsafe_generator
def frame_generator(batch_size, data_folder, data_type):

    '''

    Files to be arranged in this way
    The "data_folder" parm should point to the "train" or "validate" folder path

    train \  (The "data_folder" parm pointing to this)
        -- meteor \  (class folder)
            -- < xxxxx folder >  (sample folder)
                -- *.jpg
            -- < yyyyy folder >
                -- *.jpg

        -- Plane \
            -- < xxxxx folder >
                -- *.jpg
            -- < yyyyy folder >
                -- *.jpg

        -- Others \
            -- < xxxxx folder >
                -- *.jpg
            -- < yyyyy folder >
                -- *.jpg

    validate \
        -- meteor \
        ...
    '''

    '''
    Get all sub folders in the "data_folder" -> the "class" list
    '''
    class_list = os.listdir(data_folder)
    print('\n')
    print("Creating %s generator with %d classes." % (data_type, len(class_list)))
    print(class_list)

    while 1:
        X, y = [], []

        # Generate batch_size samples.
        for _ in range(batch_size):
            # Reset to be safe.
            # sequence = None
            sequence = []

            # Get a random class.
            selected_class = random.choice(class_list)
            class_folder = os.path. join(data_folder, selected_class)

            # Get all samples' folder name.
            sample_list = os.listdir(class_folder)

            # Get a random sample.
            selected_sample = random.choice(sample_list)
            sample_folder = os.path.join(class_folder, selected_sample)

            # print(selected_sample)

            # frames = get_frames_for_sample(sample_folder)
            # Get the image file list
            frames_in_folder = sorted(glob.glob(os.path.join(sample_folder, '*.jpg')))
            total_frames = len(frames_in_folder)
            # print('Total frames in the sample is: %d', total_frames)

            # frames = rescale_list(frames, SEQ_LENGTH)
            # If we have too many images, just select one image from very N images
            # assert total_frames >= SEQ_LENGTH
            # skip = len(frames_in_folder) // SEQ_LENGTH
            # frame_list = [frames_in_folder[i] for i in range(0, len(frames_in_folder), skip)]

            if total_frames > environment.SEQ_LENGTH:
                # Need to select frames from the original list evenly
                step_to_skip = (total_frames - environment.SEQ_LENGTH) / (environment.SEQ_LENGTH - 1)
                frame_list=[]
                step=0
                for i in range(0, environment.SEQ_LENGTH):
                    index = int(i + step)
                    if index > total_frames-1:
                        index = total_frames-1
                    frame_list.append(frames_in_folder[index])
                    step += step_to_skip
            else:
                # Fill all remaining frames with the last frame in the video
                frame_list = frames_in_folder
                for i in range(1, environment.SEQ_LENGTH - total_frames + 1):
                    frame_list.append(frames_in_folder[total_frames-1])

            # print('Number of frames selected: %d', len(frame_list))
            # Build the image sequence
            # sequence = build_image_sequence(frame_list)
            h, w, _ = environment.IMAGE_SHAPE
            sequence = []

            for frame in frame_list:
                image = load_img(frame, grayscale=False, target_size=(h, w))
                img_arr = img_to_array(image)

                # We don't need 3 color layers (RGB), just need one layer
                # img_arr_1_layer = img_arr[ :, :, 0].reshape(h, w, 1)
                img_arr = img_arr[:, :, 0].reshape(h, w, 1)

                # x = (img_arr[ :, :, 0] / 255.).astype(np.float32)
                # x = [img_arr[:, :, 0] / 255.]
                # x = (img_arr_1_layer / 255.).astype(np.float32)
                x = (img_arr / 255.).astype(np.float32)

                sequence.append(x)

            X.append(sequence)

            # Get the class index
            label_encoded = class_list.index(selected_class)
            # print(selected_class)
            # print(label_encoded)
            label_hot = to_categorical(label_encoded, len(class_list))
            # print(label_hot)
            y.append(label_hot)

        # print(y)
        # print(np.array(y))
        yield np.array(X), np.array(y)


def main():
    # =========================================================================
    # Configure if we'd use GPU or CPU for the training

    num_cores = 24

    GPU = True
    # GPU = False

    if GPU:
        num_GPU = 1
        num_CPU = 1
    else:
        num_CPU = 1
        num_GPU = 0

    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores, \
                            inter_op_parallelism_threads=num_cores, allow_soft_placement=True, \
                            device_count={'CPU': num_CPU, 'GPU': num_GPU})
    session = tf.Session(config=config)
    K.set_session(session)

    # Training parameters
    batch_size = 3
    nb_epoch = 100
    training_sample_size = 1200
    validation_sample_size = 152

    # Get the model
    model = conv_rnn.build_Conv_RNN()
    print(model.summary())

    # Helper: Save the model.
    checkpoint_folder = os.path.join(environment.TRAINING_FOLDER, '_checkpoints')
    if not os.path.exists(checkpoint_folder):
        os.mkdir(checkpoint_folder)

    checkpointer = ModelCheckpoint(
        filepath=os.path.join(checkpoint_folder, 'LSTM-train.{epoch:03d}-{val_loss:.3f}.hdf5'),
        verbose=1,
        save_best_only=True)

    # Helper: TensorBoard
    logs_folder = os.path.join(environment.TRAINING_FOLDER, '_logs')
    if not os.path.exists(logs_folder):
        os.mkdir(logs_folder)

    tensorboard_folder = os.path.join(environment.TRAINING_FOLDER, '_logs', 'LSTM')
    if not os.path.exists(tensorboard_folder):
        os.mkdir(tensorboard_folder)

    tb = TensorBoard(log_dir=tensorboard_folder)

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=20)

    # Helper: Save results.
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join(logs_folder, 'LSTM-training-' + \
                                        str(timestamp) + '.log'))

    # Get samples per epoch.
    steps_per_epoch = training_sample_size / batch_size

    # Save the trained model
    trained_model_folder = os.path.join(environment.TRAINING_FOLDER, '_trained_model')
    if not os.path.exists(trained_model_folder):
        os.mkdir(trained_model_folder)

    # Get generators.
    train_generator = frame_generator(batch_size, os.path.join(environment.TRAINING_FOLDER, 'Train'), 'Train')
    val_generator = frame_generator(batch_size, os.path.join(environment.TRAINING_FOLDER, 'Validate'), 'Validation')

    # Do the training now
    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=nb_epoch,
        verbose=1,
        callbacks=[tb, early_stopper, csv_logger, checkpointer],
        validation_data=val_generator,
        validation_steps=validation_sample_size / batch_size,
        workers=12)

    model.save_weights(os.path.join(trained_model_folder, 'meteor-video-model.h5'))


if __name__ == "__main__":
    main()
