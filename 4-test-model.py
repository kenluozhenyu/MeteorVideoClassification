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
import math

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


# @threadsafe_generator

def frame_generator_no_random(data_folder, i_from, i_to, data_type):

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

    # while 1:
    X, y, z = [], [], []

    i_count = 0
    b_finished = False

    # Generate batch_size samples.
    # for _ in range(batch_size):

    # Reset to be safe.
    # sequence = None
    sequence = []

    # Get a random class.
    # selected_class = random.choice(class_list)

    for selected_class in class_list:
        if b_finished:
            break

        class_folder = os.path.join(data_folder, selected_class)

        # Get all samples' folder name.
        sample_list = os.listdir(class_folder)

        # Get a random sample.
        # selected_sample = random.choice(sample_list)

        for selected_sample in sample_list:

            i_count += 1
            if i_count < i_from:
                continue

            if i_count > i_to:
                b_finished = True
                break

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
                frame_list = []
                step = 0
                for i in range(0, environment.SEQ_LENGTH):
                    index = int(i + step)
                    if index > total_frames - 1:
                        index = total_frames - 1
                    frame_list.append(frames_in_folder[index])
                    step += step_to_skip
            else:
                # Fill all remaining frames with the last frame in the video
                frame_list = frames_in_folder
                for i in range(1, environment.SEQ_LENGTH - total_frames + 1):
                    frame_list.append(frames_in_folder[total_frames - 1])

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

            z.append(selected_sample)



        # print(y)
        # print(np.array(y))
        # yield np.array(X), np.array(y)

    # print(np.array(X))
    # print(np.array(y))
    # print(np.array(z))
    return np.array(X), np.array(y), np.array(z)

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

    batch_size = 4

    model = conv_rnn.build_Conv_RNN()
    print(model.summary())

    checkpoint_folder = os.path.join(environment.TRAINING_FOLDER, '_checkpoints')
    model.load_weights(os.path.join(checkpoint_folder, 'LSTM-train.064-0.400.hdf5'))
    # model.load_weights("D:/meteor-monitor/360-cropped-subtracted/_checkpoints/LSTM-train.030-0.372.hdf5")

    test_sample_folder = os.path.join(environment.TRAINING_FOLDER, 'Test')
    # test_generator_Data, test_generator_Lable, test_generator_File = frame_generator_no_random(test_sample_folder, 'Test')
    # test_generator_Data, test_generator_Lable, test_generator_File = frame_generator(batch_size, test_sample_folder, 'Test')

    # Due to large sample size, would not be enough physical memory to get all
    # data be fit in at one time. Need to do that in several times.
    #
    test_sample_size = 516
    segment_step = 100

    segment_num = math.ceil(test_sample_size / segment_step)
    # print(segment_num)

    correct_count = 0
    count = 0

    for j in range(segment_num):
        i_from = j*segment_step + 1
        i_to = j*segment_step + segment_step

        if i_to > test_sample_size:
            i_to = test_sample_size

        print("\n")
        print(i_from)
        print(i_to)

        # test_generator_Data, test_generator_Lable, test_generator_File = [], [], []

        test_generator_Data, test_generator_Lable, test_generator_File = \
            frame_generator_no_random(test_sample_folder, i_from, i_to, 'Test')

        print(test_generator_Lable)
        print(test_generator_File)

        # scores_predict = model.predict_generator(generator=test_generator)
        scores_predict = model.predict(test_generator_Data, batch_size=batch_size, verbose=1)
        print("\n")
        print(scores_predict)

        print("\n")
        for i in range(len(scores_predict)):
            count += 1
            correct_predict = False

            # if i < 278:
            if count <= 278:
                sample_class = "Meteor"
                if scores_predict[i][0] >= 0.5:
                    correct_count += 1
                    correct_predict = True
            else:
                sample_class = "Others"
                if scores_predict[i][0] <= 0.5:
                    correct_count += 1
                    correct_predict = True

            print("%2d. %s -- %45s: predicted result: [%0.8f %0.8f], %r" % (
                count, sample_class, test_generator_File[i], scores_predict[i][0], scores_predict[i][1], correct_predict))

    print("\n%d out of %d correct. %f accuracy" % (correct_count, count, correct_count / count))


if __name__ == "__main__":
    main()
