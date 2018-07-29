import os.path
from subprocess import call
from shutil import copyfile
from distutils.dir_util import copy_tree
import random
import environment


def distribute_method_1_samples(sample_type):
    processed_method = environment.PRE_PROCESSED_METHOD

    training_sample_folder = environment.TRAINING_FOLDER
    subtracted_image_folder = '%s/%s/%s/4_FINAL_image_compressed_subtracted' % (environment.ORIGINAL_DATA_FOLDER, sample_type, processed_method)

    '''
    Meteor samples
    239 x 4 (x 4 for flipping)
        200 x 4 train
        24 x 4 validate
        15 x 4 test
    
    'Others' samples
    111 x 4
        90 x 4 train
        14 x 4 validate
        7 x 4 test
    
    '''
    numTrain = 0
    numValidate = 0
    numTest = 0

    '''
    if sample_type == 'Meteor':
        numTrain = 200 * 4
        numValidate = 24 * 4
        numTest = 15 * 4
    elif sample_type == 'Others':
        numTrain = 90 * 4
        numValidate = 14 * 4
        numTest = 7 * 4
    '''

    subfolders_list = os.listdir(subtracted_image_folder)
    sample_size = len(subfolders_list)

    numTrain = int(sample_size * 0.8)
    numValidate = int(sample_size * 0.1)
    numTest = sample_size - numTrain - numValidate

    subfolders_Train_list = random.sample(subfolders_list, numTrain)

    # generate the left list subtracted by the train list
    subfolders_No_Train_list = [item for item in subfolders_list if item not in subfolders_Train_list]

    subfolders_Validate_list = random.sample(subfolders_No_Train_list, numValidate)
    subfolders_Test_list = [item for item in subfolders_No_Train_list if item not in subfolders_Validate_list]

    # Folder directory structure:
    #
    # /project/
    #         Training/  <- environment.TRAINING_SAMPLE_FOLDER
    #                 Train/
    #                       Meteor
    #                             image-folder1
    #                             image-folder2
    #                             ...
    #                       Others
    #                 Validation/
    #                       Meteor
    #                       Others
    #                 Test/
    #                       Meteor
    #                       Others
    #
    print("\nProcessing %s Training samples..." % sample_type)

    dest_folder = os.path.join(training_sample_folder, "Train")
    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)

    dest_folder = os.path.join(training_sample_folder, "Train", "%s" % sample_type)
    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)

    count = 0
    for subfolder in subfolders_Train_list:
        src_folder = os.path.join(subtracted_image_folder, subfolder)
        dest_folder = os.path.join(training_sample_folder, "Train", "%s" % sample_type, subfolder)

        count += 1
        print("....(%3d of %d), %s ...." % (count, numTrain, subfolder))

        os.mkdir(dest_folder)
        copy_tree(src_folder, dest_folder)

    print("\nProcessing %s Validation samples..." % sample_type)

    dest_folder = os.path.join(training_sample_folder, "Validate")
    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)

    dest_folder = os.path.join(training_sample_folder, "Validate", "%s" % sample_type)
    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)

    count = 0
    for subfolder in subfolders_Validate_list:
        src_folder = os.path.join(subtracted_image_folder, subfolder)
        dest_folder = os.path.join(training_sample_folder, "Validate", "%s" % sample_type, subfolder)

        count += 1
        print("....(%3d of %d), %s ...." % (count, numValidate, subfolder))

        os.mkdir(dest_folder)
        copy_tree(src_folder, dest_folder)

    print("\nProcessing %s Test samples..." % sample_type)

    dest_folder = os.path.join(training_sample_folder, "Test")
    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)

    dest_folder = os.path.join(training_sample_folder, "Test", "%s" % sample_type)
    if not os.path.exists(dest_folder):
        os.mkdir(dest_folder)

    count = 0
    for subfolder in subfolders_Test_list:
        src_folder = os.path.join(subtracted_image_folder, subfolder)
        dest_folder = os.path.join(training_sample_folder, "Test", "%s" % sample_type, subfolder)

        count += 1
        print("....(%3d of %d), %s ...." % (count, numTest, subfolder))

        os.mkdir(dest_folder)
        copy_tree(src_folder, dest_folder)


def main():
    # 'Meteor' or 'Noise' or 'Plane' or 'fortest'
    sample_type = ['Meteor', 'Others']
    # sample_type = ['Others']

    for sample in sample_type:
        distribute_method_1_samples(sample)


if __name__ == "__main__":
    main()
