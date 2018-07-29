import cv2
import os.path
from subprocess import call
from shutil import copyfile
import environment

'''
Pre-process order for meteor video files:
=================================================
Step 1: Crop the video caption in the bottom

Step 2: Compress the video file

Step 3: Flip the video

Step 4: Subtract the image background
        and extract the image frame
        *** This are the materials used for training ***

Step 5: (Just to verify) Combine the subtracted images back to video. To check the quality
'''


# 'Meteor' or 'Others' or 'fortest'
def pre_process_method_1(sample_type):
    # sample_type = 'Meteor' or 'Others' or 'fortest'

    data_folder = environment.ORIGINAL_DATA_FOLDER

    # processed_method = 'Processed-method1'
    processed_method = environment.PRE_PROCESSED_METHOD

    # video_folder = 'D:/meteor-monitor/data/%s/video' % sample_type
    video_folder = '%s/%s/video' % (data_folder, sample_type)

    '''
    cropped_video_folder = 'D:/meteor-monitor/data/%s/%s/1_video_cropped' % (sample_type, processed_method)
    compressed_video_folder = 'D:/meteor-monitor/data/%s/%s/2_video_compressed' % (sample_type, processed_method)
    flipped_video_folder = 'D:/meteor-monitor/data/%s/%s/3_video_flipped' % (sample_type, processed_method)
    subtracted_image_folder = 'D:/meteor-monitor/data/%s/%s/4_FINAL_image_compressed_subtracted' % (sample_type, processed_method)
    subtracted_bg_video_folder = 'D:/meteor-monitor/data/%s/%s/5_video_bg_subtracted' % (sample_type, processed_method)
    '''
    #
    # /project/
    #          data/ <- ORIGINAL_DATA_FOLDER
    #               Meteor/                  <- sample_type
    #                     Processed-method1/
    #                                      (several sub-folders)
    #
    process_method_folder = '%s/%s/%s' % (data_folder, sample_type, processed_method)

    cropped_video_folder = '%s/1_video_cropped' % process_method_folder
    compressed_video_folder = '%s/2_video_compressed' % process_method_folder
    flipped_video_folder = '%s/3_video_flipped' % process_method_folder
    subtracted_image_folder = '%s/4_FINAL_image_compressed_subtracted' % process_method_folder
    subtracted_bg_video_folder = '%s/5_video_bg_subtracted' % process_method_folder

    if not os.path.exists(process_method_folder):
        os.mkdir(process_method_folder)

    if not os.path.exists(cropped_video_folder):
        os.mkdir(cropped_video_folder)

    if not os.path.exists(compressed_video_folder):
        os.mkdir(compressed_video_folder)

    if not os.path.exists(flipped_video_folder):
        os.mkdir(flipped_video_folder)

    if not os.path.exists(subtracted_image_folder):
        os.mkdir(subtracted_image_folder)

    if not os.path.exists(subtracted_bg_video_folder):
        os.mkdir(subtracted_bg_video_folder)

    # For step 1, cropping
    cropped_width = 720
    cropped_height = 560
    original_bit_rate = '195312k'

    # For step 2, compressing
    # compress_size = '480x384'
    # compress_size = '360x288'
    compress_size = '360x280'
    compress_bit_rate = '86805k'

    # ===============================================
    # Step 1: Crop the video files to remove
    #         the captions
    # ===============================================
    files = os.listdir(video_folder)

    for filename in files:
        # filename_no_ext = filename.split('.')[0]
        src = os.path.join(video_folder, filename)
        dest = os.path.join(cropped_video_folder, filename)

        # call('D:/Software/ffmpeg-20180226-f4709f1-win64-static/bin/ffmpeg -i '
        call('%s -i ' % environment.FFMPEG_COMMAND
             + src
             + ' -filter:v "crop=%d:%d:0:0" -b:v %s ' % (cropped_width, cropped_height, original_bit_rate)
             + dest)

    # ===============================================
    # Step 2: Compress the video files
    # ===============================================
    files = os.listdir(cropped_video_folder)

    for filename in files:
        # filename_no_ext = filename.split('.')[0]
        src = os.path.join(cropped_video_folder, filename)
        dest = os.path.join(compressed_video_folder, filename)

        '''
        -y
        覆盖输出文件
        -threads
        8
        指定多线程
        -i
        输入文件
        -s
        分辨率
        输出文件直接指定
    
        -b:v 1M -vcodec h264
        确保足够的转换质量
        '''
        # call("D:/Software/ffmpeg-20180226-f4709f1-win64-static/bin/ffmpeg -y -threads 8 -i "
        call("%s -y -threads 8 -i " % environment.FFMPEG_COMMAND
             + src
             # + " -s %s -b:v 86805k " % compress_size  # -vcodec h264
             + " -s %s -b:v %s " % (compress_size, compress_bit_rate)
             + dest)

    # ===============================================
    # Step 3: Flipped the compressed video files
    # ===============================================
    files = os.listdir(compressed_video_folder)
    for filename in files:
        # filename_no_ext = filename.split('.')[0]
        src = os.path.join(compressed_video_folder, filename)

        # Copy the original compressed video file to the "flipped" folder as well
        # so as to facilitate later processing
        dest = os.path.join(flipped_video_folder, filename)
        copyfile(src, dest)

        # Start to flip

        filename_no_ext = filename.split('.')[0]

        filename_new = filename_no_ext + "-h-v.avi"
        dest = os.path.join(flipped_video_folder, filename_new)

        # call('D:/Software/ffmpeg-20180226-f4709f1-win64-static/bin/ffmpeg -y -threads 8 -i '
        call('%s -y -threads 8 -i ' % environment.FFMPEG_COMMAND
             + src
             # + ' -vf "transpose=2, transpose=2" -b:v 86805k '  # 195312k for uncompressed filed
             + ' -vf "transpose=2, transpose=2" -b:v %s ' % compress_bit_rate
             + dest)

        filename_new = filename_no_ext + "-v.avi"
        dest = os.path.join(flipped_video_folder, filename_new)
        # call('D:/Software/ffmpeg-20180226-f4709f1-win64-static/bin/ffmpeg -y -threads 8 -i '
        call('%s -y -threads 8 -i ' % environment.FFMPEG_COMMAND
             + src
             # + ' -vf "transpose=2, transpose=3" -b:v 86805k '
             + ' -vf "transpose=2, transpose=3" -b:v %s ' % compress_bit_rate
             + dest)

        filename_new = filename_no_ext + "-h.avi"
        dest = os.path.join(flipped_video_folder, filename_new)
        # call('D:/Software/ffmpeg-20180226-f4709f1-win64-static/bin/ffmpeg -y -threads 8 -i '
        call('%s -y -threads 8 -i ' % environment.FFMPEG_COMMAND
             + src
             # + ' -vf "transpose=3, transpose=2" -b:v 86805k '
             + ' -vf "transpose=3, transpose=2" -b:v %s ' % compress_bit_rate
             + dest)

    # ===============================================
    # Step 4: Subtract the image background
    #         and extract the image frames
    # ===============================================
    files = os.listdir(flipped_video_folder)

    for filename in files:
        print("\nProcessing %s ..." % filename)

        filename_no_ext = filename.split('.')[0]

        # mkdir one folder for each video's frames
        strfolder = os.path.join(subtracted_image_folder, filename_no_ext)
        os.mkdir(strfolder)

        dest = os.path.join(subtracted_image_folder, filename_no_ext, filename_no_ext + '-%04d.jpg')

        # The "-qscale:v 2" option is to ensure the best quality of jpeg image is generated
        # call("D:/Software/ffmpeg-20180226-f4709f1-win64-static/bin/ffmpeg -threads 8 -i " + src + " -qscale:v 2 " + dest)

        # src = os.path.join(compressed_video_folder, filename)
        src = os.path.join(flipped_video_folder, filename)
        cap = cv2.VideoCapture(src)

        #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        #fgbg = cv2.createBackgroundSubtractorKNN()
        fgbg = cv2.createBackgroundSubtractorMOG2()

        count = 0
        success, frame = cap.read()
        # success = Truen

        while(success):
            fgmask = fgbg.apply(frame)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

            if count > 4:
                # Skip the first blank frame (don't know why)
                # Also skip the fist few frames to ensure the background was clearly subtracted
                cv2.imwrite(dest % count, fgmask)

            count += 1

            success, frame = cap.read()

            # cv2.imshow('frame',fgmask)
            # k = cv2.waitKey(30) & 0xff
            # if k == 27:
            #     break

        cap.release()
        # cv2.destroyAllWindows()

    # ===============================================
    # Step 5: (Just to verify)
    #         Combine the subtracted images back to
    #         video. To check the quality
    # ===============================================
    subfolders = os.listdir(subtracted_image_folder)

    for subfolder in subfolders:
        # filename_no_ext = filename.split('.')[0]

        # mkdir one folder for each video's frames
        # strfolder = os.path.join(subtracted_image_folder, filename_no_ext)
        # os.mkdir(strfolder)

        # Folder structure:
        # D:\meteor-monitor\data\fortest\image_compressed_subtracted\M20160915_043641_GDPY01_\M20160915_043641_GDPY01_-0000.jpg
        src_file = os.path.join(subtracted_image_folder, subfolder, subfolder)

        dest_file = os.path.join(subtracted_bg_video_folder, subfolder + '.avi')

        # strCommand = "D:/Software/ffmpeg-20180226-f4709f1-win64-static/bin/ffmpeg"
        strCommand = environment.FFMPEG_COMMAND

        # image file name also start with the "subfolder" name
        strCommand += " -r 25 -f image2 -s %s -start_number 1 -i %s" % (compress_size, src_file)
        strCommand += "-%04d.jpg  -vcodec libx264 -crf 15 -pix_fmt yuv420p "
        strCommand += dest_file

        call(strCommand)


def main():
    # pre_process_method_1('fortest')
    pre_process_method_1('Meteor')
    pre_process_method_1('Others')


if __name__ == "__main__":
    main()
