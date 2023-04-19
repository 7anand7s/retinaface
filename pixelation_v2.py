import numpy as np
import pandas as pd
from retinaface import RetinaFace
from timeit import default_timer as timer
import cv2
import tensorflow as tf
import dlib

source_video_path = 'D:/Tiki/retinaface/60fps_trialrun.mp4'


def vconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    w_min = min(im.shape[1] for im in im_list)
    im_list_resize = [cv2.resize(im, (w_min, int(im.shape[0] * w_min / im.shape[1])), interpolation=interpolation)
                      for im in im_list]
    return cv2.vconcat(im_list_resize)

def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)


def integrate_blur(left, right, top, bottom, h, w, frame, factor=30):
    left = left - 10
    right = right + 10
    top = top - 10
    bottom = bottom + 10
    if right > int(w):
        right = int(w)
    if bottom > int(h):
        bottom = int(h)
    if left < 0: left = 0
    if top < 0: top = 0

    cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), 2)

    # Blur the area inside the rectangle
    orig = frame[top:bottom, left:right]
    try:
        blurred = cv2.blur(orig, (factor, factor))
        frame[top:bottom, left:right] = blurred
    except:
        print("An exception occurred in frame ")

    return frame


def blur_and_write(frames_list, coord_list, writer, h, w):
    # faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    for frame in frames_list:
        if len(coord_list):
            for every_coord in coord_list:
                if type(every_coord) is not float:
                    for face in every_coord:
                        left, top, right, bottom = face
                        frame = integrate_blur(left, right, top, bottom, h, w, frame, factor=30)

        # Model 2 on top for increased efficiency
        face_detector = dlib.get_frontal_face_detector()  # face_detector
        faces = face_detector(frame)
        for (x1, y1, x2, y2) in faces:
            frame = integrate_blur(x1, x1 + x2, y1, y1 + y2, h, w, frame, factor=30)

        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(frame)
    return writer


def compare_sets(set1, set2):
    if set1 == set2:
        return 0
    if np.allclose(set1, set2, atol=200):
        return 1
    return 2


def inter_polate(l1, l2, buff_size):
    lists = [[np.nan, np.nan, np.nan, np.nan] for _ in range(buff_size - 1)]
    x = [l1]
    for y in lists:
        x.append(y)
    x.append(l2)
    new_lists = np.transpose(np.array(x)).tolist()
    new_c_lists = []
    for each in new_lists:
        s = pd.Series(each)
        new_c = s.interpolate()
        new_c_lists.append(new_c)
    new_coords = np.array(new_c_lists).transpose().tolist()
    return new_coords[1: -1]


def blur_video(output, factor=30):
    # LOAD THE ORIGINAL CLIP
    cap = cv2.VideoCapture(source_video_path)
    # cap1 = cv2.VideoCapture('D:/Tiki/retinaface/Sample Videos 2023-03-13/CMWN5290.MP4')
    # cap2 = cv2.VideoCapture('D:/Tiki/retinaface/Sample Videos 2023-03-13/DITF7989.MP4')
    # cap3 = cv2.VideoCapture('D:/Tiki/retinaface/Sample Videos 2023-03-13/GSUN5783.MP4')
    # cap4 = cv2.VideoCapture('D:/Tiki/retinaface/Sample Videos 2023-03-13/MNPY6056.MP4')
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    print("FPS, W, H, D", fps, width, height, duration)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(output, fourcc, int(fps), (width, height))

    # progress bar
    # run_timer = timer()
    # faces_frames_detected = 0
    no_of_frames = 0
    frame_buffer = []
    coord_buffer = []
    pred_coord_b = []
    buff_size = 3

    while True:
        # Grab a single frame of video
        ret, frame = cap.read()
        # Grab all frames of diff video streams
        # ret1, frame1 = cap1.read()
        # ret2, frame2 = cap2.read()
        # ret3, frame3 = cap3.read()
        # ret4, frame4 = cap4.read()

        # When the video is over, break out of the loop
        # if not ret1 or not ret2 or not ret3 or not ret4:
        #     break

        # When the video is over, break out of the loop
        if not ret:
            break

        # # How many frames are processed each second
        # if timer() - run_timer > 1:
        #     faces_frames_detected = no_of_frames - faces_frames_detected
        #     print(f'{faces_frames_detected} Frames detected & blurred in {timer() - run_timer} seconds.')
        #     run_timer = timer()
        #     faces_frames_detected = no_of_frames

        # Resize frames
        # frame1 = frame1[175:1280, 545:1750]  # x0 545 y0 175 x1 1750 y1 1280
        # frame2 = frame2[175:1000, 0:1725]  # x0 0 y0 175 x1 1725 y1 1280
        # frame3 = frame3[175:1280, 250:1870]  # x0 250 y0 175 x1 1870 y1 1280
        # frame4 = frame4[175:1280, 0:1775]  # x0 0 y0 175 x1 1775 y1 1280
        #
        # im_v1 = hconcat_resize_min([frame1, frame2])
        # im_v2 = hconcat_resize_min([frame3, frame4])
        # frame = vconcat_resize_min([im_v1, im_v2])
        #
        # new_height, new_width = frame.shape[:2]
        new_width = int(width)
        new_height = int(height)

        frame_buffer.append(frame)
        x = 0

        # Recognize faces on selected frames
        if no_of_frames % buff_size == 0:
            coord = RetinaFace.detect_faces(frame_buffer[x], allow_upscaling=False)
            # print(type(coord), coord)
            coord_buffer.append(coord)
        else:
            coord_buffer.append(tuple())
        # print(coord_buffer)

        no_of_frames += 1

        # Extract locations if faces were located
        # if type(coord_buffer[x]) is not float:
        if not isinstance(coord_buffer[x], tuple):
            # print(coord_buffer)
            temp = []
            for key in coord_buffer[x].keys():
                if str(key).startswith('face_'):
                    # temp.append({str(key): coord_buffer[x][str(key)]['facial_area']})
                    temp.append(coord_buffer[x][str(key)]['facial_area'])
            pred_coord_b.append(temp)

        else:
            pred_coord_b.append(np.nan)

        # Interpolate locations with the existing locations
        if len(frame_buffer) > 9:
            # print(len(frame_buffer), len(pred_coord_b), len(coord_buffer))
            n = 0
            while True:
                m = n
                counter= counter+1
                if n > 5:
                    writer = blur_and_write(frame_buffer[0:6], pred_coord_b[0:6], writer, new_height, new_width)
                    del frame_buffer[0:6]
                    del coord_buffer[0:6]
                    del pred_coord_b[0:6]
                    break
                # print(n)
                list1 = pred_coord_b[n]
                list2 = pred_coord_b[n + buff_size]
                n = n + buff_size
                if type(list1) is float:
                    continue
                for each_c1 in list1:
                    if type(list2) is float:
                        print('copying')
                        for a in range(buff_size-1):
                            temp = n + 1
                            if pred_coord_b[temp] is not float:
                                pred_coord_b[temp] = [each_c1]
                            else:
                                pred_coord_b[temp].append(each_c1)
                        continue
                    for each_c2 in list2:
                        if compare_sets(each_c1, each_c2) == 0:
                            print('Copying')
                            for a in range(buff_size-1):
                                temp = m + 1
                                if pred_coord_b[temp] is not float:
                                    pred_coord_b[temp] = [each_c1]
                                else:
                                    pred_coord_b[temp].append(each_c1)
                            break
                        elif compare_sets(each_c1, each_c2) == 1:
                            print('Interpolating...')
                            coordinates = inter_polate(each_c1, each_c2, buff_size)
                            for a in coordinates:
                                temp = m + 1
                                a = list(map(int, a))
                                print(a)
                                if pred_coord_b[temp] is not float:
                                    pred_coord_b[temp] = [a]
                                else:
                                    pred_coord_b[temp].append(a)
                            break
                    print('copying')
                    for a in range(buff_size-1):
                        temp = m + 1
                        if pred_coord_b[temp] is not float:
                            pred_coord_b[temp] = [each_c1]
                        else:
                            pred_coord_b[temp].append(each_c1)

    # release and create a video file as mentioned in path
    writer.release()


if __name__ == "__main__":
    start = timer()
    # List of available GPUs
    GPUs = tf.config.list_physical_devices('GPU')
    print("GPUs Available: ", GPUs)
    # Make sure the GPU you want to force it into --> is chosen below
    with tf.device('/GPU:0'):
        blur_video('blurred_video_interp4_2.mp4')
        print("Time taken in GPU:", timer() - start)
