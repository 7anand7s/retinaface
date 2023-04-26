from retinaface import RetinaFace
from timeit import default_timer as timer
import cv2
import tensorflow as tf
import numpy as np
import dlib
import pandas as pd

source_video_path = 'D:/Tiki/retinaface/60fps_trialrun.mp4'

def convert_and_trim_bb(image, rect):
    # extract the starting and ending (x, y)-coordinates of the
    # bounding box
    startX = rect.left()
    startY = rect.top()
    endX = rect.right()
    endY = rect.bottom()
    # ensure the bounding box coordinates fall within the spatial
    # dimensions of the image
    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(endX, image.shape[1])
    endY = min(endY, image.shape[0])
    # compute the width and height of the bounding box
    w = endX - startX
    h = endY - startY
    # return our bounding box coordinates
    return startX, startY, w, h


def dlib_detection(frame):
    # Model 2 on top for increased efficiency
    face_detector = dlib.get_frontal_face_detector()  # face_detector
    rects = face_detector(frame)
    faces = [convert_and_trim_bb(frame, r) for r in rects]
    if len(faces):
        for each_face in faces:
            x1, y1, x2, y2 = list(each_face)
            frame = integrate_blur(x1, x1 + x2, y1, y1 + y2, frame, factor=30)

    return frame


def integrate_blur(left, right, top, bottom, frame, factor=30):
    left = left - 10
    right = right + 10
    top = top - 10
    bottom = bottom + 10
    right = min(right, frame.shape[1])
    bottom = min(bottom, frame.shape[0])
    left = max(0, left)
    top = max(0, top)

    cv2.rectangle(frame, (left, top), (right, bottom), (255, 255, 255), 2)

    # Blur the area inside the rectangle
    orig = frame[top:bottom, left:right]
    try:
        blurred = cv2.blur(orig, (factor, factor))
        frame[top:bottom, left:right] = blurred
    except:
        print("An exception occurred in frame ")

    return frame


def blur_and_write(frames_list, coord_list, writer, h, w, add_dlib):
    for frame in frames_list:
        if len(coord_list):
            for every_coord in coord_list:
                if type(every_coord) is not float:
                    for face in every_coord:
                        left, top, right, bottom = face
                        frame = integrate_blur(left, right, top, bottom, frame, factor=30)

        if add_dlib:
            # Model 2 on top for increased efficiency
            frame = dlib_detection(frame)

        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        writer.write(frame)
    return writer


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


def blur_video(output, factor, method=None, upscaling=False, add_dlib=False):
    # LOAD THE ORIGINAL CLIP
    cap = cv2.VideoCapture(source_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    print("FPS, W, H, D", fps, width, height, duration)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(output, fourcc, int(fps), (width, height))

    # progress bar
    run_timer = timer()
    faces_frames_detected = 0
    no_of_frames = 0
    frame_buffer = []
    coord_buffer = []
    pred_coord_b = []
    buff_size = 2
    counter = 0
    previous_blur = False

    while True:
        # Grab a single frame of video
        ret, frame = cap.read()

        # When the video is over, break out of the loop
        if not ret:
            break

        # How many frames are processed each second
        if timer() - run_timer > 1:
            faces_frames_detected = no_of_frames - faces_frames_detected
            print(f'{faces_frames_detected} Frames detected & blurred in {timer() - run_timer} seconds.')
            run_timer = timer()
            faces_frames_detected = no_of_frames

        no_of_frames += 1

        if method == 'sampling':
            # interpolation
            if no_of_frames % 2:
                if previous_blur:
                    for key in resp.keys():
                        if str(key).startswith('face_'):
                            face_key_list.append(str(key))

                    for f in face_key_list:
                        # Take out the locations of face detected
                        left, top, right, bottom = resp[f]['facial_area']
                        frame = integrate_blur(left, right, top, bottom, frame, factor=30)
                    # write every frame back as a video
                    previous_blur = False

                if add_dlib:
                    # Model 2 on top for increased efficiency
                    frame = dlib_detection(frame)

                writer.write(frame)

            # Find all the faces in the current frame of video & it's face_locations
            resp = RetinaFace.detect_faces(frame, allow_upscaling=upscaling)

            if not isinstance(resp, tuple):
                previous_blur = True
                face_key_list = list()
                for key in resp.keys():
                    if str(key).startswith('face_'):
                        face_key_list.append(str(key))

                for f in face_key_list:
                    # Take out the locations of face detected
                    left, top, right, bottom = resp[f]['facial_area']
                    frame = integrate_blur(left, right, top, bottom, frame, factor=30)

            if add_dlib:
                # Model 2 on top for increased efficiency
                frame = dlib_detection(frame)

            # Resize back to desired Output size 1080x1920 or 4k
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

            # write every frame back as a video
            writer.write(frame)

        elif method == 'interpolation':
            frame_buffer.append(frame)
            x = 0

            # Recognize faces on selected frames
            if no_of_frames % buff_size == 0:
                coord = RetinaFace.detect_faces(frame_buffer[x], allow_upscaling=upscaling)
                coord_buffer.append(coord)
            else:
                coord_buffer.append(tuple())

            no_of_frames += 1

            # Extract locations if faces were located
            if not isinstance(coord_buffer[x], tuple):
                # print(coord_buffer)
                temp = []
                for key in coord_buffer[x].keys():
                    if str(key).startswith('face_'):
                        temp.append(coord_buffer[x][str(key)]['facial_area'])
                pred_coord_b.append(temp)

            else:
                pred_coord_b.append(np.nan)

            # Interpolate locations with the existing locations
            if len(frame_buffer) > 9:
                n = 0
                while True:
                    m = n
                    counter = counter + 1
                    if n > 5:
                        writer = blur_and_write(frame_buffer[0:6], pred_coord_b[0:6], writer, height, width, add_dlib)
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
                            for a in range(buff_size - 1):
                                temp = n + 1
                                if pred_coord_b[temp] is not float:
                                    pred_coord_b[temp] = [each_c1]
                                else:
                                    pred_coord_b[temp].append(each_c1)
                            continue
                        for each_c2 in list2:
                            if compare_sets(each_c1, each_c2) == 0:
                                for a in range(buff_size - 1):
                                    temp = m + 1
                                    if pred_coord_b[temp] is not float:
                                        pred_coord_b[temp] = [each_c1]
                                    else:
                                        pred_coord_b[temp].append(each_c1)
                                break
                            elif compare_sets(each_c1, each_c2) == 1:
                                # print('Interpolating...')
                                coordinates = inter_polate(each_c1, each_c2, buff_size)
                                for a in coordinates:
                                    temp = m + 1
                                    a = list(map(int, a))
                                    if pred_coord_b[temp] is not float:
                                        pred_coord_b[temp] = [a]
                                    else:
                                        pred_coord_b[temp].append(a)
                                break

                        for a in range(buff_size - 1):
                            temp = m + 1
                            if pred_coord_b[temp] is not float:
                                pred_coord_b[temp] = [each_c1]
                            else:
                                pred_coord_b[temp].append(each_c1)

        else:
            resp = RetinaFace.detect_faces(frame, allow_upscaling=upscaling)

            if not isinstance(resp, tuple):
                # Take out the locations of face detected
                left, top, right, bottom = resp['face_1']['facial_area']

                frame = integrate_blur(left, right, top, bottom, frame, factor=30)
                
                if add_dlib:
                    # Model 2 on top for increased efficiency
                    frame = dlib_detection(frame)
                
            writer.write(frame)

    # release and create a video file as mentioned in path
    writer.release()


if __name__ == "__main__":
    start = timer()
    # List of available GPUs
    GPUs = tf.config.list_physical_devices('GPU')
    print("GPUs Available: ", GPUs)
    # Make sure the GPU you want to force it into --> is chosen below
    with tf.device('/GPU:0'):
        blur_video('Final_v1.mp4', 30, method='sampling', upscaling=True, add_dlib=False)
        # Method attribute can be sampling, interpolation or None(default)
        # dlib is the second model on top, can be enabled or disabled (add_dlib)
        print("Time taken in GPU:", timer() - start)
