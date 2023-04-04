from retinaface import RetinaFace
from timeit import default_timer as timer
import cv2
import tensorflow as tf
from collections import deque
from multiprocessing.pool import ThreadPool
import os


def process_frame(frame):
    resp = RetinaFace.detect_faces(frame, allow_upscaling=False)
    if not isinstance(resp, tuple):
        face_key_list = list()
        for key in resp.keys():
            if str(key).startswith('face_'):
                face_key_list.append(str(key))

        for f in face_key_list:
            # Take out the locations of face detected
            left, top, right, bottom = resp[f]['facial_area']

        # adjust the border for a higher blur
        # left = left - 10
        # right = right + 10
        # top = top - 10
        # bottom = bottom + 10
        # if right > width:
        #     right = width
        # if bottom > height:
        #     bottom = height
        # if left < 0: left = 0
        # if top < 0: top = 0

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Blur the area inside the rectangle
        orig = frame[top:bottom, left:right]
        blurred = cv2.blur(orig, (30, 30))
        frame[top:bottom, left:right] = blurred

    return frame


def blur_video(video_file, name):
    # LOAD THE ORIGINAL CLIP
    cap = cv2.VideoCapture(video_file)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    try:
        if not os.path.exists('D:/Tiki/retinaface/output_videos'):
            os.makedirs('D:/Tiki/retinaface/output_videos')
    except OSError:
        print('Error: Creating directory of data')

    output_file = os.path.join("D:/Tiki/retinaface/output_videos", name + '.mp4')
    writer = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    thread_num = cv2.getNumberOfCPUs()
    pool = ThreadPool(processes=thread_num)
    pending_task = deque()

    run_timer = timer()
    frames_processed = 0
    while True:

        # if timer() - run_timer > 1:
        #     print(f'{frames_processed} frames_processed in {timer() - run_timer}')
        #     run_timer = timer()
        #     frames_processed = 0

        # Consume the queue.
        while len(pending_task) > 0 and pending_task[0].ready():
            res = pending_task.popleft().get()
            writer.write(res)
            frames_processed += 1

        # Populate the queue.
        if len(pending_task) < thread_num:
            frame_got, frame = cap.read()
            if frame_got:
                task = pool.apply_async(process_frame, (frame.copy(),))
                pending_task.append(task)

        # Show preview.
        if cv2.waitKey(1) == 27 or not frame_got:
            break

    # release and create a video file as mentioned in path
    writer.release()


if __name__ == "__main__":

    # Run with CPU
    start = timer()
    with tf.device('/CPU:0'):
        source_videos_path = 'D:/Tiki/retinaface/input_videos'

        for file in os.listdir("D:/Tiki/retinaface/input_videos"):
            if file.endswith(".avi") or file.endswith(".mp4"):
                path = os.path.join("D:/Tiki/retinaface/input_videos", file)
                blur_video(path, file)
        print("Time taken in CPU:", timer() - start)
