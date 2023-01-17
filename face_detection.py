from retinaface import RetinaFace
from timeit import default_timer as timer
import numpy as np
import cv2
import tensorflow as tf

# Check the number of available GPUs
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

source_video_path = './in/sample.mp4'

def blur_video():
    # LOAD THE ORIGINAL CLIP
    cap = cv2.VideoCapture(source_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter('blurred_video2.0.mp4', fourcc, fps, (width, height))
    counter = 0
    start = timer()
    while True:
        # Grab a single frame of video
        ret, frame = cap.read()
        counter +=1

        # When the video is over, break out of the loop
        if frame is None:
            break

        # Display the original image
        # cv2.imshow("Faces found1", frame)
        # cv2.waitKey(1)

        # Find all the faces in the current frame of video
        # face_locations
        resp = RetinaFace.detect_faces(frame)
        if counter % 10 == 0:
                end = timer()
                diff = end - start
                fps = diff / 10.0
                print(f"FPS: {fps}")
                start = timer()
        if not isinstance(resp, tuple):
            # Take out the locations of face detected
            left, top, right, bottom = resp['face_1']['facial_area']

            # adjust the border for a higher blur
            left = left - 10
            if left < 0:
                left = 0
            right = right + 10
            if right > int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)):
                right = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            top = top - 10
            if top < 0:
                top = 0
            bottom = bottom + 10
            if bottom > int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)):
                bottom = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Blur the area inside the rectangle
            orig = frame[top:bottom, left:right]
            try:    
                blurred = cv2.blur(orig, (30, 30))
                frame[top:bottom, left:right] = blurred
            except cv2.error:
                pass

            # Display the resulting image
            # cv2.imshow("Faces found2", frame)
            # cv2.waitKey(1)

        # write every frame back as a video
        writer.write(frame)

    # release and create a video file as mentioned in path
    writer.release()


if __name__ == "__main__":

    start = timer()
    # List of available GPUs
    GPUs = tf.config.list_physical_devices('GPU')
    print("GPUs Available: ", GPUs)
    # Make sure the GPU you want to force it into --> is chosen below
    if len(GPUs) > 0:
        print("Run GPU Processing")
        with tf.device('/GPU:0'):
            blur_video()
            print("Time taken in GPU:", timer() - start)
    else:
        print("No GPU available")

    # Run with CPU
    start = timer()
    print("Run CPU Processing")
    with tf.device('/CPU:0'):
        blur_video()
        print("Time taken in CPU:", timer() - start)
