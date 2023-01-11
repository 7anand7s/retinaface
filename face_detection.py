from retinaface import RetinaFace
from timeit import default_timer as timer
import numpy as np
import cv2

source_video_path = 'C:/Users/77ana/Videos/VideoProc_Vlogger/Output/TikiSample.mp4'

def blur_video():
    # LOAD THE ORIGINAL CLIP
    cap = cv2.VideoCapture(source_video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter('blurred_video2.0.mp4', fourcc, fps, (width, height))

    while True:
        # Grab a single frame of video
        ret, frame = cap.read()

        # When the video is over, break out of the loop
        if frame is None:
            break

        # Display the original image
        # cv2.imshow("Faces found1", frame)
        # cv2.waitKey(1)

        # Find all the faces in the current frame of video
        # face_locations
        resp = RetinaFace.detect_faces(frame)
        if not isinstance(resp, tuple):

            # Take out the locations of face detected
            left, top, right, bottom = resp['face_1']['facial_area']

            # adjust the border for a higher blur
            left = left - 10
            right = right + 10
            top = top - 10
            bottom = bottom + 10

            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Blur the area inside the rectangle
            orig = frame[top:bottom, left:right]
            blurred = cv2.blur(orig, (30, 30))
            frame[top:bottom, left:right] = blurred

            # Display the resulting image
            # cv2.imshow("Faces found2", frame)
            # cv2.waitKey(1)

        # write every frame back as a video
        writer.write(frame)

    # release and create a video file as mentioned in path
    writer.release()


if __name__ == "__main__":

    start = timer()
    blur_video()
    print("Time taken in seconds:", timer() - start)
