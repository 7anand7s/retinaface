from retinaface import RetinaFace
from timeit import default_timer as timer
import cv2
import tensorflow as tf

source_video_path = 'D:/Tiki/retinaface/60fps_trialrun.mp4'


def process_face(frame, resp, face_key_list, width, height, factor, no_of_frames):
    for key in resp.keys():
        if str(key).startswith('face_'):
            face_key_list.append(str(key))

    for f in face_key_list:

        # Take out the locations of face detected
        left, top, right, bottom = resp[f]['facial_area']

        # adjust the border for a higher blur
        left = left - 10
        right = right + 10
        top = top - 10
        bottom = bottom + 10
        if right > width:
            right = width
        if bottom > height:
            bottom = height
        if left < 0: left = 0
        if top < 0: top = 0

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Blur the area inside the rectangle
        orig = frame[top:bottom, left:right]
        try:
            blurred = cv2.blur(orig, (factor, factor))
            frame[top:bottom, left:right] = blurred
        except:
            print("An exception occurred in frame ", no_of_frames)


def blur_video(output, factor):
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

        # Resize frame
        frame = frame[0:400, 0:800]
        new_width = 800
        new_height = 400

        # Concatenate 4 Frames : 2 vertically & 2 horizontally
        im_v = cv2.vconcat([frame, frame])
        frame = cv2.hconcat([im_v, im_v])

        # Resize to square (add white background instead of stretching and shrinking)
        # Needs to be implemented when using 4 different videos

        # Resize further for faster results
        # frame = frame[0:256, 0:256]

        no_of_frames += 1

        # interpolation
        if no_of_frames % 2:
            if not previous_blur:
                writer.write(frame)
                continue
            else:
                process_face(frame, resp, face_key_list, new_width, new_height, factor, no_of_frames)

                # write every frame back as a video
                writer.write(frame)
                previous_blur = False
                continue

        # Find all the faces in the current frame of video & it's face_locations
        resp = RetinaFace.detect_faces(frame, allow_upscaling=False)

        if not isinstance(resp, tuple):
            previous_blur = True
            face_key_list = list()
            process_face(frame, resp, face_key_list, new_width, new_height, factor, no_of_frames)

        # Resize back to desired Output size 1080x1920 or 4k
        # Irregular rescale here(shrink/stretch), have to be upgraded depending on camera coordinates
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

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
    with tf.device('/GPU:0'):
        blur_video('blurred_video1_0.mp4', 20)
        print("Time taken in GPU:", timer() - start)
