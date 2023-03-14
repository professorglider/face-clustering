import cv2
import numpy as np
import face_recognition as fr
from detector import Detector
import time
from datetime import datetime
import uuid
import pandas as pd

RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
SIMILARITY_THRESHOLD = 50
FONT = cv2.FONT_HERSHEY_COMPLEX
RESIZE_K = 1

origin_path = "/home/viktor/Documents/me.jpg"
data_path = "/home/viktor/Documents/faces/data"
photo_path = "/home/viktor/Documents/faces/photos"
cropped_path = "/home/viktor/Documents/faces/cropped"
# origin_path = "/home/viktor/me-three.jpg"
origin = cv2.imread(origin_path)

origin_faces = fr.face_locations(origin)
if len(origin_faces) != 1:
    print("Origin should have exactly one face; found = " + str(len(origin_faces)))
    exit(1)

origin_encoding = fr.face_encodings(origin, origin_faces)[0]

capture = cv2.VideoCapture(0)

detector = Detector(face_min_height=10)

# used to record the time when we processed last frame
prev_frame_time = 0

# used to record the time at which we processed current frame
new_frame_time = 0

data = []


def print_faces(faces):
    for face in sorted(faces, key=lambda f: (f.bounding_box[0], f.bounding_box[1])):
        print(face.is_clear(), face.blur, face.angles, face.keypoints)


session_id = uuid.uuid4()
session_start = datetime.now()
name = input("Enter your name: ")

while capture.isOpened():
    ret, frame = capture.read()
    dt = datetime.now()
    filename = f'{session_id}_{dt}'.format(dt=dt, session_id=session_id)
    frame_path = f"{photo_path}/{filename}.jpeg"
    cropped_face_path = f"{cropped_path}/{filename}.jpeg"
    cv2.imwrite(frame_path, frame)
    image = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (0, 0), None, 1/RESIZE_K, 1/RESIZE_K)
    # image = origin

    # ksize = (10, 10)
    #
    # # Using cv2.blur() method
    # image = cv2.blur(image, ksize)
    faces = detector.detect(image, mode="bgr", angle_estimation=True, crop_faces=True)

    print_faces(faces)

    for face in faces:
        print(face)
        datum = vars(face)
        datum['dt'] = dt
        datum['session_start'] = session_start
        datum['name'] = name
        datum['session_id'] = session_id
        datum['frame_path'] = frame_path
        data.append(datum)

        x1, y1, x2, y2 = face.bounding_box
        cv2.imwrite(cropped_face_path, frame[y1:y2, x1:x2])
        datum['cropped_path'] = cropped_face_path

        face_encoding = fr.face_encodings(image, [[y1, x2, y2, x1]])
        datum['face_encoding'] = face_encoding
        [x1, y1, x2, y2] = (np.asarray([x1, y1, x2, y2]) * RESIZE_K).astype(int)
        distances = fr.face_distance([face_encoding], origin_encoding)[0]
        e_distance = np.sqrt(np.sum(np.square(distances)))
        match_percentage = (1 - e_distance) * 100
        print(match_percentage)

        color = GREEN if SIMILARITY_THRESHOLD < match_percentage else RED
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)

        for point in face.keypoints.values():
            cv2.circle(frame, tuple((point * RESIZE_K).astype(int)), 2, BLUE, 2)

        cv2.putText(frame, str(round(match_percentage, 2)) + "%", (x2 + 6, y1 - 6), FONT, 1, color,
                    1)

        cv2.putText(frame, "Yaw: " + str(round(face.angles["yaw"])), (x2 + 6, y1 + 30), FONT, 1,
                    BLUE, 1)
        cv2.putText(frame, "Roll: " + str(round(face.angles["roll"])), (x2 + 6, y1 + 90), FONT, 1,
                    BLUE, 1)
        cv2.putText(frame, "Pitch: " + str(round(face.angles["pitch"])), (x2 + 6, y1 + 60), FONT, 1,
                    BLUE, 1)

    # if cv2.waitKey(100) & 0xFF == ord('q'):
        filename = f'{session_id}_{session_start}'.format(session_start=session_start, session_id=session_id)
        path = f"{data_path}/{filename}.csv"
        pd.DataFrame(data).drop(columns=["image"]).to_csv(path, index=False)
        break


    # time when we finish processing for this frame
    new_frame_time = time.time()

    # Calculating the fps
    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # converting the fps into integer
    fps = str(round(fps, 2))

    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)

    # putting the FPS count on the frame
    cv2.putText(frame, fps, (7, 70), FONT, 3, (100, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow('frame', frame)