import time

import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
import cv2

import constants as c

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

### Streaming
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

RESULT = None
def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global RESULT
    RESULT = result
    # print('hand landmarker result: {}'.format(result))
    
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=c.MODEL_PATH),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to get frame")
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        landmarker.detect_async(mp_image, int(round(time.time()*1000)))
        total_x = 0
        total_y = 0
        total_z = 0
        if type(RESULT) is not type(None):
            hand_landmarks_list = RESULT.hand_landmarks
            for idx in range(len(hand_landmarks_list)):
                hand_landmarks = hand_landmarks_list[idx]
                for landmark in hand_landmarks:
                    total_x += landmark.x
                    total_y += landmark.y
                    total_z += landmark.z
                # Draw the hand landmarks.
                hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
                hand_landmarks_proto.landmark.extend([
                    landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in
                    hand_landmarks
                ])
                solutions.drawing_utils.draw_landmarks(
                    frame,
                    hand_landmarks_proto,
                    solutions.hands.HAND_CONNECTIONS,
                    solutions.drawing_styles.get_default_hand_landmarks_style(),
                    solutions.drawing_styles.get_default_hand_connections_style())
            total_x = (total_x / c.NUM_PTS) * c.WEBCAM_WIDTH
            total_y = (total_y / c.NUM_PTS) * c.WEBCAM_HEIGHT
            total_z = total_z / c.NUM_PTS
        else:
            print('else')
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        print(f'{total_x:.2f}, {total_y:.2f}, {total_z:.2f}')
    cap.release()
    cv2.destroyAllWindows()