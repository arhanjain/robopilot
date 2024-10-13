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
import zerorpc
from threading import Thread

### Streaming
BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

class Teleop:
    def __init__(self) -> None:
        self._gesture = 1 # 1 for open, 0 for close
        self._pose = None
        self.frame = None
        self.t = Thread(target=self.launch_pose_tracker)
        self.t.start()
        # self.launch_pose_tracker()

    def launch_pose_tracker(self):
        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=c.GESTURE_PATH), 
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.set_gesture)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")


        with GestureRecognizer.create_from_options(options) as recognizer:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to get frame")
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                recognizer.recognize_async(mp_image, int(round(time.time()*1000)))
                # print(f'{total_x:.2f}, {total_y:.2f}, {total_z:.2f}')
                self.frame = frame

    @property
    def pose(self):
        return self._pose
    
    @property
    def gesture(self):
        return self._gesture

    def set_gesture(self, result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
        total_x = 0
        total_y = 0
        total_z = 0
        # Retrieve hand landmarks
        hand_landmarks_list = result.hand_landmarks
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            total_x = hand_landmarks[c.TARGET_POSE_IDX].x
            total_y = hand_landmarks[c.TARGET_POSE_IDX].y
            total_z = hand_landmarks[c.TARGET_POSE_IDX].z
            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in
                hand_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                self.frame,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style())
        self._pose = [total_x, total_z, total_y]
        
        # Retrieve gesture
        if len(result.gestures) == 0:
            return
        gesture = result.gestures[0][0].category_name
        if gesture == "Closed_Fist" or gesture == "Open_Palm":
            self._gesture = 1 if gesture == "Open_Palm" else 0 

if __name__ == "__main__":
    client = zerorpc.Client()
    client.connect("tcp://127.0.0.1:4242")
    teleop = Teleop()
    i = 1000
    while True:
        if not teleop.frame is None:
            cv2.imshow("test", teleop.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # client.set_ee_pos(teleop.pose)
        client.step(teleop.pose, teleop.gesture)
        # print(teleop.pose)
        # print(teleop.gesture)
        i -= 1
        if i == 0: 
            break
    client.reset()
    teleop.t.join()

