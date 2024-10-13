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
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult
VisionRunningMode = mp.tasks.vision.RunningMode

class Teleop:
    def __init__(self) -> None:
        self._gesture = None
        self._pose = None
        self.frame = None
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Cannot open camera")
        self.t = Thread(target=self.launch_pose_tracker)
        self.t.start()
        # self.launch_pose_tracker()

    def launch_pose_tracker(self):
        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=c.GESTURE_PATH), 
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.set_gesture)
        with GestureRecognizer.create_from_options(options) as recognizer:
            while True:
                ret, frame = self.cap.read()
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
                self.frame,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                solutions.drawing_styles.get_default_hand_connections_style())
            total_x = (total_x / c.NUM_PTS) # * c.WEBCAM_WIDTH
            total_y = (total_y / c.NUM_PTS) # * c.WEBCAM_HEIGHT
            total_z = total_z / c.NUM_PTS
        self._pose = [total_x, total_z, total_y]
        
        # Retrieve gesture
        gesture = result.gestures[0][0].category_name
        if gesture == "Closed_Fist" or gesture == "Open_Palm":
            self._gesture = gesture  

if __name__ == "__main__":
    client = zerorpc.Client()
    client.connect("tcp://127.0.0.1:4242")
    teleop = Teleop()
    while True:
        if not teleop.frame is None:
            cv2.imshow("test", teleop.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        client.set_ee_pos(teleop.pose)
        client.step()
        print(teleop.pose)
        print(teleop.gesture)

    teleop.t.join()

