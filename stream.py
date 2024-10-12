import time

import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
import cv2

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = "./hand_landmarker.task"

# Orient y down, x right, z inwards
fc = 500
hc = 720
wc = 1280
P1 = np.array([[fc, 0, wc/2],
               [0, fc, hc/2],
               [0, 0, 1]])
P1 = P1 @ np.eye(4)[:3]
# iphone: 4000 x 3000 pixels, 13684.2105263
f = 13684.2105263
h = 3000
w = 4000
K2 = np.array([[f, 0, w/2],
               [0, f, h/2],
               [0, 0, 1],])
x_trans = 500 # mm
z_trans = 500
R2 = np.array([[0, 0, -1, -x_trans],
               [0, 1, 0, 0],
               [1, 0, 0, -z_trans]])
P2 = K2 @ R2
def pixelToWorld(uv1, uv2):
    # input should be homogenous
    # normalized?
    uv1 = np.expand_dims(uv1, axis=1)
    uv2 = np.expand_dims(uv2, axis=1)
    X = cv2.triangulatePoints(P1, P2, uv1[:2], uv2[:2]).astype(np.float64)
    X /= X[3]
    return X

### Streaming
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode

def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    print('hand landmarker result: {}'.format(result))
    
options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='/path/to/model.task'),
    running_mode=VisionRunningMode.LIVE_STREAM,
    result_callback=print_result)
cap = cv2.VideoCapture()
if not cap.isOpened():
    print("Cannot open camera")
with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to get frame")
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        landmarker.detect_async(mp_image, int(round(time.time()*1000)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()