import numpy as np
import cv2

import constants as c

P1 = np.array([[c.WEBCAM_FOCAL, 0, c.WEBCAM_WIDTH/2],
               [0, c.WEBCAM_FOCAL, c.WEBCAM_HEIGHT/2],
               [0, 0, 1]])
P1 = P1 @ np.eye(4)[:3]
K2 = np.array([[c.EXT_FOCAL, 0, c.EXT_WIDTH/2],
               [0, c.EXT_FOCAL, c.EXT_HEIGHT/2],
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