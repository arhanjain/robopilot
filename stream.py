import numpy as np
import matplotlib.pyplot as plt
import cv2

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

# Testing
uv1 = np.array([[wc/2],
                [hc/2],
                [1]])
uv2 = np.array([[w/2],
                [h/2],
                [1]])
X = pixelToWorld(uv1, uv2)
print("X")
print(X)
x1 = np.dot(P1[:3],X)
x2 = np.dot(P2[:3],X)
# Again, put in homogeneous form before using them
x1 /= x1[2]
x2 /= x2[2]
print("x1")
print(x1)
print("x2")
print(x2)
