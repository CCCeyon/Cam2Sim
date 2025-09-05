#!/usr/bin/env python3
import cv2 as cv
import numpy as np
import math
import os

from utils.argparser import parse_calibration2_args
from utils.distortion import build_3d_board, euler_ypr_from_R_camera, find_checkerboard_corners, get_frame_of_video, solvePnPCalculation
from utils.save_data import get_model_data


# ---------------- Hardcoded calibration ----------------

args = parse_calibration2_args()
model = get_model_data(args.model)
if model is None:
    print(f"Model '{args.model}' not found. Please check the model name or remove the model argument.")
    exit(1)

K = np.array(model["camera"]["calibration"]["K"], dtype=np.float64)
D = np.array(model["camera"]["calibration"]["distortion"], dtype=np.float64)
nx, ny = model["camera"]["calibration"]["checkerboard_size"][0], model["camera"]["calibration"]["checkerboard_size"][1]
square_size = model["camera"]["calibration"]["square_size"]



img = get_frame_of_video(args.video, args.frame)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
corners = find_checkerboard_corners(nx,ny, gray)
objp = build_3d_board(nx,ny, square_size)
rvec, tvec = solvePnPCalculation(objp, corners, K, D)
# ---------------- Build 3D object points (board frame) ----------------


R, _ = cv.Rodrigues(rvec)

# ---------------- Euler angles in CAMERA frame ----------------

yaw, pitch, roll = euler_ypr_from_R_camera(R)
x, y, z = tvec.reshape(3).tolist()

# ---------------- Print results ----------------
# x was distance
# 2800
# 3
# 1.1

#dist_car_and_center = 2.2 using bounding box

#2.8 - 1.1 = 1.7

#camera_height = checkerboard_height + script diff


print("\n=== Camera pose relative to CHECKERBOARD ===")
print(f"x, y, z: {x:.3f}, {y:.3f}, {z:.3f} (same units as square_size)")
print(f"Pitch (deg, rotation around camera X): {math.degrees(pitch):.3f}")
print(f"Yaw   (deg, rotation around camera Y): {math.degrees(yaw):.3f}")
print(f"Roll  (deg, rotation around camera Z): {math.degrees(roll):.3f}")

# ---------------- Visualization ----------------
vis = img.copy()
axis_len = square_size * 3.0
try:
    cv.drawFrameAxes(vis, K, D, rvec, tvec, axis_len)
except Exception:
    # fallback
    axes3d = np.float32([[0,0,0],[axis_len,0,0],[0,axis_len,0],[0,0,axis_len]])
    axes2d, _ = cv.projectPoints(axes3d, rvec, tvec, K, D)
    axes2d = axes2d.reshape(-1,2).astype(int)
    cv.circle(vis, tuple(axes2d[0]), 6, (0,0,0), -1)
    cv.line(vis, tuple(axes2d[0]), tuple(axes2d[1]), (255,0,0), 3)
    cv.line(vis, tuple(axes2d[0]), tuple(axes2d[2]), (0,255,0), 3)
    cv.line(vis, tuple(axes2d[0]), tuple(axes2d[3]), (0,0,255), 3)

#out_path = f"frame{args.frame}_pose.jpg"
#cv.imwrite(out_path, vis)
#print(f"Saved pose visualization: {out_path}")

# ---------------- Optional undistort preview ----------------
#h, w = img.shape[:2]
#newK, _ = cv.getOptimalNewCameraMatrix(K, D, (w,h), alpha=0.0)
#und = cv.undistort(img, K, D, None, newK)
#out_u = f"frame{args.frame}_undist.jpg"
#cv.imwrite(out_u, und)
#print(f"Saved undistorted preview: {out_u}")
