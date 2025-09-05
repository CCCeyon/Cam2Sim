# This script calibrates a camera using a checkerboard pattern from a video file.
# Credit to Stefano Carlo Lambertenghi for the original code
# This code is adapted to work with this project's structure and requirements.

import cv2
import numpy as np
import matplotlib.pyplot as plt

from config import CHECKERBOARD_ROWS, CHECKERBOARD_COLS, CHECKERBOARD_MIN_FRAMES
from utils.argparser import parse_calibration_args
from utils.save_data import get_model_data, save_camera_calibration

# ==== Configuration ====
args = parse_calibration_args()

checkerboard_size = (CHECKERBOARD_COLS, CHECKERBOARD_ROWS)  # inner corners (columns, rows)
square_size = 24.0  # mm or your unit
min_frames = 15

# ==== Calibration Data Containers ====
objpoints = []
imgpoints = []

model = None
if args.model is not None:
    model = get_model_data(args.model)
    if model is None:
        print(f"Model '{args.model}' not found. Please check the model name or remove the model argument.")
        exit(1)

# 3D object points template
objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:checkerboard_size[0], 0:checkerboard_size[1]].T.reshape(-1, 2)
objp *= square_size

cap = cv2.VideoCapture(args.video)
if not cap.isOpened():
    print(f"Error: Could not open video file {args.video}. Please check the path and file format.")
    exit(1)
frame_count = 0
found_count = 0

print("Starting checkerboard detection...\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    print(f"Processing frame {frame_count}...", end=' ')

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

    if ret:
        found_count += 1
        print(f"Checkerboard detected! Total: {found_count}")

        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        imgpoints.append(corners2)

        # Draw and display
        frame_drawn = frame.copy()
        cv2.drawChessboardCorners(frame_drawn, checkerboard_size, corners2, ret)
        frame_rgb = cv2.cvtColor(frame_drawn, cv2.COLOR_BGR2RGB)

        if args.plot:
            plt.figure(figsize=(6, 6))
            plt.title(f"Frame {frame_count}: Checkerboard Detected")
            plt.axis('off')
            plt.imshow(frame_rgb)
            plt.show()

    else:
        print("Checkerboard NOT found.")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Early exit requested.")
        break

cap.release()
cv2.destroyAllWindows()

# ==== Calibration ====
print(f"\nFinished processing {frame_count} frames.")
print(f"Valid calibration frames collected: {found_count}")

if found_count >= CHECKERBOARD_MIN_FRAMES:
    print("\nRunning calibration...")
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("\n=== Camera Calibration Results ===")
    print("Intrinsic Matrix (K):\n", K)
    print("\nDistortion Coefficients:\n", dist.ravel())

    if model is not None:
        print("\nSaving calibration data to model...")

        save_camera_calibration(args.model, {
            "K": K.tolist(),
            "distortion": dist.ravel().tolist(),
            "checkerboard_size": checkerboard_size,
            "square_size": square_size
        })

        print("Calibration data saved successfully.")

else:
    print(f"\nNot enough valid frames for calibration. Need at least {min_frames}. Got {found_count}.")