import math
import numpy as np
import cv2
from PIL import Image

def compute_intrinsic_matrix(width, height, fov_deg):
    """
    Compute pinhole camera intrinsic matrix from image size and horizontal FOV.
    """
    fov_rad = np.deg2rad(fov_deg)
    fx = width / (2 * np.tan(fov_rad / 2))
    fy = fx  # Assume square pixels
    cx = width / 2
    cy = height / 2
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0,  0,  1]
    ], dtype=np.float32)

    return K

def simulate_distortion_from_pinhole(pil_image, K_pinhole, K_real, dist_real):
    """
    Simulate how a real camera would distort a synthetic pinhole image.

    Args:
        pil_image (PIL.Image): Input pinhole image (e.g., from CARLA).
        K_pinhole (np.ndarray): Intrinsic matrix of the synthetic (ideal) camera.
        K_real (np.ndarray): Intrinsic matrix of the real camera.
        dist_real (np.ndarray): Distortion coefficients of the real camera.

    Returns:
        PIL.Image: Simulated distorted image.
    """
    # Convert PIL → OpenCV format
    image = np.array(pil_image.convert('RGB'))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    h, w = image.shape[:2]

    map1, map2 = cv2.initUndistortRectifyMap(
        cameraMatrix=K_pinhole,
        distCoeffs=None,
        R=np.eye(3),
        newCameraMatrix=K_real,
        size=(w, h),
        m1type=cv2.CV_32FC1
    )

    distorted = cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    distorted = cv2.undistort(distorted, K_real, dist_real)

    #distorted_resized = cv2.resize(distorted, (512, 512), interpolation=cv2.INTER_LINEAR)

    # Convert back to PIL
    return Image.fromarray(cv2.cvtColor(distorted, cv2.COLOR_BGR2RGB))



def get_frame_of_video(video_path, frame_index):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, img = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Cannot read frame {frame_index} from video.")
    
    return img

# ---------------- Load N-th frame from video ----------------

def find_checkerboard_corners(nx, ny, gray_img):
    pattern_size = (nx, ny)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray_img, pattern_size, flags=flags)
    if not found:
        raise RuntimeError("Checkerboard not found. Check nx/ny or frame quality.")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
    cv2.cornerSubPix(gray_img, corners, winSize=(11,11), zeroZone=(-1,-1), criteria=criteria)

    return corners

def build_3d_board(nx,ny, square_size):
    objp = np.zeros((nx*ny, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    objp *= square_size
    # center board at origin
    objp[:,0] -= square_size*(nx-1)/2
    objp[:,1] -= square_size*(ny-1)/2
    return objp

def solvePnPCalculation(objp, corners, K, D):
    ok, rvec, tvec = cv2.solvePnP(objp, corners, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise RuntimeError("solvePnP failed.")
    return rvec, tvec

def euler_ypr_from_R_camera(R):
    """
    Convert rotation matrix (board -> camera) to yaw-pitch-roll in camera frame.
    Convention:
        - Pitch: rotation about camera X (horizontal)
        - Yaw: rotation about camera Y (vertical)
        - Roll: rotation about camera Z (forward)
    """
    sy = math.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6

    if not singular:
        pitch = math.atan2(-R[2,1], R[2,2])
        yaw   = math.atan2(R[2,0], sy)
        roll  = math.atan2(R[1,0], R[0,0])
    else:
        pitch = math.atan2(-R[2,1], R[2,2])
        yaw   = math.atan2(R[2,0], sy)
        roll  = 0
    return yaw, pitch, roll
