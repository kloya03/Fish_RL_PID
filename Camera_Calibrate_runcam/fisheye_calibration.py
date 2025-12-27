import numpy as np
import cv2 as cv
import glob
import pickle
import os

#################### FIND CHESSBOARD CORNERS ##############################

chessboardSize = (10,7)
# frameSize = (1280, 720)
# frameSize = (1920, 1080)
frameSize = (2560,1440)
# Termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.1)

# Prepare object points
objp = np.zeros((1, chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[0, :, :2] = np.mgrid[0:chessboardSize[0], 0:chessboardSize[1]].T.reshape(-1, 2)

size_of_chessboard_squares_mm = 22  # Adjust this to the real size of your squares
objp = objp * size_of_chessboard_squares_mm

# Arrays to store points
objpoints = []  # 3D points
imgpoints = []  # 2D points

# Load images
images = glob.glob(os.path.expanduser('runcam_images_2k/*.png'))
if not images:
    print("No images found. Please check the directory path.")
else:
    for image_path in images:
        img = cv.imread(image_path)
        print(f"Processing: {image_path}")
        if img is None:
            print(f"Unable to read image: {image_path}")
            continue
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

        if ret:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display corners
            cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
            cv.imshow('img', img)
            cv.waitKey(500)

    cv.destroyAllWindows()

############## FISHEYE CALIBRATION ########################################

K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = []
tvecs = []

# Fisheye calibration
ret, K, D, rvecs, tvecs = cv.fisheye.calibrate(
    objpoints,
    imgpoints,
    frameSize,
    K,
    D,
    rvecs,
    tvecs,
    flags=cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv.fisheye.CALIB_FIX_SKEW
)

# Save calibration data
with open("fisheye_calibration_runcam_2k_22.pkl", "wb") as f:
    pickle.dump((K, D), f)

################ UNDISTORTION #############################################

img = cv.imread('runcam_images_2k/img5.png')
h, w = img.shape[:2]
new_K = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (w, h), np.eye(3), balance=1)

map1, map2 = cv.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (w, h), cv.CV_16SC2)
undistorted_img = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR)

# Save undistorted image
cv.imwrite("fisheye_undistorted_2k_img5.png", undistorted_img)

# Display images
# cv.imshow("Original", img)
# cv.imshow("Undistorted", undistorted_img)
# cv.waitKey(0)
# cv.destroyAllWindows()

################ REPROJECTION ERROR #######################################

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.fisheye.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
    # Reshape to ensure both arrays have the same shape
    imgpoints2 = np.squeeze(imgpoints2)  # Shape becomes (70, 2)
    imgpoints[i] = np.squeeze(imgpoints[i])  # Shape becomes (70, 2)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error

print(f"Total reprojection error: {mean_error / len(objpoints)}")
