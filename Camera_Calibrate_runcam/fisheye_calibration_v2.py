import numpy as np
import cv2 as cv
import glob
import pickle
import os

# ==================================================
# USER CONFIG
# ==================================================
CHESSBOARD_SIZE = (10, 7)     # inner corners (cols, rows)
SQUARE_SIZE_MM = 22           # real square size
FRAME_SIZE = (2560, 1440)     # (width, height)
IMAGE_DIR = "runcam_images_2k"
TEST_IMAGE = "runcam_images_2k/img5.png"

# ==================================================
# TERMINATION CRITERIA
# ==================================================
CRITERIA = (
    cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER,
    100,
    1e-7
)

# ==================================================
# OBJECT POINTS (CONSISTENT ORDER)
# ==================================================
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[
    0:CHESSBOARD_SIZE[0],
    0:CHESSBOARD_SIZE[1]
].T.reshape(-1, 2)
objp *= SQUARE_SIZE_MM
objp = objp.reshape(1, -1, 3)

objpoints = []
imgpoints = []

# ==================================================
# CHESSBOARD DETECTION + FRAME FILTERING
# ==================================================
images = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.png")))
assert len(images) >= 15, "Need at least 15 calibration images"

for fname in images:
    img = cv.imread(fname)
    if img is None:
        continue

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCorners(
        gray,
        CHESSBOARD_SIZE,
        cv.CALIB_CB_ADAPTIVE_THRESH |
        cv.CALIB_CB_NORMALIZE_IMAGE
    )

    if not ret:
        continue

    corners = cv.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        CRITERIA
    )

    # ---------- FRAME QUALITY CHECKS ----------
    x, y, w, h = cv.boundingRect(corners)
    board_area = w * h
    image_area = FRAME_SIZE[0] * FRAME_SIZE[1]

    # Board must be reasonably large
    if board_area < 0.10 * image_area:
        continue

    # Board must not touch image borders
    margin = 25
    if (
        x < margin or y < margin or
        x + w > FRAME_SIZE[0] - margin or
        y + h > FRAME_SIZE[1] - margin
    ):
        continue

    objpoints.append(objp)
    imgpoints.append(corners)

    cv.drawChessboardCorners(img, CHESSBOARD_SIZE, corners, ret)
    cv.imshow("Accepted Frames", img)
    cv.waitKey(120)

cv.destroyAllWindows()

print(f"\nUsing {len(objpoints)} high-quality calibration images")

# ==================================================
# FISHEYE CALIBRATION (STABLE FIRST PASS)
# ==================================================
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = []
tvecs = []

flags = (
    cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC |
    cv.fisheye.CALIB_FIX_SKEW |
    cv.fisheye.CALIB_FIX_PRINCIPAL_POINT |
    cv.fisheye.CALIB_FIX_K3
)

rms, K, D, rvecs, tvecs = cv.fisheye.calibrate(
    objpoints,
    imgpoints,
    FRAME_SIZE,
    K,
    D,
    rvecs,
    tvecs,
    flags=flags,
    criteria=CRITERIA
)

print("\n=== CALIBRATION RESULTS ===")
print(f"RMS reprojection error: {rms:.4f}")
print("Camera matrix K:\n", K)
print("Distortion coefficients D:\n", D)

# ==================================================
# SAVE CALIBRATION
# ==================================================
with open("fisheye_calibration_final_v2.pkl", "wb") as f:
    pickle.dump((K, D), f)

# ==================================================
# UNDISTORT TEST IMAGE
# ==================================================
img = cv.imread(TEST_IMAGE)
h, w = img.shape[:2]

new_K = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(
    K, D, (w, h), np.eye(3), balance=0.85
)

map1, map2 = cv.fisheye.initUndistortRectifyMap(
    K, D, np.eye(3), new_K, (w, h), cv.CV_16SC2
)

undistorted = cv.remap(img, map1, map2, interpolation=cv.INTER_LINEAR)

cv.imwrite("fisheye_undistorted_final_v2.png", undistorted)

cv.imshow("Original", img)
cv.imshow("Undistorted", undistorted)
cv.waitKey(0)
cv.destroyAllWindows()

# ==================================================
# PER-IMAGE REPROJECTION ERROR (DEBUG)
# ==================================================
errors = []
for i in range(len(objpoints)):
    projected, _ = cv.fisheye.projectPoints(
        objpoints[i], rvecs[i], tvecs[i], K, D
    )
    err = cv.norm(
        imgpoints[i].reshape(-1, 2),
        projected.reshape(-1, 2),
        cv.NORM_L2
    ) / len(projected)
    errors.append(err)
    print(i)

print("\nMean reprojection error:", np.mean(errors))
print("Max reprojection error :", np.max(errors))
