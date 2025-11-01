# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 16:50:47 2024

@author: kloya
"""
import cv2
import numpy as np
import time
import pickle

def nothing(x):
    pass

cap = cv2.VideoCapture(0)
width = 2560
height = 1440
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FPS, 30)  # Set fps to 1
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0

cv2.namedWindow('Trackbars')
cv2.moveWindow('Trackbars', 1400, 100)
cv2.resizeWindow('Trackbars',350, 250)  # Set the desired size (width, height)

cv2.createTrackbar('h_Low', 'Trackbars', 3, 255, nothing)
cv2.createTrackbar('h_Hi', 'Trackbars', 45, 255, nothing)

cv2.createTrackbar('satLow', 'Trackbars', 140, 255, nothing)
cv2.createTrackbar('satHigh', 'Trackbars', 255, 255, nothing)

cv2.createTrackbar('valLow', 'Trackbars', 150, 255, nothing)
cv2.createTrackbar('valHigh', 'Trackbars', 245, 255, nothing)

# Load the camera fisheye calibration data
file_path = "fisheye_calibration_runcam_2k_25.pkl"
with open(file_path, "rb") as file:  # "rb" means read in binary mode
    data = pickle.load(file)
camera_matrix, distortion_coeffs = data

# Print the variables to verify
print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", distortion_coeffs)

# new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (width, height), 1, (width, height))
new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify\
    (camera_matrix, distortion_coeffs, (width, height), np.eye(3), balance=1)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(camera_matrix, distortion_coeffs,\
            np.eye(3), new_camera_matrix, (width, height), cv2.CV_16SC2)
  
# Assume the real-world size of the object (in meters)
object_size = np.array([0.2, 0.1])   #  in m (w, h)
# Define positions
# positions = ['left', 'center', 'right']
calibration_data = {}

# for position in positions:
#     print(f"Calibrating for {position} position...")
sum_pixels_per_m = 0
sum_pixel_size = np.array([0,0])
pixels_per_m_f = 100 # final pixels/m
err = 1
frame_initialize = 1050
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    frame = cv2.flip(frame, 0)
    frame  = cv2.flip(frame, 1)
    # cv2.imshow("frame", frame)
    # Undistort the frame
    # undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
    undistorted_frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
    # Convert to HSV
    hsv = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2HSV)
    # Make the specified region black based on the ROI
    # hsv[:110, :] = 0  # Top boundary
    # hsv[-110:, :] = 0  # Bottom boundary
    # hsv[:, :110] = 0  # Left boundary
    # hsv[:, -100:] = 0  # Right boundary

    #   Create mask for orange color
    hl = cv2.getTrackbarPos('h_Low', 'Trackbars')
    hu = cv2.getTrackbarPos('h_Hi', 'Trackbars')
    sl = cv2.getTrackbarPos('satLow', 'Trackbars')
    su = cv2.getTrackbarPos('satHigh', 'Trackbars')
    vl = cv2.getTrackbarPos('valLow', 'Trackbars')
    vu = cv2.getTrackbarPos('valHigh', 'Trackbars')

    lower_orange = (hl,sl,vl)   # Adjust these values based on the orange color in your environment
    upper_orange = (hu,su,vu)
    mask1 = cv2.inRange(hsv, lower_orange, upper_orange)
    mask1= cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask1= cv2.GaussianBlur(mask1, (5, 5), 0)
    # cv2.imshow("mask", mask2)
    # cv2.imshow("Calibrate2", mask1)
    # Find contours
    contours, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print()
    
    if not contours:
        # cv2.imshow("mask", mask1)
        print("No contours found. Exiting.")
        continue
    else:
        # Calculate contour areas and sort them
        contour_areas = [(cv2.contourArea(c), c) for c in contours]
        contour_areas.sort(key=lambda x: x[0], reverse=True)
        
        # Select the top 1 contours by area
        largest_contours = [c for area, c in contour_areas[:1]]
        rect = cv2.minAreaRect(largest_contours[0])
        box = cv2.boxPoints(rect)
        bounding_box = np.intp(box)
        # print(bounding_box)
        
        if frame_count > frame_initialize:
            frame_no = frame_count - frame_initialize
            # Calculate the number of pixels that span the object
            pixel_size = np.array(sorted(rect[1],reverse=True))
            pixels_per_m = (pixel_size/object_size)

        
            # Calculate the cumulative average number of pixels that span the object
            sum_pixel_size = sum_pixel_size + pixel_size
            Cum_pixel_size = sum_pixel_size/frame_no  # width in pixels
            sum_pixels_per_m = pixels_per_m + sum_pixels_per_m
            Cum_pixels_per_m = sum_pixels_per_m/frame_no
            
            if err < 1e-4:
                err_mm = (Cum_pixel_size/pixels_per_m_f - object_size)*1000
                print("Converged pixels per meter = ",pixels_per_m_f," with an error of ",err_mm,"mm")
                break
            else:
                pixels_per_m_prev = pixels_per_m_f
                pixels_per_m_f = sum(Cum_pixels_per_m*(object_size/sum(object_size)))
                err = abs(pixels_per_m_f - pixels_per_m_prev)
                print(frame_count,'frame',Cum_pixels_per_m,pixels_per_m,Cum_pixel_size,pixel_size)
                continue
        else:
            print('~',frame_count,'frame')

    cv2.drawContours(undistorted_frame, [bounding_box], -1, (0, 0, 255), 2)
#     # Display the frame
    cv2.imshow("Calibrate", undistorted_frame)
    # time.sleep()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
calibration_data = {'pixels_per_m': pixels_per_m_f,
'camera_matrix': camera_matrix,
'distortion_coeffs': distortion_coeffs}
# Wait for user input before proceeding to the next position
# input(f"Calibration for {position} position complete. Press Enter to continue to the next position...")


np.savez('runcam_calibration_data_2k.npz', **calibration_data)
    
print("Camera calibration successful.")

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()
    
