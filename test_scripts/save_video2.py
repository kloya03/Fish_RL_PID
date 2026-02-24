import cv2
import datetime
import numpy as np
import time
import pickle

# camera_matrix = np.array([[1.252e+03, 0, 1280],
#                           [0, 1e+03, 720],
#                           [0, 0, 1]], dtype=np.float32)
# distortion_coeffs = np.array([-0.02327429, 0.04284276, -0.0447615,  0.01550882], dtype=np.float32)



file_path = "../Camera_Calibrate_runcam/fisheye_calibration_final_v2.pkl"
# file_path = "../Camera_Calibrate_runcam/fisheye_calibration_1080p_25.pkl"
with open(file_path, "rb") as file:  # "rb" means read in binary mode
    data = pickle.load(file)
camera_matrix, distortion_coeffs = data

# Print the variables to verify
print("Camera Matrix:\n", camera_matrix)
print("Distortion Coefficients:\n", distortion_coeffs)

# Load the video
h, w = 1440, 2560
# h, w = 1080, 1920
# h, w = 720, 1280
# new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs, (w, h), 1, (w, h))
# new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(camera_matrix, distortion_coeffs, (w, h), np.eye(3), balance=1)
# new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(camera_matrix, distortion_coeffs, (w, h), np.eye(3), balance=1)
map1, map2 = cv2.fisheye.initUndistortRectifyMap(camera_matrix, distortion_coeffs, np.eye(3), new_camera_matrix, (w, h), cv2.CV_16SC2)

# Open the video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
cap.set(cv2.CAP_PROP_FPS,30)
fps = cap.get(cv2.CAP_PROP_FPS)
current_datetime = datetime.datetime.now()
# timestamped_filename_video = f"Test1_11080p_dist{current_datetime.strftime('%Y%m%d_%H%M')}.mp4"
# Define the codec and create VideoWriter object
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('/home/kloya/' + timestamped_filename_video, fourcc, fps, (w, h))
while(cap.isOpened()):
    ts = time.time()
    ret, frame = cap.read()
    if ret:
        # Get the current timestamp
        
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        frame = cv2.flip(frame, 0)
        frame = cv2.flip(frame, 1)
        # map1, map2 = cv2.fisheye.initUndistortRectifyMap(camera_matrix, distortion_coeffs, np.eye(3), new_camera_matrix, (w, h), cv2.CV_16SC2)
        frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
        # frame[:80, :] = 0  # Top boundary
        # frame[-145:, :] = 0  # Bottom boundary
        # frame[:, :200] = 0  # Left boundary
        # frame[:, -10:] = 0  # Right boundary

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Make the specified region black based on the ROI
        hsv[:190, :] = 0  # Top boundary
        hsv[-200:, :] = 0  # Bottom boundary
        hsv[:, :315] = 0  # Left boundary
        hsv[:, -160:] = 0  # Right boundary

            #   Create masnp.array([[roll], [pitch], [yaw]])
        hl = 3#cv2.getTrackbarPos('h_Low', 'Trackbars')
        hu = 45#cv2.getTrackbarPos('h_Hi', 'Trackbars')
        sl = 140#cv2.getTrackbarPos('satLow', 'Trackbars')
        su = 255#cv2.getTrackbarPos('satHigh', 'Trackbars')
        vl = 150#cv2.getTrackbarPos('valLow', 'Trackbars')
        vu = 245#cv2.getTrackbarPos('valHigh', 'Trackbars')

        lower_orange = (hl,sl,vl)   # Adjust these values based on the orange color in your environment
        upper_orange = (hu,su,vu)
        mask1 = cv2.inRange(hsv, lower_orange, upper_orange)
        mask2 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask2 = cv2.GaussianBlur(mask2, (5, 5), 0)
        # cv2.imshow("mask", mask2)

        # # Find contours and sort them along areas
        contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_areas = [(cv2.contourArea(c), c) for c in contours]
        contour_areas.sort(key=lambda x: x[0], reverse=True)
        largest_contours = [c for area, c in contour_areas[:4]] # top 4 contours by area

        cv2.drawContours(frame, largest_contours, -1, (0,90,200), 1) # Draw the contour and center

        # Put the timestamp on the frame
        cv2.putText(frame, timestamp, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Write the frame
        # out.write(frame)

        # Display the frame
        cv2.imshow('frame', frame)
        # dt = ts - time.time()
        # print(1/dt)
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
# out.release()
cv2.destroyAllWindows()