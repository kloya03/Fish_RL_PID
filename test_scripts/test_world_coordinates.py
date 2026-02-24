#!/usr/bin/env python3

from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library
import numpy as np
import time
import datetime
import pickle

def nothing(x):
    pass

class LowPassFilter:
    def __init__(self, alpha_sigma=0.1, epsilon = 1e-6):
        self.alpha_sigma = alpha_sigma  # Smoothing factor
        self.filtered_value = None  # Initially, no filtered value
        self.epsilon = epsilon
        self.sigma_recent = 0
        self.alpha_t = 1

    def apply(self, new_value):
        if self.filtered_value is None:
            # First-time use, set the initial filtered value as the first raw value
            self.filtered_value = new_value
        else:
            # Apply the EMA formula
            self.sigma_recent = np.sqrt(self.alpha_sigma * (new_value - self.filtered_value) ** 2 + 
                                    (1 - self.alpha_sigma) * self.sigma_recent ** 2)
            
            deviation = abs(new_value - self.filtered_value)
            
            self.alpha_t = 1 / (1 + deviation / (self.sigma_recent + self.epsilon))
            
            self.filtered_value = self.alpha_t * new_value + (1 - self.alpha_t) * self.filtered_value
            
        return self.filtered_value
    
# Initialize LowPassFilter instances
lpf_com = LowPassFilter()
lpf_head_angle = LowPassFilter()
lpf_body_vel = LowPassFilter()
lpf_angZ_vel = LowPassFilter()
lpf_TailAngle_1 = LowPassFilter()
lpf_TailAngle_2 = LowPassFilter()
    


        
        
if __name__ == '__main__':
    try:
        
        com = (0,0)
        previous_com = []
        previous_head_angle = 0
        bridge = CvBridge() # Create a bridge object
        previous_time = None

        # cv2.namedWindow('Trackbars')
        # cv2.moveWindow('Trackbars', 1400, 100)
        # cv2.resizeWindow('Trackbars',350, 250)  # Set the desired size (width, height)

        # cv2.createTrackbar('h_Low', 'Trackbars', 3, 255, nothing)
        # cv2.createTrackbar('h_Hi', 'Trackbars', 45, 255, nothing)

        # cv2.createTrackbar('satLow', 'Trackbars', 140, 255, nothing)
        # cv2.createTrackbar('satHigh', 'Trackbars', 255, 255, nothing)

        # cv2.createTrackbar('valLow', 'Trackbars', 150, 255, nothing)
        # cv2.createTrackbar('valHigh', 'Trackbars', 245, 255, nothing)

        # These values should be obtained from a camera calibration process
        calibration_data = np.load('../Camera_Calibrate_runcam/runcam_calibration_data_2k.npz')
        pixels_per_m = 423#1calibration_data['pixels_per_m']
        # pixels_per_m = 245
        target_pix = []
        CS = (1523,489)
        # camera_matrix = np.array([[600, 0, 640],
        #                           [0, 600, 360],
        #                           [0, 0, 1]], dtype=np.float32)
        # distortion_coeffs = np.array([-0.3, 0.1, 0, 0], dtype=np.float32)
        # # Open and load the data Path to the .pkl file
        file_path = "../Camera_Calibrate_runcam/fisheye_calibration_runcam_2k_25.pkl"
        with open(file_path, "rb") as file:  # "rb" means read in binary mode
            data = pickle.load(file)
        camera_matrix, distortion_coeffs = data

        # Print the variables to verify
        print("Camera Matrix:\n", camera_matrix)
        print("Distortion Coefficients:\n", distortion_coeffs)

        # Load the video
        h, w = 1440, 2560
        # new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coeffs, (w, h), 1, (w, h))
        new_camera_matrix = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(camera_matrix, distortion_coeffs, (w, h), np.eye(3), balance=1)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(camera_matrix, distortion_coeffs, np.eye(3), new_camera_matrix, (w, h), cv2.CV_16SC2)

        filepath = 4#'/home/kloya/Videos_02_23/test8_10_3.mp4'
        cap = cv2.VideoCapture(filepath)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(fps)
        # Define the codec and create VideoWriter object
        # filename = 
        # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # out = cv2.VideoWriter('/home/kloya/catkin_fishcam_ws/src/fish_pc/Fish_Test_Data/Model_Identification_test/Test2_8_2.mp4', fourcc, 10, (1280, 720))
        while cap.isOpened():
        # Get the current timestamp
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            ret, frame = cap.read()
            if ret:
                # frame = cv2.flip(frame, 0)
                # frame = cv2.flip(frame, 1)

                # Undistort the frame
                # cv2.imshow('undistorted_frameff', frame)
                # undistorted_frame = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
                # undistorted_frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
                # Convert to HSV
                undistorted_frame = frame
                hsv = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2HSV)
                # Make the specified region black based on the ROI
                hsv[:110, :] = 0  # Top boundary
                hsv[-110:, :] = 0  # Bottom boundary
                hsv[:, :110] = 0  # Left boundary
                hsv[:, -150:] = 0  # Right boundary
                hsv[:110, :] = 0
                hsv[1200: , 1694:] = 0
                hsv[:218 , 1694:] = 0
                # cv2.imshow("hsv", hsv)
                    #   Create mask for orange color
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

                # Find contours and sort them along areas
                contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour_areas = [(cv2.contourArea(c), c) for c in contours]
                contour_areas.sort(key=lambda x: x[0], reverse=True)
                largest_contours = [c for area, c in contour_areas[:4]] # top 3 contours by area

                current_positions = []
                area = []
                for i, contour in enumerate(largest_contours, start=1):
                # for contour in largest_contours:
                    if cv2.contourArea(contour) > 2:  # Filter small contours
                        M = cv2.moments(contour)  # Calculate moments of the contour
                        if M["m00"] != 0:  # Ensure the contour has a non-zero area
                            cx = int(M["m10"] / M["m00"])  # Calculate the x-coordinate of the contour's centroid
                            cy = int(M["m01"] / M["m00"])  # Calculate the y-coordinate of the contour's centroid
                            area.append(cv2.contourArea(contour))  # Append the contour area to the list
                            current_positions.append((cx, cy))  # Append the centroid position to the list
                            cv2.drawContours(undistorted_frame, [contour], -1, (0, 0, 0), 1) # Draw the contour and center
                            if i == 3:
                                dist_12 = np.linalg.norm(np.array(current_positions[0]) - np.array(current_positions[1]))
                                dist_13 = np.linalg.norm(np.array(current_positions[0]) - np.array(current_positions[2]))
                            elif i == 4:
                                dist_14 = np.linalg.norm(np.array(current_positions[0]) - np.array(current_positions[3]))
                                
                                # Determine the nearest points to point 1
                                distances = [(dist_12, 1), (dist_13, 2), (dist_14, 3)]
                                distances.sort(key=lambda x: x[0])
                                
                                # Reorder current_positions based on the sorted distances
                                nearest_indices = [0] + [distances[i][1] for i in range(3)]
                                current_positions = [current_positions[i] for i in nearest_indices]

                ## Draw the vertical and horizontal axis
                cv2.line(undistorted_frame, (CS[0], 0), (CS[0], h), [0,0,255], 3)
                cv2.line(undistorted_frame, (0, CS[1]), (w, CS[1]), [0,0,255], 3)
                
                stamp = time.time()
                x_position = np.nan
                y_position = np.nan
                z_angular_velocity = np.nan
                body_velocity = np.nan
                head_angle = np.nan
                tail_angle_1 = np.nan
                tail_angle_2 = np.nan

                if len(current_positions)>=2:
                    (x1,y1) = current_positions[0]
                    (x2,y2)= current_positions[1]
                    com_f = ((x2 + x1) / (2), (y2 + y1) / (2))
                    com_p = [1,-1]*(np.array(com_f) - CS)  # tranlsate to frame reference (180 deg rot in x-axis)
                    com = com_p/pixels_per_m  # convert to meters
                    print(com)

                    filtered_com = lpf_com.apply(np.array(com))
                    filtered_com = filtered_com.tolist()  # Ensure com is a list
                    x_position = filtered_com[0]
                    y_position = filtered_com[1]
                    head_angle = np.degrees(np.arctan2(-y1+y2,x1-x2)) # in degrees, direction of fish head  opposite Y as it is in pixel
                    filtered_head_angle = lpf_head_angle.apply(head_angle)
                    head_angle = filtered_head_angle
                    
                    if not np.isnan(com).any():
                        if previous_com:
                            # print(previous_time, previous_com, com)
                            dt = (stamp - previous_time)
                            # print(1/dt)
                            # Calculate distance between previous and current positions
                            distance = np.linalg.norm(np.array(com) - np.array(previous_com))
                            # Estimate velocity (distance per frame)
                            body_vel = distance/dt
                            filtered_body_vel = lpf_body_vel.apply(body_vel)
                            body_velocity = filtered_body_vel  # In m per second
                            # print(velocity)
                            angZ_vel = (head_angle-previous_head_angle)/dt
                            filtered_angZ_vel = lpf_angZ_vel.apply(angZ_vel)
                            z_angular_velocity = filtered_angZ_vel # In degrees per second      

                    if len(current_positions)>=3:
                        (x3,y3) = current_positions[2]
                        TailAngle_1 = np.degrees(np.arctan2(-y2+y3,x2-x3)) # - head_angle # in degrees, 
                        filtered_TailAngle_1 = lpf_TailAngle_1.apply(TailAngle_1)  # direction of fish tail (w.r.t head)
                        tail_angle_1 = filtered_TailAngle_1
                    
                    if len(current_positions)>=4:
                        (x4,y4) = current_positions[3]
                        TailAngle_2 = np.degrees(np.arctan2(-y3+y4,x3-x4)) # - head_angle # in degrees,
                        filtered_TailAngle_2 = lpf_TailAngle_2.apply(TailAngle_2)  #  direction of fish tail (w.r.t head)
                        tail_angle_2 = filtered_TailAngle_2 
                
                    if not np.isnan(com).any():
                        previous_com = filtered_com
                        previous_head_angle = head_angle
                        previous_time = stamp
                        # print(previous_time)
                        
                        
                col = (255,255,255)
                cv2.putText(undistorted_frame, current_time, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, col, 1)
                cv2.putText(undistorted_frame, f"Position: ({x_position:.2f} , {y_position:.2f}) [m]", (w-700, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
                cv2.putText(undistorted_frame, f"Lin Vel: {body_velocity:.2f} [m/s]" , (w-700,60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1) 
                cv2.putText(undistorted_frame, f"Head Angle: {head_angle:.2f} [deg]", (w-700, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
                cv2.putText(undistorted_frame, f"Ang Vel: {z_angular_velocity:.2f} [deg/s]",(w-450,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1) 
                cv2.putText(undistorted_frame, f"Tail Ang 1: {tail_angle_1:.2f} [deg]", (w-450, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
                cv2.putText(undistorted_frame, f"Tail Ang 2: {tail_angle_2:.2f} [deg]", (w-450, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
                for i in range(len(current_positions)):
                    cv2.putText(undistorted_frame, f"{i+1}", (current_positions[i][0]+10, current_positions[i][1]+10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
                    
                # Publish the target pose
                if target_pix:
                    cv2.circle(undistorted_frame, target_pix, 5, (0, 0, 255), -1)
                    cv2.putText(undistorted_frame, "Target", (target_pix[0]+10, target_pix[1]+10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                # Convert from BGR (OpenCV format) to RGB (ROS expects RGB)
                # rgb_image = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2RGB)
                # cv2.imshow('frame', frame)
                cv2.imshow('undistorted_frame', undistorted_frame)
                # Write the frame
                # out.write(undistorted_frame)
                cv2.waitKey(1)


    except KeyboardInterrupt:
        cap.release()
        # out.release()
        cv2.destroyAllWindows()
        pass