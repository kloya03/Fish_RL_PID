#!/usr/bin/env python3

# Import the necessary libraries
import rospy # Python library for ROS
from sensor_msgs.msg import Image # Image is the message type
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import cv2 # OpenCV library
import numpy as np
from custom_msgs.msg import cam_tracking_data
from std_msgs.msg import Header
import datetime
import pickle
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped


def nothing(x):
    pass

com = (0,0)
previous_com = []
previous_head_angle = 0
unwrapped_head_angle = None
bridge = CvBridge() # Create a bridge object
previous_time = None
prev_rostime = 0

# These values should be obtained from a camera calibration process
calibration_data = np.load('/home/fishcam_ros_ws/src/fish_pc/scripts/runcam_calibration_data_2k.npz')
pixels_per_m = calibration_data['pixels_per_m']
print(pixels_per_m)
# pixels_per_m = 573.7
latest_target = None
CS = (375,1200)
# # Open and load the data Path to the .pkl file
file_path = "/home/fishcam_ros_ws/src/fish_pc/scripts/fisheye_calibration_final_v2.pkl"
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
        
def pixel_to_pose(px, py, pixels_per_m, CS):
    """None
    Convert pixel to PoseStamped in meters relative to the CS origin.
    Flip x due to 180° rotation.
    """
    pose = PoseStamped()
    pose.header.frame_id = "world_frame"  # Use your robot's fixed frame (e.g., "world" or "camera_frame")
    pose.pose.position.x = (px - CS[0])  / pixels_per_m
    pose.pose.position.y = (py - CS[1]) * (-1)/ pixels_per_m
    pose.pose.position.z = 0.0
    pose.pose.orientation.w = 1.0  # no rotation
    return pose

def undistort_image(frame):
    # Undistort the frame
    undistorted_frame = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
    # undistorted_frame[:10, :] = 0  # Top boundary
    # undistorted_frame[950:, :] = 0  # Bottom boundary
    # undistorted_frame[:, :150] = 0  # Left boundary
    undistorted_frame[:, -250:] = 0  # Right boundary
    return undistorted_frame

# cv2.namedWindow('Trackbars')
# cv2.moveWindow('Trackbars', 1400, 100)
# cv2.resizeWindow('Trackbars',350, 250)  # Set the desired size (width, height)

# cv2.createTrackbar('h_Low', 'Trackbars', 3, 255, nothing)
# cv2.createTrackbar('h_Hi', 'Trackbars', 45, 255, nothing)

# cv2.createTrackbar('satLow', 'Trackbars', 140, 255, nothing)
# cv2.createTrackbar('satHigh', 'Trackbars', 255, 255, nothing)

# cv2.createTrackbar('valLow', 'Trackbars', 150, 255, nothing)
# cv2.createTrackbar('valHigh', 'Trackbars', 245, 255, nothing)


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
            self.filtered_value = new_value
            # Apply the EMA formula
            # self.sigma_recent = np.sqrt(self.alpha_sigma * (new_value - self.filtered_value) ** 2 + 
            #                         (1 - self.alpha_sigma) * self.sigma_recent ** 2)
            # deviation = abs(new_value - self.filtered_value)
            # self.alpha_t = 1 / (1 + deviation / (self.sigma_recent + self.epsilon))
            # self.filtered_value = self.alpha_t * new_value + (1 - self.alpha_t) * self.filtered_value
        return self.filtered_value
    
# Initialize LowPassFilter instances
lpf_com = LowPassFilter()
lpf_head_angle = LowPassFilter()
lpf_body_vel = LowPassFilter()
lpf_angZ_vel = LowPassFilter()
lpf_TailAngle_1 = LowPassFilter()
lpf_TailAngle_2 = LowPassFilter()
    
def publish_video(cap, prev_rostime):
    while not rospy.is_shutdown():
        tstart = rospy.Time.now()
        # print(f"FPS: {1/(tstart - prev_rostime).to_sec()}")
        # prev_rostime = tstart
        global com, missed_frame, previous_com, previous_head_angle,\
        area, pixels_per_m, bridge, camera_matrix, distortion_coeffs,\
             new_camera_matrix, h, w, CS, previous_time, unwrapped_head_angle
        # Get the current timestamp
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Node is publishing to the video_frames topic using message 
        # type Image and video data using cam_tracking_data
        pub_cam_data= rospy.Publisher('/runcam_tracking', cam_tracking_data, queue_size=20)
        # pub_waypoints = rospy.Publisher('runcam_waypoints', Pose2D, queue_size=30)
        cam = cam_tracking_data()
        
        ret, frame = cap.read()
        if not ret:
            break
        

        undistorted_frame = undistort_image(frame)
        undistorted_frame = cv2.flip(undistorted_frame, 0)
        undistorted_frame = cv2.flip(undistorted_frame, 1)
        # # Convert to HSV
        hsv = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2HSV)
        hsv[:200, :] = 0  # Top boundary
        hsv[-230:, :] = 0  # Bottom boundary
        hsv[:, :350] = 0  # Left boundary
        hsv[:, -150:] = 0  # Right boundary
        
        #   Create mask for orange color
        # hl = cv2.getTrackbarPos('h_Low', 'Trackbars')
        # hu = cv2.getTrackbarPos('h_Hi', 'Trackbars')
        # sl = cv2.getTrackbarPos('satLow', 'Trackbars')
        # su = cv2.getTrackbarPos('satHigh', 'Trackbars')
        # vl = cv2.getTrackbarPos('valLow', 'Trackbars')
        # vu = cv2.getTrackbarPos('valHigh', 'Trackbars')
        # lower_orange = (hl,sl,vl)   # Adjust these values based on the orange color in your environment
        # upper_orange = (hu,su,vu)
        
        lower_orange = (0,150,145)   # Adjust these values based on the orange color in your environment
        upper_orange = (60,255,255)

        mask1 = cv2.inRange(hsv, lower_orange, upper_orange)
        mask2 = cv2.morphologyEx(mask1, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        mask2 = cv2.GaussianBlur(mask2, (5, 5), 0)

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

        cam.header = Header()
        cam.header.frame_id = "World_camera_frame"  # Set the frame ID
        cam.header.stamp = rospy.Time.now()
        cam.x_position = np.nan
        cam.y_position = np.nan
        cam.z_angular_velocity = np.nan
        cam.body_velocity = np.nan
        cam.head_angle = np.nan
        cam.tail_angle_1 = np.nan
        cam.tail_angle_2 = np.nan
        
        # dt = 'nan'

        if len(current_positions)>=2:
            (x1,y1) = current_positions[0]
            (x2,y2)= current_positions[1]
            com_f = ((x2 + x1) / (2), (y2 + y1) / (2))
            com_p = [1,-1]*(np.array(com_f) - CS)  # tranlsate to frame reference (180 deg rot in x-axis)
            com = com_p/pixels_per_m  # convert to meters
            filtered_com = lpf_com.apply(np.array(com))
            filtered_com = filtered_com.tolist()  # Ensure com is a list
            cam.x_position = filtered_com[0]
            cam.y_position = filtered_com[1]
            head_angle =np.arctan2(-y1+y2,x1-x2)  # in rad, direction of fish head  opposite Y as it is in pixel
            
            
            filtered_head_angle = lpf_head_angle.apply(head_angle)
            cam.head_angle = filtered_head_angle
            
            if not np.isnan(com).any():
                if previous_com:
                    # print(previous_time, previous_com, com)
                    dt = (cam.header.stamp - previous_time).to_sec()
                    # print(1/dt)
                    # Calculate distance between previous and current positions
                    distance = np.linalg.norm(np.array(com) - np.array(previous_com))
                    cam.Vx = (com[0] - previous_com[0])/dt
                    cam.Vy = (com[1] - previous_com[1])/dt
                    # Estimate velocity (distance per frame)
                    body_vel = distance/dt
                    filtered_body_vel = lpf_body_vel.apply(body_vel)
                    cam.body_velocity = filtered_body_vel  # In m per second
                    # print(velocity)
                    if unwrapped_head_angle is None:
                        unwrapped_head_angle = cam.head_angle

                    d = head_angle -previous_head_angle
                    if d < -np.pi:
                        d+= 2*np.pi 
                    elif d > np.pi:
                        d -= 2*np.pi 
                        
                    unwrapped_head_angle += d
                    angZ_vel = d/dt
                    # angZ_vel = (cam.head_angle-previous_head_angle)/dt
                    filtered_angZ_vel = lpf_angZ_vel.apply(angZ_vel)
                    cam.z_angular_velocity = filtered_angZ_vel # In rad per second      

            if len(current_positions)>=3:
                (x3,y3) = current_positions[2]
                TailAngle_1 = np.arctan2(-y2+y3,x2-x3) # - cam.head_angle # in rad, 
                filtered_TailAngle_1 = lpf_TailAngle_1.apply(TailAngle_1)  # direction of fish tail (w.r.t head)
                cam.tail_angle_1 = filtered_TailAngle_1
            
            if len(current_positions)>=4:
                (x4,y4) = current_positions[3]
                TailAngle_2 = np.arctan2(-y3+y4,x3-x4) # - cam.head_angle # in rad,
                filtered_TailAngle_2 = lpf_TailAngle_2.apply(TailAngle_2)  #  direction of fish tail (w.r.t head)
                cam.tail_angle_2 = filtered_TailAngle_2 
           
            if not np.isnan(com).any():
                previous_com = filtered_com
                previous_head_angle = cam.head_angle
                previous_time = cam.header.stamp
                # print(previous_time)
                
                
        col = (255,255,255)
        w_off = 1700
        text_w = 500
        h_off = 50
        Font_size = 1
        ## Draw the vertical and horizontal axis
        cv2.line(undistorted_frame, (CS[0], 0), (CS[0], h), [0,0,0], 4)
        cv2.line(undistorted_frame, (0, CS[1]), (w, CS[1]), [0,0,0], 4)
        # cv2.putText(undistorted_frame, dt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, col, 1)
        cv2.putText(undistorted_frame, current_time, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)
        cv2.putText(undistorted_frame, current_time, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)
        cv2.putText(undistorted_frame, f"Position: ({cam.x_position:.2f} , {cam.y_position:.2f}) [m]", (w-w_off, h_off),
                    cv2.FONT_HERSHEY_SIMPLEX, Font_size, col, 2)
        cv2.putText(undistorted_frame, f"Lin Vel: {cam.body_velocity:.2f} [m/s]" , (w-w_off,2*h_off),
                        cv2.FONT_HERSHEY_SIMPLEX, Font_size, col, 2) 
        cv2.putText(undistorted_frame, f"Head Angle: {np.rad2deg(cam.head_angle):.2f} [deg]", (w-w_off, 3*h_off),
                    cv2.FONT_HERSHEY_SIMPLEX, Font_size, col, 2)
        cv2.putText(undistorted_frame, f"Ang Vel: {np.rad2deg(cam.z_angular_velocity):.2f} [deg/s]",(w-w_off+text_w,h_off),
                     cv2.FONT_HERSHEY_SIMPLEX, Font_size, col, 2) 
        cv2.putText(undistorted_frame, f"Tail Ang 1: {np.rad2deg(cam.tail_angle_1):.2f} [deg]", (w-w_off+text_w, 2*h_off),
                        cv2.FONT_HERSHEY_SIMPLEX, Font_size, col, 2)
        cv2.putText(undistorted_frame, f"Tail Ang 2: {np.rad2deg(cam.tail_angle_2):.2f} [deg]", (w-w_off+text_w, 3*h_off),
                        cv2.FONT_HERSHEY_SIMPLEX, Font_size, col, 2)
        # cv2.circle(undistorted_frame, center=(1200, 720), radius=300, color=(0, 255, 0), thickness=3)
        for i in range(len(current_positions)):
            cv2.putText(undistorted_frame, f"{i+1}", (current_positions[i][0]+10, current_positions[i][1]+10),
                cv2.FONT_HERSHEY_SIMPLEX, Font_size-0.5, [0,0,0], 1)
                
        # Convert from BGR (OpenCV format) to RGB (ROS expects RGB)
        # rgb_image = cv2.cvtColor(undistorted_frame, cv2.COLOR_BGR2RGB)
        # cv2.imshow('frame', hsv)
        cv2.imshow('runcam', undistorted_frame)
        # Write the frame
        out.write(undistorted_frame)
        cv2.waitKey(1)
        # Publish the converted image
        # pub_cam.publish(bridge.cv2_to_imgmsg(rgb_image, "rgb8"))
        pub_cam_data.publish(cam)
        
        # Go through the loop 10 times per second
        rate = rospy.Rate(30) # 10hz
        rate.sleep()
                
if __name__ == '__main__':
    try:
        rospy.init_node('runcam', anonymous=True)
        prev_rostime = rospy.Time.now()
        filepath = 0#'/home/kloya/Videos_02_23/test8_10_3.mp4'
        cap = cv2.VideoCapture(filepath)
        cv2.namedWindow("runcam")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS,30)
        fps = cap.get(cv2.CAP_PROP_FPS)
        current_datetime = datetime.datetime.now()
        timestamped_filename_video = f"/home/fishcam_ros_ws/src/fish_pc/RAL_turns/sinusoidal_turns_30_{current_datetime.strftime('%Y%m%d_%H%M')}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(timestamped_filename_video, fourcc, 15, (w, h))

        publish_video(cap, prev_rostime)

    except rospy.ROSInterruptException:
        cap.release()
        # After quitting, waypoints are available
        out.release()
        cv2.destroyAllWindows()
        pass






# def calculate_distance(point1, point2):
#     """
#     Calculate the Euclidean distance between two points.
    
#     Parameters:
#     point1 (tuple): The first point as a tuple (x1, y1).
#     point2 (tuple): The second point as a tuple (x2, y2).
    
#     Returns:
#     float: The Euclidean distance between the two points.
#     """
#     x1, y1 = point1
#     x2, y2 = point2
#     distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
#     return distancehat it does

    # Measures bus voltage, current, and pow