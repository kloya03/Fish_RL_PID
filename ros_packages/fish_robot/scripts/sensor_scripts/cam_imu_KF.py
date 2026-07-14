#!/usr/bin/env python3

import rospy
import numpy as np
from filterpy.kalman import KalmanFilter
from custom_msgs.msg import cam_tracking_data, KFstate
from sensor_msgs.msg import Imu
import time
# from diagnostic_msgs.msg import DiagnosticArray
from tf.transformations import euler_from_quaternion, quaternion_matrix, euler_matrix

class SensorFusionKF:
    def __init__(self):
        rospy.init_node("sensor_fusion_kf", anonymous=True)

        # Initialize Kalman Filter with 9 states:
        # [x, y, z, vx, vy, vz, roll, pitch, yaw]
        self.kf = KalmanFilter(dim_x=9, dim_z=5)
        self.kf.x = np.zeros((9, 1))  # Initial state
        
        self.prev_zcam = np.zeros((5,1))
        self.last_imu_time = None  
        self.last_cam_time = None
        self.dt_history = []
        self.prev_unwrap_yaw = None

        # State Transition Matrix (F) - Will be updated dynamically
        self.kf.F = np.eye(9)

        # Process Noise Covariance (Q)
        self.kf.Q = np.eye(9) *0.01

        # Covariance Matrix (P)
        self.kf.P = np.eye(9) * 1  

        # Measurement Matrices
        self.H_cam = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0],  # x from camera
            [0, 1, 0, 0, 0, 0, 0, 0, 0],  # y from camera
            [0, 0, 0, 0, 0, 0, 0, 0, 1],  # yaw from camera
            [0, 0, 0, 1, 0, 0, 0, 0, 0],  # Vx from camera
            [0, 0, 0, 0, 1, 0, 0, 0, 0],  # Vy from camera
        ])
        
        self.H_imu = np.array([
            [0, 0, 0, 0, 0, 0, 1, 0, 0],  # roll from IMU
            [0, 0, 0, 0, 0, 0, 0, 1, 0],  # pitch from IMU
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # nothing from IMU
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # nothing from IMU
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # nothing from IMU
        ])

        # Measurement Noise Covariances
        self.R_cam = 1e-6*np.diag([0.005, 0.005, 0.005, 0.05, 0.05])  # x, y, yaw from camera
        self.R_imu = 0.02*np.diag([0.01, 0.01, 1, 1, 1])  # roll, pitch, yaw from IMU

        # Subscribers
        self.imu_sub = rospy.Subscriber("/IMU_bno08x/raw", Imu, self.imu_callback, queue_size=10)
        self.cam_sub = rospy.Subscriber("/runcam_tracking", cam_tracking_data, self.cam_callback, queue_size=10)
        # self.imu_status_sub = rospy.Subscriber('/IMU_bno08x/status', DiagnosticArray, self.imu_status_callback, queue_size=10)
        # Publisher
        self.pub = rospy.Publisher("/KF_state_estimate", KFstate, queue_size=10)
        rospy.loginfo(rospy.get_caller_id() + "  cam-imu kalman fusion launched.")

    def imu_callback(self, msg):
        """ IMU provides acceleration and orientation updates for prediction and update """
        imu_current_time = msg.header.stamp.to_sec()  #time.time()
        
        # print("imu_update_time",imu_current_time)
        if self.last_imu_time is None:
            self.last_imu_time = imu_current_time
            return  # Skip first frame

        dt = imu_current_time - self.last_imu_time
        dt = np.array(dt)
        self.last_imu_time = imu_current_time
        # print(imu_current_time, "dt:", 1/dt)
        # Update state transition matrix (F)
        self.kf.F = np.array([
            [1, 0, 0, dt, 0, 0, 0, 0, 0],  
            [0, 1, 0, 0, dt, 0, 0, 0, 0],  
            [0, 0, 1, 0, 0, dt, 0, 0, 0],  
            [0, 0, 0, 1, 0, 0, 0, 0, 0],  
            [0, 0, 0, 0, 1, 0, 0, 0, 0],  
            [0, 0, 0, 0, 0, 1, 0, 0, 0],  
            [0, 0, 0, 0, 0, 0, 1, 0, 0],  
            [0, 0, 0, 0, 0, 0, 0, 1, 0],  
            [0, 0, 0, 0, 0, 0, 0, 0, 1],  
        ])

        # Control Input (u) - IMU acceleration
        u_body = np.array([[msg.linear_acceleration.x], 
                      [msg.linear_acceleration.y], 
                      [msg.linear_acceleration.z],
                      [msg.angular_velocity.x],
                      [msg.angular_velocity.y],
                      [msg.angular_velocity.z]])

        # Control Matrix (B)
        B = np.array([
            [0.5 * dt**2, 0, 0],
            [0, 0.5 * dt**2, 0],
            [0, 0, 0.5 * dt**2],
            [dt, 0, 0],
            [0, dt, 0],
            [0, 0, dt]  
        ])

        R = euler_matrix(self.kf.x[6, 0], self.kf.x[7, 0], self.kf.x[8, 0],'rxyz')  # This gives you a 4x4 matrix
        R_3x3 = R[:3, :3]  # Extract the 3x3 rotation matrix
        a_world = np.dot(R_3x3, u_body[:3])  # Transform acceleration
        #omega_world = np.dot(R_3x3, u_body[3:])  # Transform angular velocity
        # u_world = np.concatenate((a_world, omega_world), axis=0)  # Concatenate into a single array
        u_world = np.concatenate((a_world), axis=0)  # Concatenate into a single array
        
        # Convert IMU quaternion to roll, pitch, yaw62
        self.quaternion = (
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        )
        roll, pitch, yaw = euler_from_quaternion(self.quaternion)

        # IMU Update
        z_imu = np.array([[roll], [pitch], [0], [0], [0]])
        self.kf.update(z_imu, H=self.H_imu, R=self.R_imu)
        
        # Predict Step
        self.kf.predict(u=u_world.flatten(), B=B)

        self.publish_state(msg.header.stamp)
        
    def cam_callback(self, msg):
        cam_current_time = msg.header.stamp.to_sec()  #time.time()
        # print("imu_update_time",imu_current_time)
        if self.last_cam_time is None:
            self.last_cam_time = cam_current_time
            return  # Skip first frame

        dt = cam_current_time - self.last_cam_time
        dt = np.array(dt)
        self.last_cam_time = cam_current_time
        # print(cam_current_time, "dt:", 1/dt)
        # print(self.kf.x)
        # Check for NaN values in the camera data
        if np.isnan(msg.y_position) or np.isnan(msg.x_position) or np.isnan(msg.head_angle):
            rospy.logwarn("NaN values detected in camera data, skipping update.")
            return
        """ Camera provides position updates (x, y, yaw) """
        yaw_cam = np.deg2rad(msg.head_angle)
        if self.prev_unwrap_yaw is not None:
            self.prev_unwrap_yaw = self.unwrap_angle(self.prev_unwrap_yaw, yaw_cam)
        else:
            self.prev_unwrap_yaw = yaw_cam
            
        z_cam = np.array([[msg.x_position], [msg.y_position], [self.prev_unwrap_yaw], [msg.Vx], [msg.Vy]])
        self.prev_zcam = self.smooth_vector(self.prev_zcam, z_cam, 0.25)
        wrap_yaw = self.wrap_angle(self.prev_zcam[2])
        # print(self.prev_zcam.shape, wrap_yaw.shape)
        zcam = np.concatenate((self.prev_zcam[0:2],wrap_yaw,self.prev_zcam[3:6]), axis=0)
        # print(self.prev_zcam)
        self.kf.update(zcam , H=self.H_cam, R=self.R_cam)
        self.publish_state(msg.header.stamp)
        
    def smooth_vector(self, prev, new, alpha):
        """
        Exponential smoothing for vectors.
        """
        # prev = np.array(prev)
        # print(prev)
        # print(new)
        new = np.array(new)
        if prev is None:
            return new
        return prev + alpha * (new - prev)

    def wrap_angle(self, angle):
        return np.array([(angle + np.pi) % (2 * np.pi) - np.pi])

    def unwrap_angle(self, prev_angle, new_angle):
        """
        Unwrap angle to avoid jumps at ±pi boundary.
        """
        delta = new_angle - prev_angle
        delta = (delta + np.pi) % (2 * np.pi) - np.pi
        return prev_angle + delta

    def publish_state(self,timestamp):
        """ Publish the estimated state """

        R = euler_matrix(self.kf.x[6, 0], self.kf.x[7, 0], self.kf.x[8, 0],'rxyz')  # This gives you a 4x4 matrix
        R_3x3 = R[:3, :3]  # Extract the 3x3 rotation matrix
        V_body = np.dot(R_3x3.T, self.kf.x[3:6])
        state_msg = KFstate()
        state_msg.header.stamp = timestamp
        state_msg.x = self.kf.x[0, 0]
        state_msg.y = self.kf.x[1, 0]
        state_msg.z = self.kf.x[2, 0]
        state_msg.Vx = self.kf.x[3, 0]
        state_msg.Vy = self.kf.x[4, 0]
        state_msg.Vz = self.kf.x[5, 0]
        state_msg.roll = self.kf.x[6, 0] #np.rad2deg(self.kf.x[6, 0])
        state_msg.pitch = self.kf.x[7, 0] #np.rad2deg(self.kf.x[7, 0])
        state_msg.yaw = self.kf.x[8, 0] #np.rad2deg(self.kf.x[8, 0])
        state_msg.vb_x = V_body[0, 0]
        state_msg.vb_y = V_body[1, 0]
        state_msg.vb_z = V_body[2, 0]
        self.pub.publish(state_msg)

if __name__ == "__main__":
    try:
        SensorFusionKF()
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo(rospy.get_caller_id() + "  cam-imu kalman fusion exited with exception.")
        pass



