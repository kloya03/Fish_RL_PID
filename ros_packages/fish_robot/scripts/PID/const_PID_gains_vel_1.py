#!/usr/bin/env python3

import rospy
import iqmotion as iq
import numpy as np
import os
import time
from math import sin, cos
import pigpio
import board
import busio
import adafruit_bno08x
from filterpy.kalman import KalmanFilter
from tf.transformations import euler_from_quaternion, quaternion_matrix, euler_matrix
from adafruit_bno08x.i2c import BNO08X_I2C
from nav_msgs.msg import Path
from sensor_msgs.msg import Imu
from custom_msgs.msg import KFstate, purepursuit_control, cam_tracking_data
from RL_base_128.actor_policy_lowrange import ActorNumpy

class Fish_PurePursuitController:
    def __init__(self):
        rospy.init_node('fish_pure_pursuit_controller')
        
        # Initialize cam-Imu ekf 
        
        # Initialize Kalman Filter with 9 states:
        # [x, y, z, vx, vy, vz, roll, pitch, yaw]
        self.kf = KalmanFilter(dim_x=9, dim_z=5)
        self.kf.x = np.zeros((9, 1))  # Initial state
        
        self.prev_zcam = None
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


        # Parameters
        self.actor = ActorNumpy("/home/pi/fish_ros/src/fish_robot/scripts/RL_PID/RL_base_128/policy_weights_low_range.npz")
        self.vb_des = rospy.get_param('~vb_des', 0.25)
        self.Ld_min = rospy.get_param('~lookahead_distance_min', 0.25)
        self.Ld_max = rospy.get_param('~lookahead_distance_max', 0.65)
        self.angle_range = rospy.get_param('~angle_range', 160)  # servo range
        self.SERVO_PIN = 12
        self.Ld = 0.5 #self.Ld_max
        self.min_pw = 775 
        self.max_pw = 2275
        self.servo_angle = 0
        self.prev_servo_angle = 0
        self.t = 0
        self.start_time = None
        self.t_avg_erroryaw = 0
        self.t0 = 0
        self.vI=0
        self.aI = 0
        self.prev_error_yaw = 0
        self.prev_error_vel = 0
        self.forward = True
        self.pose = None
        self.path = None
        self.A = 0
        self.omega = 0
        self.phase = 0
        self.e_fa = 0  
        
        #Gains
        self.K_Ld = 4
        self.C_k = 0.5
        self.vb_max = 0.3
        self.vel_cntrl = True
        self.aKp = 0
        self.aKi = 0
        self.aKd = 0
        
        if self.vel_cntrl ==True:
            self.vKp = 2.5
            self.vKi = 0.001
            self.vKd = 0.05
            # 1. Setup the throttle tables
            self.a_tab = np.arange(5, 13.25, 0.25)        # 5:1:13
            self.w_tab = np.arange(1.5, 3.25, 0.25)   # 1.5:0.5:3
            self.A_tab, self.Om_tab = np.meshgrid(self.a_tab, self.w_tab)
            self.throttle_grid =  (self.A_tab * ((2 * np.pi *self.Om_tab)**2)/4614)**2
        
        # Hardware setup
        # IMU setup
        i2c = busio.I2C(board.SCL, board.SDA)
        self.bno = BNO08X_I2C(i2c,address=0x4a) # BNO080 (0x4b) BNO085 (0x4a)
        self.bno.enable_feature(adafruit_bno08x.BNO_REPORT_LINEAR_ACCELERATION)
        self.bno.enable_feature(adafruit_bno08x.BNO_REPORT_GYROSCOPE)
        self.bno.enable_feature(adafruit_bno08x.BNO_REPORT_GAME_ROTATION_VECTOR)
        
        # vertiq Motor setup
        com = iq.SerialCommunicator("/dev/ttyS0",baudrate=115200)
        #module = iq.Vertiq2306(com,module_idn=0)
        self.module = iq.Vertiq4006(com,0)
        voltage_now = self.module.get("power_monitor","volts")
        print(voltage_now)
        
        # start Pigpiod for servo control
        print("Starting pigpiod...")
        if os.system("pgrep pigpiod > /dev/null") != 0:
            os.system('sudo pigpiod')
        time.sleep(1)
        self.pi = pigpio.pi()
        if not self.pi.connected:
            rospy.logerr("Failed to connect to pigpiod daemon!")
            rospy.signal_shutdown("pigpiod connection failed")
            return
        self.pi.set_PWM_frequency(self.SERVO_PIN, 333)

        # Subscribers
        rospy.Subscriber('/runcam_tracking', cam_tracking_data, self.cam_callback)
        rospy.Subscriber('/runcam_path', Path, self.TargetPath_callback)

        # Publisher
        self.pub_imu_raw= rospy.Publisher('IMU_bno08x/raw', Imu, queue_size=50)
        self.pub_KF = rospy.Publisher('/KF_state_estimate', KFstate, queue_size=50)
        self.pub_PP = rospy.Publisher('/purepursuit_target', purepursuit_control, queue_size=10)

        # Register shutdown hook
        rospy.on_shutdown(self.shutdown_hook)
        self.rate = rospy.Rate(100)  # Loop at 100 Hz
        
    
    def smooth_data(self, prev, new, alpha):
        """ Exponential smoothing for data."""
        # new = np.array(new)
        if prev is None:
            return new
        return prev + alpha * (new - prev)
    

    def wrap_angle(self, angle):
        return np.array([(angle + np.pi) % (2 * np.pi) - np.pi])
    

    def unwrap_angle(self, prev_angle, new_angle):
        """  Unwrap angle to avoid jumps at +-pi boundary. """
        delta = new_angle - prev_angle
        delta = (delta + np.pi) % (2 * np.pi) - np.pi
        return prev_angle + delta
        
        
    def cam_callback(self, msg):
        tc = rospy.Time.now()
        if np.isnan(msg.y_position) or np.isnan(msg.x_position) or np.isnan(msg.head_angle):
            rospy.logwarn("NaN values detected in camera data, skipping update.")
            return
        """ Camera provides position updates (x, y, yaw, vx, vy) """
        yaw_cam = np.deg2rad(msg.head_angle)
        if self.prev_unwrap_yaw is not None and self.prev_zcam is not None:
            self.prev_unwrap_yaw = self.unwrap_angle(self.prev_unwrap_yaw, yaw_cam)
            
            z_cam = np.array([[msg.x_position], [msg.y_position], [self.prev_unwrap_yaw], [msg.Vx], [msg.Vy]])
            x_camS = float(self.smooth_data(self.prev_zcam[0], z_cam[0],0.25))
            y_camS = float(self.smooth_data(self.prev_zcam[1], z_cam[1],0.25))
            yaw_camS = float(self.smooth_data(self.prev_zcam[2], z_cam[2],1))
            wrap_yaw_camS = float(self.wrap_angle(yaw_camS))
            Vx_camS = float(self.smooth_data(self.prev_zcam[3], z_cam[3],0.25))
            Vy_camS = float(self.smooth_data(self.prev_zcam[4], z_cam[4],0.25))
            self.prev_zcam = np.array([x_camS, y_camS, wrap_yaw_camS, Vx_camS, 
                                       Vy_camS],dtype=float).reshape(5, 1)
        else:
            self.prev_unwrap_yaw = yaw_cam
            self.prev_zcam = np.array([[msg.x_position], [msg.y_position], [yaw_cam], [msg.Vx], [msg.Vy]])

        self.kf.update(self.prev_zcam , H=self.H_cam, R=self.R_cam)
        self.publish_KFstate(tc)
    
    def KF_prediction(self,dt):
        # print(dt)
        raw_msg = Imu()
        tc = rospy.Time.now()
        raw_msg.header.stamp = tc
        # dt = tc.to_nsec()*1e-9 -t0
        # t0 = tc.to_nsec()*1e-9
        # print("IMU freq:", 1/dt)
        self.accel_x, self.accel_y, self.accel_z = self.bno.linear_acceleration
        raw_msg.linear_acceleration.x = self.accel_x
        raw_msg.linear_acceleration.y = self.accel_y
        raw_msg.linear_acceleration.z = self.accel_z

        self.gyro_x, self.gyro_y, self.gyro_z = self.bno.gyro
        raw_msg.angular_velocity.x = self.gyro_x
        raw_msg.angular_velocity.y = self.gyro_y
        raw_msg.angular_velocity.z = self.gyro_z
        
        self.quat_i, self.quat_j, self.quat_k, self.quat_real = self.bno.game_quaternion
        raw_msg.orientation.w = self.quat_real
        raw_msg.orientation.x = self.quat_i
        raw_msg.orientation.y = self.quat_j
        raw_msg.orientation.z = self.quat_k

        self.pub_imu_raw.publish(raw_msg)
        
        """ IMU provides acceleration and orientation updates for prediction and update """
        
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
        u_body = np.array([[self.accel_x], 
                      [self.accel_y], 
                      [self.accel_z],
                      [self.gyro_x],
                      [self.gyro_y],
                      [self.gyro_z]])

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
        self.quaternion = (self.quat_i,self.quat_j,self.quat_k,self.quat_real,)
        roll, pitch, yaw = euler_from_quaternion(self.quaternion)

        # IMU Update
        z_imu = np.array([[roll], [pitch], [0], [0], [0]])
        self.kf.update(z_imu, H=self.H_imu, R=self.R_imu)
        
        # Predict Step
        self.kf.predict(u=u_world.flatten(), B=B)

        self.publish_KFstate(tc)
        
        
    def publish_KFstate(self,timestamp):
        """ Publish the estimated state """

        R = euler_matrix(self.kf.x[6, 0], self.kf.x[7, 0], self.kf.x[8, 0],'rxyz')  # This gives you a 4x4 matrix
        R_3x3 = R[:3, :3]  # Extract the 3x3 rotation matrix
        Vb = np.dot(R_3x3.T, self.kf.x[3:6])
        # if self.pose is not None:
        #     vb_x = self.smooth_data(self.pose[3],Vb[0,0],0.3)
        #     vb_y = self.smooth_data(self.pose[4],Vb[1,0],0.3)
        # else:
        vb_x = Vb[0,0]
        vb_y = Vb[1,0]
            
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
        state_msg.vb_x = vb_x
        state_msg.vb_y = vb_y
        state_msg.vb_z = Vb[2, 0]
        self.pub_KF.publish(state_msg)
        
        self.pose = np.array([state_msg.x, state_msg.y, state_msg.yaw, state_msg.vb_x, state_msg.vb_y])
    
    def TargetPath_callback(self, msg):
        self.path = [np.array([p.pose.position.x, p.pose.position.y]) for p in msg.poses]
        print("TargetPath_callback: Path received with", len(self.path), "points")
        
    def path_tangent_cte(self):
        # --------------------------------------------------
        # Compute CTE and theta_p from CLOSEST segment
        # --------------------------------------------------
        
        if self.pose is None or len(self.path) < 2:
            return None

        if not hasattr(self, 'last_index'):
            self.last_index = 0

        currentX, currentY= self.pose[0], self.pose[1]
        path_len = len(self.path)
        min_dist = float('inf')
        closest_i = 0

        for i in range(path_len - 1):
            x1, y1 = self.path[i]
            x2, y2 = self.path[i + 1]

            sx = x2 - x1
            sy = y2 - y1
            seg_len = np.hypot(sx, sy) + 1e-12

            # Projection of robot onto segment
            vx = currentX - x1
            vy = currentY - y1
            t = (vx * sx + vy * sy) / (seg_len ** 2)
            t = np.clip(t, 0.0, 1.0)

            proj_x = x1 + t * sx
            proj_y = y1 + t * sy

            dist = np.hypot(currentX - proj_x, currentY - proj_y)

            if dist < min_dist:
                min_dist = dist
                closest_i = i

        # Compute theta_p and cross-track error from closest segment
        x_i, y_i = self.path[closest_i]
        x_j, y_j = self.path[closest_i + 1]

        sx = x_j - x_i
        sy = y_j - y_i
        seg_len = np.hypot(sx, sy) + 1e-12

        theta_p = np.arctan2(sy, sx)

        vx = currentX - x_i
        vy = currentY - y_i
        e_fa = (sx * vy - sy * vx) / seg_len  # + left of segment

        # Handle reverse
        if not self.forward:
            self.theta_p = (theta_p + np.pi) % (2*np.pi) - np.pi
            self.e_fa = -e_fa
        else:
            self.theta_p = theta_p
            self.e_fa = e_fa

    def find_lookahead_point(self):
        if self.pose is None or len(self.path) < 2:
            return None

        if not hasattr(self, 'last_index'):
            self.last_index = 0

        currentX, currentY, Vb_x = self.pose[0], self.pose[1], self.pose[3]
        pos = np.array([currentX, currentY])
        # self.Ld = max(self.Ld_min, min(self.K_Ld*abs(Vb_x), self.Ld_max))
        path_len = len(self.path)

        # --------------------------------------------------
        # 2. Lookahead search (NO CTE computation here)
        # --------------------------------------------------
         # Reverse direction when near end or beginning
        if self.forward:
            dist_to_end = np.linalg.norm(pos - self.path[-1])
            if dist_to_end < self.Ld:
                rospy.loginfo_throttle(5.0, "Reached end. reversing direction.")
                self.forward = False
                self.last_index = path_len - 1  # second last point
        else:
            dist_to_start = np.linalg.norm(pos - self.path[0])
            if dist_to_start < self.Ld:
                rospy.loginfo_throttle(5.0, "Reached start. reversing direction.")
                self.forward = True
                self.last_index = 0
                
         # Define path segment range based on direction
        indices = range(self.last_index, path_len - 1) if self.forward else range(self.last_index, 0, -1)

        for i in indices:
            j = i + 1 if self.forward else i - 1
            x1, y1 = self.path[i][0] - currentX, self.path[i][1] - currentY
            x2, y2 = self.path[j][0] - currentX, self.path[j][1] - currentY
            dx, dy = x2 - x1, y2 - y1
            dr = np.hypot(dx, dy)
            D = x1 * y2 - x2 * y1
            discriminant = (self.Ld ** 2) * (dr ** 2) - D ** 2

            if discriminant < 0:
                continue

            sqrt_disc = np.sqrt(discriminant)
            sol_x1 = (D * dy + np.sign(dy) * dx * sqrt_disc) / dr ** 2
            sol_x2 = (D * dy - np.sign(dy) * dx * sqrt_disc) / dr ** 2
            sol_y1 = (-D * dx + abs(dy) * sqrt_disc) / dr ** 2
            sol_y2 = (-D * dx - abs(dy) * sqrt_disc) / dr ** 2

            pt1 = [sol_x1 + currentX, sol_y1 + currentY]
            pt2 = [sol_x2 + currentX, sol_y2 + currentY]

            minX = min(self.path[i][0], self.path[j][0])
            maxX = max(self.path[i][0], self.path[j][0])
            minY = min(self.path[i][1], self.path[j][1])
            maxY = max(self.path[i][1], self.path[j][1])

            in_range_1 = (minX <= pt1[0] <= maxX) and (minY <= pt1[1] <= maxY)
            in_range_2 = (minX <= pt2[0] <= maxX) and (minY <= pt2[1] <= maxY)

            if in_range_1 or in_range_2:
                # print("yes")
                # Choose better goal point
                if in_range_1 and in_range_2:
                    goalPt = pt1 if np.linalg.norm(np.array(pt1) - self.path[i + 1]) < np.linalg.norm(np.array(pt2) - self.path[i + 1]) else pt2
                elif in_range_1:
                    goalPt = pt1
                else:
                    goalPt = pt2

                if np.linalg.norm(np.array(goalPt) - self.path[j]) < np.linalg.norm(pos - self.path[j]):
                    self.last_index = i
                    # print(i)
                    return np.array(goalPt)
                else:
                    self.last_index = j
                    # print(j)
                    continue
        # print("No")
        # Fallback if nothing found
        return self.path[self.last_index]

    def set_servo_angle(self):
        pulse = self.min_pw + (self.max_pw - self.min_pw) * (self.servo_angle + (self.angle_range*0.5)) / self.angle_range
        pulse = max(self.min_pw, min(self.max_pw, pulse))
        if self.pi is not None:
            self.pi.set_servo_pulsewidth(self.SERVO_PIN, pulse)

    def update_control(self):
        while not rospy.is_shutdown():
            
            tc = rospy.Time.now()
            control_loop_time = tc.to_nsec()*1e-9
            if self.start_time is None:
                self.start_time = control_loop_time
                self.dt = 0.01
            else:
                self.t = control_loop_time - self.start_time
                self.dt = control_loop_time -self.t0
            # if self.dt > 0:
            #     print("Control loop rate (Hz):", 1 / self.dt)
            self.t0 = control_loop_time
            
            self.KF_prediction(self.dt)
            
            if self.path is not None and self.forward is True:
                # Get lookahead point
                goal = self.find_lookahead_point()
                self.path_tangent_cte()
                
                # Yaw Control (Purepursuit)
                yaw_target = np.arctan2(goal[1] - self.pose[1], goal[0] - self.pose[0])
                error_yaw = self.pose[2] - yaw_target
                error_yaw = ((error_yaw+np.pi) %(np.pi*2))-np.pi
                mid_servo = np.deg2rad(self.angle_range*0.5)
                error_vel = self.vb_des - self.pose[3]
                # self.t_avg_erroryaw = (error_yaw + self.t_avg_erroryaw*self.dt) // self.t
                a=0.75
                error_yaw = (1-a)*error_yaw + a*self.prev_error_yaw
                theta_norm = (2*self.pose[2])/(np.pi)
                u_norm = (self.pose[3])/self.vb_max
                u_des = self.vb_des/self.vb_max
                error_yaw_norm = (2*error_yaw)/(np.pi)
                error_vel_norm = error_vel/self.vb_max
                
                obs = np.array([0,theta_norm,u_norm,u_des,error_yaw_norm,error_vel_norm,0,0])
                action = self.actor.act(obs)
                
                self.aKp = 0.03
                self.aKi =0.1
                self.aKd = 0.03
                self.vKp = 2.5
                self.vKi = 0.09
                self.vKd = 0.006 
                
                self.aI = min(max(self.aI + error_yaw*self.dt, -mid_servo ) ,mid_servo )
                aD = (error_yaw - self.prev_error_yaw)/self.dt
                alpha = self.aKp*error_yaw + self.aKi*self.aI + self.aKd*aD 
                
                d_alpha = np.clip(alpha - self.prev_servo_angle, -5*self.dt, +5*self.dt)
                self.prev_servo_angle = np.clip(self.prev_servo_angle + d_alpha, -1.3,1.3)
                self.servo_angle = np.rad2deg(self.prev_servo_angle)
                self.set_servo_angle()                                      
                
                msg_pp = purepursuit_control()
                msg_pp.header.frame_id = "PID_controller"
                msg_pp.header.stamp = rospy.Time.now()
                msg_pp.goal_x = goal[0]
                msg_pp.goal_y = goal[1]
                msg_pp.Ld = self.Ld  # Lookahead distance
                msg_pp.error_yaw = error_yaw
                msg_pp.yaw_Kp = self.aKp
                msg_pp.yaw_Ki = self.aKi
                msg_pp.yaw_Kd = self.aKd
                msg_pp.servo_angle = self.servo_angle
                msg_pp.e_fa = self.e_fa
                msg_pp.theta_p = self.theta_p
                
                # Velocity Control
                if self.vel_cntrl == True:
                    self.vI = min(max(self.vI + error_vel*self.dt, 0), 1.5) 
                    vD = (error_vel - self.prev_error_vel)/self.dt
                    throttle = np.clip(self.vKp*error_vel + self.vKi*self.vI + self.vKd*vD,0.0,1.0)
                    self.prev_error_vel = error_vel
                    # self.A = min(self.a_tab) + ( max(self.a_tab) - min(self.a_tab) ) * throttle
                    # omega_sqr = (min(self.w_tab))**2 + ( (max(self.w_tab))**2 - (min(self.w_tab))**2 ) * throttle
                    # self.omega = np.sqrt(omega_sqr)
                    
                    idx = np.abs(self.throttle_grid - throttle).argmin()
                    self.A = self.A_tab.flat[idx]
                    self.omega = self.Om_tab.flat[idx]
                    
                    # msg_pp.curvature_des = k_des
                    msg_pp.velocity_x_des = self.vb_des
                    msg_pp.throttle = throttle #kappa  # Curvature
                    msg_pp.error_vel = error_vel
                    msg_pp.vel_Kp = self.vKp
                    msg_pp.vel_Ki = self.vKi
                    msg_pp.vel_Kd = self.vKd
                else:
                    self.A = 9
                    self.omega = 3
                    
                w = self.omega * 2 * np.pi
                Ramp = 1 #min((self.t)/2,1)
                self.phase = w*self.dt + self.phase
                target_vel = Ramp* self.A* w* sin(self.phase)
                self.module.set("propeller_motor_control","ctrl_velocity",target_vel)
                
                msg_pp.A = self.A
                msg_pp.omega = self.omega
                self.pub_PP.publish(msg_pp)   
                
            self.rate.sleep()     

    def shutdown_hook(self):
        """Shutdown hook to stop the motor and clean up resources."""
        rospy.loginfo("Shutdown initiated!")
        time.sleep(0.1)
        
        self.module.set("propeller_motor_control","ctrl_velocity",0)
        self.module.set("propeller_motor_control","ctrl_coast")
        print("Motor stopped and set to coast.")
        
        self.servo_angle = 0
        self.set_servo_angle()
        print("Motor stopped and set to idle.")
        if self.pi.connected:
            print("Disconnecting pigpio safely...")
            self.pi.stop()
            time.sleep(1)
        if os.system("pgrep pigpiod > /dev/null") == 0:
            os.system('sudo killall pigpiod')
            print("pigpiod daemon stopped.")
        else:
            print("pigpiod daemon was not running.")            

if __name__ == '__main__':
    try:
        controller = Fish_PurePursuitController()
        controller.update_control()
    except rospy.ROSInterruptException:
        pass

