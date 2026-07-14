#!/usr/bin/env python3

import math
import rospy
import iqmotion as iq
import threading
import numpy as np
import os
import time
from math import sin, cos
import pigpio
from nav_msgs.msg import Path
from custom_msgs.msg import KFstate, motor_estimates, purepursuit_control, cam_tracking_data

class Fish_PurePursuitController:
    def __init__(self):
        rospy.init_node('fish_pure_pursuit_controller')
        self.running = True
        # Parameters
        self.Ld_min = rospy.get_param('~lookahead_distance_min', 0.25)
        self.Ld_max = rospy.get_param('~lookahead_distance_max', 0.4)
        self.angle_range = rospy.get_param('~angle_range', 160)  # servo range
        self.SERVO_PIN = 12
        self.Ld = self.Ld_max
        self.min_pw = 700 
        self.max_pw = 2300
        self.servo_angle = 0
        self.t = 0
        self.start_time = None
        self.t0 = 0
        self.vI=0
        self.prev_error_yaw = 0
        self.prev_error_vel = 0
        self.forward = True
        self.pose = None
        self.path = []
        self.A = 0
        self.omega = 0
        self.theta_p = 0
        self.e_fa = 0   
             
        #Gains
        self.lb = 0.15
        self.K_Ld = 4
        self.C_k = 0.5
        self.vb_max = 0.3
        self.vel_cntrl = False
        
        if self.vel_cntrl ==True:
            self.vKp = 1
            self.vKi = 1
            self.vKd = 0
            # 1. Setup the throttle tables
            a_tab = np.arange(5, 12, 0.25)        # 5:1:13
            w_tab = np.arange(1.5, 2.5, 0.25)   # 1.5:0.5:3
            self.A_tab, self.Om_tab = np.meshgrid(a_tab, w_tab)
            self.throttle_grid = 2 * np.pi * self.A_tab * (self.Om_tab**2)/750
        
        # Hardware setup
        com = iq.SerialCommunicator("/dev/ttyS0",baudrate=115200)
        #module = iq.Vertiq2306(com,module_idn=0)
        self.module = iq.Vertiq4006(com,0)
        voltage_now = self.module.get("power_monitor","volts")
        print(voltage_now)
        
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
        # rospy.Subscriber('/runcam_tracking', cam_tracking_data, self.cam_callback)
        rospy.Subscriber('/KF_state_estimate', KFstate, self.KFstate_callback)
        rospy.Subscriber('/runcam_path', Path, self.TargetPath_callback)

        # Publisher
        self.pub_enc = rospy.Publisher('/encoder_estimates', motor_estimates, queue_size=10)
        self.pub_PP = rospy.Publisher('/purepursuit_target', purepursuit_control, queue_size=10)

        # Register shutdown hook
        rospy.on_shutdown(self.shutdown_hook)

        self.rate = rospy.Rate(100)  # Loop at 30 Hz
        
    # def cam_callback(self, msg):
    #     self.pose = np.array([msg.x_position, msg.y_position, msg.head_angle, msg.body_velocity])
    #     # print("KFstate_callback: Pose set to", self.pose)
        
    def KFstate_callback(self, msg):
        self.pose = np.array([msg.x, msg.y, msg.yaw, msg.vb_x])
    #     # print("KFstate_callback: Pose set to", self.pose)
    
    def TargetPath_callback(self, msg):
        self.path = [np.array([p.pose.position.x, p.pose.position.y]) for p in msg.poses]
        print("TargetPath_callback: Path received with", len(self.path), "points")

    def find_lookahead_point(self):
        if self.pose is None or len(self.path) < 2:
            return None

        if not hasattr(self, 'last_index'):
            self.last_index = 0

        currentX, currentY, Vb_x = self.pose[0], self.pose[1], self.pose[3]
        pos = np.array([currentX, currentY])
        self.Ld = max(self.Ld_min, min(self.K_Ld*abs(Vb_x), self.Ld_max))
        path_len = len(self.path)
            
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

        for offset in indices:
            i = offset
            
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
                    
                    yaw = self.pose[2]
                    x_i, y_i = self.path[i][0], self.path[i][1]
                    x_j, y_j = self.path[j][0], self.path[j][1]
                    sx = x_j - x_i
                    sy = y_j - y_i
                    seg_len = np.hypot(sx, sy) + 1e-12
                    theta_p = np.arctan2(sy, sx)
                    vx = currentX - x_i
                    vy = currentY - y_i
                    e_fa = (sx * vy - sy * vx) / seg_len   # +left of segment direction
                    # reverse mode: flip tangent and error so controller stays consistent
                    if not self.forward:
                        theta_p = (theta_p + np.pi + np.pi) % (2*np.pi) - np.pi  # wrap_to_pi without helper
                        e_fa = -e_fa
                    # e_theta = (theta_p - yaw + np.pi) % (2*np.pi) - np.pi
                    # store for the controller
                    self.theta_p = theta_p
                    self.e_fa = e_fa
                    # self.e_theta = e_theta

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
            if self.pose is not None and self.path:
                tc = rospy.Time.now()
                control_loop_time = tc.to_nsec()*1e-9
                if self.start_time is None:
                    self.start_time = control_loop_time
                    
                self.t = control_loop_time - self.start_time
                self.dt = control_loop_time -self.t0
                if self.dt > 0:
                    print("Control loop rate (Hz):", 1 / self.dt)
                self.t0 = control_loop_time
                
                # Get lookahead point
                goal = self.find_lookahead_point()
                # print(goal)
                pos = self.pose[:2]
                
                # Yaw Control
                yaw_target = np.arctan2(goal[1] - pos[1], goal[0] - pos[0])
                error_yaw = yaw_target-self.pose[2]
                error_yaw = ((error_yaw+np.pi) %(np.pi*2))-np.pi
                self.servo_angle  = -np.rad2deg(math.atan2(2*self.lb*sin(error_yaw),self.Ld))
                self.prev_error_yaw = error_yaw
                self.set_servo_angle()
                
                msg_pp = purepursuit_control()
                msg_pp.header.frame_id = "PurePursuit_controller"
                msg_pp.header.stamp = rospy.Time.now()
                msg_pp.goal_x = goal[0]
                msg_pp.goal_y = goal[1]
                msg_pp.Ld = self.Ld  # Lookahead distance
                msg_pp.error_yaw = error_yaw
                msg_pp.yaw_Kp = self.K_Ld
                msg_pp.yaw_Ki = self.lb
                msg_pp.e_fa = self.e_fa
                msg_pp.yaw_Kd = self.theta_p
                msg_pp.servo_angle = self.servo_angle

                # Velocity Control
                if self.vel_cntrl == True:
                    k_des = 2*sin(error_yaw)/self.Ld
                    vb_des = self.vb_max * (1/(1+(self.C_k*abs(k_des))**2))
                    error_vel = vb_des - self.pose[3]
                    self.vI = min(max(self.vI + error_vel*self.dt, 0), 1) 
                    throttle = self.vKp*error_vel + self.vKi*self.vI + self.vKd*(error_vel - self.prev_error_vel)/self.dt
                    self.prev_error_vel = error_vel
                    idx = np.abs(self.throttle_grid - throttle).argmin()
                    self.A = self.A_tab.flat[idx]
                    self.omega = self.Om_tab.flat[idx]
                    
                    msg_pp.curvature_des = k_des
                    msg_pp.velocity_x_des = vb_des
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
                target_vel = Ramp* self.A* w* cos(w * self.t)
                self.module.set("propeller_motor_control","ctrl_velocity",target_vel)

                msg_pp.A = self.A
                msg_pp.omega = self.omega
                self.pub_PP.publish(msg_pp)

            self.rate.sleep()     

    def shutdown_hook(self):
        """Shutdown hook to stop the motor and clean up resources."""
        rospy.loginfo("Shutdown initiated!")
        # Stop threads FIRST
        self.running = False
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

