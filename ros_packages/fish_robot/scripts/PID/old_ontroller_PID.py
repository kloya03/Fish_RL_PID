#!/usr/bin/env python3

import math
import rospy
import iqmotion as iq
import numpy as np
import os
import time
from math import sin, cos
import pigpio
from nav_msgs.msg import Path
from custom_msgs.msg import KFstate, motor_estimates, purepursuit_control

class Fish_PurePursuitController:
    def __init__(self):
        rospy.init_node('fish_pure_pursuit_controller')

        # Parameters
        self.Ld_min = rospy.get_param('~lookahead_distance_min', 0.15)
        self.Ld_max = rospy.get_param('~lookahead_distance_max', 0.35)
        self.angle_range = rospy.get_param('~angle_range', 160)  # servo range
        self.SERVO_PIN = 12
        self.Ld = self.Ld_max
        self.min_pw = 700 
        self.max_pw = 2300
        self.servo_angle = 0
        self.t = 0
        self.start_time = None
        self.t0 = 0
        self.I=0
        self.prev_error_yaw = 0
        self.prev_error_vel = 0
        self.forward = True
        self.pose = None
        self.path = []
        
        #Gains
        self.K_Ld = 3
        self.C_k = 0.1
        self.vb_max = 0.6
        self.aKp, self.aKi, self.aKd = 1, 0, 0
        self.vKp, self.vKi, self.vKd = 1, 0, 0
        
        # 1. Setup the throttle tables
        a_tab = np.arange(5, 14, 1)        # 5:1:13
        w_tab = np.arange(1.5, 3.5, 0.5)   # 1.5:0.5:3
        self.A_tab, self.Om_tab = np.meshgrid(a_tab, w_tab)
        self.throttle_grid = 2 * np.pi * self.A_tab * (self.Om_tab**2)
        
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
        rospy.Subscriber('/KF_state_estimate', KFstate, self.KFstate_callback)
        rospy.Subscriber('/runcam_path', Path, self.TargetPath_callback)

        # Publisher
        self.pub_enc = rospy.Publisher('/encoder_estimates', motor_estimates, queue_size=10)
        self.pub_PP = rospy.Publisher('/purepursuit_target', purepursuit_control, queue_size=10)

        # Register shutdown hook
        rospy.on_shutdown(self.shutdown_hook)

        self.rate = rospy.Rate(100)  # Loop at 30 Hz
        
    def KFstate_callback(self, msg):
        self.pose = np.array([msg.x, msg.y, msg.yaw, msg.Vx])
        # print("KFstate_callback: Pose set to", self.pose)
    
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
        self.Ld =  max(self.Ld_min, min(self.K_Ld*abs(Vb_x), self.Ld_max))
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
                    self.last_index = i
                    print(i)
                    return np.array(goalPt)
                else:
                    self.last_index = j
                    print(j)
                    continue
        # print("No")
        # Fallback if nothing found
        return self.path[self.last_index]

    def set_servo_angle(self):
        pulse = self.min_pw + (self.max_pw - self.min_pw) * (self.servo_angle + (self.angle_range*0.5)) / self.angle_range
        pulse = max(self.min_pw, min(self.max_pw, pulse))
        if self.pi is not None:
            self.pi.set_servo_pulsewidth(self.SERVO_PIN, pulse)

        # self.pi.set_servo_pulsewidth(18, pulse)

    def update_control(self):
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
        error_yaw = ((error_yaw+np.pi) %(2*np.pi))-np.pi
        self.aKi = self.aKi + error_yaw*self.dt
        self.servo_angle = -self.aKp*np.rad2deg(error_yaw) #- self.aKi*self.aKi + self.aKd*(error_yaw - self.prev_error_yaw)/self.dt
        self.prev_error_yaw = error_yaw
        self.set_servo_angle()
        
        # Velocity Control
        # k_des = 2*sin(error_yaw)/self.Ld
        # vb_des = self.vb_max * (1/(1+(self.C_k*abs(k_des))**2))
        # error_vel = vb_des - self.pose[3]
        # self.vKi = self.vKi + error_vel*self.dt
        # throttle = self.vKp*error_vel + self.vKi*self.vKi + self.vKd*(error_vel - self.prev_error_vel)/self.dt
        # self.prev_error_vel = error_vel
        # idx = np.abs(self.throttle_grid - throttle).argmin()
        A = 9 # self.A_tab.flat[idx]
        omega = 2.5 # self.Om_tab.flat[idx]
        w = omega * 2 * np.pi
        Ramp = 1 #min((self.t)/2,1)
        target_vel = Ramp* A* w* cos(w * self.t)
        self.module.set("propeller_motor_control","ctrl_velocity",target_vel)

        msg_pp = purepursuit_control()
        msg_pp.header.frame_id = "P_controller "
        msg_pp.header.stamp = rospy.Time.now()
        msg_pp.goal_x = goal[0]
        msg_pp.goal_y = goal[1]
        msg_pp.Ld = self.Ld  # Lookahead distance
        # msg_pp.curvature_des = k_des
        # msg_pp.velocity_x_des = vb_des
        # msg_pp.throttle = throttle #kappa  # Curvature
        msg_pp.error_yaw = error_yaw
        # msg_pp.error_vel = error_vel
        msg_pp.yaw_Kp = self.aKp
        msg_pp.yaw_Ki = self.aKi
        msg_pp.yaw_kd = self.aKd
        # msg_pp.vel_kp = self.vKp
        # msg_pp.vel_Ki = self.vKi
        # msg_pp.vel_kd = self.vKd
        self.pub_PP.publish(msg_pp)        

        # Publish encoder data
        enc_msg = motor_estimates()
        enc_msg.header.stamp = rospy.Time.now()
        # enc_msg.PWM = self.module.get("propeller_motor_control", "ctrl_pwm")
        # enc_msg.supply_volts = self.module.get("propeller_motor_control", "ctrl_volts")
        # enc_msg.encoder_vel = self.module.get("brushless_drive", "obs_velocity")   # Using x to store velocity
        # enc_msg.encoder_pos = self.module.get("brushless_drive","obs_angle")  # Using x to store velocity
        enc_msg.A = A
        enc_msg.omega = omega
        # enc_msg.voltage = self.module.get("power_monitor", "volts") #tm.Vbus.magnitude
        # enc_msg.current = self.module.get("power_monitor","amps") #tm.Iq["estimate"].magnitude
        # enc_msg.power = self.module.get("power_monitor","amps")
        enc_msg.servo_angle = self.servo_angle
        self.pub_enc.publish(enc_msg)

    def shutdown_hook(self):
        """Shutdown hook to stop the motor and clean up resources."""
        rospy.loginfo("Shutdown initiated!")
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

    def run(self):
        while not rospy.is_shutdown():
            
            if self.pose is not None and self.path:
                
                self.update_control()
            self.rate.sleep()

if __name__ == '__main__':
    try:
        controller = Fish_PurePursuitController()
        controller.run()
    except rospy.ROSInterruptException:
        pass

