#!/usr/bin/env python3

import rospy
import iqmotion as iq
import numpy as np
import os
import time
from math import sin, cos
import pigpio
from nav_msgs.msg import Path
from custom_msgs.msg import KFstate, motor_estimates, purepursuit_control, cam_tracking_data
from RL_base_64.actor_numpy import ActorNumpy
from RL_base_64.obs_norm_numpy import ObsNormalizer


class Fish_PurePursuitController:
    def __init__(self):
        rospy.init_node('fish_pure_pursuit_controller')

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
        self.alpha = 0
        self.v = 0
             
        #Gains
        self.actor = ActorNumpy("/home/pi/fish_ros/src/fish_robot/scripts/RL_PID/RL_base_64/newpolicy_with noouy_qh/actor_weights.npz")
        self.norm  = ObsNormalizer("/home/pi/fish_ros/src/fish_robot/scripts/RL_PID/RL_base_64/newpolicy_with noouy_qh/obs_norm.npz")
        self.lb = 0.15
        self.K_Ld = 4
        self.C_k = 0.5
        self.vb_max = 0.3
        # self.vel_cntrl = False
        
        # if self.vel_cntrl ==True:
        #     self.vKp = 1
        #     self.vKi = 1
        #     self.vKd = 0
        #     # 1. Setup the throttle tables
        #     a_tab = np.arange(5, 12, 0.25)        # 5:1:13
        #     w_tab = np.arange(1.5, 2.5, 0.25)   # 1.5:0.5:3
        #     self.A_tab, self.Om_tab = np.meshgrid(a_tab, w_tab)
        #     self.throttle_grid = 2 * np.pi * self.A_tab * (self.Om_tab**2)/750
        
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
        self.pub_enc = rospy.Publisher('/encoder_estimates', motor_estimates, queue_size=1)
        self.pub_PP = rospy.Publisher('/purepursuit_target', purepursuit_control, queue_size=1)

        # Register shutdown hook
        rospy.on_shutdown(self.shutdown_hook)

        self.rate = rospy.Rate(100)  # Loop at 30 Hz
        
    # def cam_callback(self, msg):
    #     self.pose = np.array([msg.x_position, msg.y_position, msg.head_angle, msg.body_velocity])
    #     # print("KFstate_callback: Pose set to", self.pose)
        
    def KFstate_callback(self, msg):
        # print("yes_KF")
        self.pose = np.array([msg.x, msg.y, msg.yaw, msg.vb_x, msg.vb_y])
        # print(self.pose)
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
        self.Ld = 0.4 #max(self.Ld_min, min(self.K_Ld*abs(Vb_x), self.Ld_max))
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
            
            # cross track error and theta_p
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
                self.theta_p = (theta_p + np.pi + np.pi) % (2*np.pi) - np.pi  # wrap_to_pi without helper
                self.e_fa = -e_fa
            else:
                self.theta_p = theta_p
                self.e_fa = e_fa

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

    def set_servo_angle(self,angle):
        pulse = self.min_pw + (self.max_pw - self.min_pw) * (angle + (self.angle_range*0.5)) / self.angle_range
        pulse = max(self.min_pw, min(self.max_pw, pulse))
        if self.pi is not None:
            self.pi.set_servo_pulsewidth(self.SERVO_PIN, pulse)

    def update_control(self):
        while not rospy.is_shutdown():
            # print("yes")
            if self.pose is not None and self.path:
                # print("yes")
                
                tc = rospy.Time.now()
                control_loop_time = tc.to_nsec()*1e-9
                if self.start_time is None:
                    self.start_time = control_loop_time
                    
                self.t = control_loop_time - self.start_time
                self.dt = control_loop_time -self.t0
                # if self.dt > 0:
                #     print("Control loop rate (Hz):", 1 / self.dt)
                self.t0 = control_loop_time
                
            
                #obs order: [ux, uy, qh, qh-0.75, ux-0.2, delta_prev, alpha_prev] 
                # obs = get_robot_obs()   # your sensors 
                # Get lookahead point
                goal = self.find_lookahead_point()
                pos = self.pose[:2]
                # yaw_target = np.arctan2(goal[1] - pos[1], goal[0] - pos[0])
                error_yaw = - self.theta_p + self.pose[2]
                error_yaw = ((error_yaw+np.pi) %(np.pi*2))-np.pi
                # k_des = 2*sin(error_yaw)/self.Ld
                # vb_des = self.vb_max * (1/(1+(self.C_k*abs(k_des))**2))
                vb_des = 0.25#+ 0.1*np.sign(sin(2*np.pi*0.05*self.t))
                error_vel = self.pose[3] - vb_des
                    
                # obs = np.array([self.pose[3],self.pose[-1],self.pose[2],error_yaw,error_vel,self.servo_angle,self.alpha])
                obs = np.array([self.pose[3],self.e_fa,error_yaw,self.servo_angle])
                obs = self.norm.normalize(obs)
                action = self.actor.act(obs)
                action = np.tanh(action)  # if your action space is bounded, e.g. [-1,1]
                # alpha_change = action[0]  
                delta_change = action[0]  
                # print("d_a:",alpha_change, "d_d:",delta_change, "e_yaw:", error_yaw, "e_vel:",error_vel)
                delta_change = 5*self.dt * delta_change
                self.servo_angle = self.servo_angle + delta_change
                self.servo_angle = np.clip(self.servo_angle, -1.3,1.3)
                # self.alpha =  self.alpha + alpha_change * 87000 * self.dt #13*((2*np.pi*3)**2)*np.sign(sin(2*np.pi*0.5*self.t))#
                # print(self.alpha)
                # self.alpha = np.clip(self.alpha, -4619, 4619)
                # self.v = np.clip(self.v + self.alpha*self.dt,-245,245)
                angle = np.rad2deg(self.servo_angle)
                self.set_servo_angle(angle)
                
                omega = 2
                A = 8
                w = omega * 2 * np.pi
                Ramp = 1 #min((self.t)/2,1)
                self.v = Ramp* A* w* cos(w * self.t)
                
                self.module.set("propeller_motor_control","ctrl_velocity",self.v)
                
                msg_pp = purepursuit_control()
                msg_pp.header.frame_id = "RL_controller"
                msg_pp.header.stamp = rospy.Time.now()
                msg_pp.goal_x = goal[0]
                msg_pp.goal_y = goal[1]
                msg_pp.Ld = self.Ld  # Lookahead distance
                msg_pp.error_yaw = error_yaw 
                msg_pp.yaw_Kp = self.theta_p #error_vel
                msg_pp.yaw_Ki = delta_change
                msg_pp.e_fa = self.e_fa
                msg_pp.yaw_Kd = self.v
                msg_pp.velocity_x_des = vb_des
                # msg_pp.vel_Ki = 
                # msg_pp.vel_Kd = self.alpha
                msg_pp.A = self.A
                msg_pp.omega = self.omega
                msg_pp.servo_angle = self.servo_angle
                self.pub_PP.publish(msg_pp) 
                  
                
            self.rate.sleep()     

    def shutdown_hook(self):
        """Shutdown hook to stop the motor and clean up resources."""
        rospy.loginfo("Shutdown initiated!")
        self.module.set("propeller_motor_control","ctrl_velocity",0)
        self.module.set("propeller_motor_control","ctrl_coast")
        print("Motor stopped and set to coast.")
        
        self.servo_angle = 0
        self.set_servo_angle(0)
        
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

