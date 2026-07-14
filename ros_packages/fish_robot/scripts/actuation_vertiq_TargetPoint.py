#!/usr/bin/env python3

import rospy
import numpy as np
import os
import can
import time
from math import sin, cos
import iqmotion as iq 
import pigpio
from custom_msgs.msg import KFstate, motor_estimates
from geometry_msgs.msg import PoseStamped
import threading

class Fish_PurePursuitController:
    def __init__(self):
        rospy.init_node('fish_pure_pursuit_controller')
        # Parameters
        self.A = rospy.get_param('~A',6)
        self.omega = rospy.get_param('~omega',1)
        self.servo_angle = 0
        self.t = 0
        self.start_time = None
        self.t0 = 0
        self.I=0
        self.prev_error_yaw = 0
        self.forward = True
        self.goal[0] = 0
        self.goal[1] = 0
        print(self.A)
        # Hardware setup
       # --- IQ module ---
        self.com = iq.SerialCommunicator("/dev/ttyS0", baudrate=115200)
        self.module = iq.SpeedModule(self.com, 0)
        print("Connected to Vertiq Motor")
        
        self.module.set("propeller_motor_control", "timeout",60)
        print(self.module.get("power_monitor", "volts"))
        
        self.min_pulse_width = 700
        self.max_pulse_width = 2300
        self.angle_range = 148
        
        print("Starting pigpiod...")
        if os.system("pgrep pigpiod > /dev/null") != 0:
            os.system('sudo pigpiod')
        time.sleep(1)
        self.pi = pigpio.pi()
        if not self.pi.connected:
            rospy.logerr("Failed to connect to pigpiod daemon!")
            rospy.signal_shutdown("pigpiod connection failed")
            return
        self.pi.set_PWM_frequency(18, 333)
        self.set_servo_angle()

        # Subscribers
        self.pose = None
        rospy.Subscriber('/KF_state_estimate', KFstate, self.KFstate_callback)
        rospy.Subscriber('/purepursuit_target', PoseStamped, self.Target_callback)

        # Publisher
        self.pub_enc = rospy.Publisher('/encoder_estimates', motor_estimates, queue_size=10)

        # Register shutdown hook
        rospy.on_shutdown(self.shutdown_hook)

        self.rate = rospy.Rate(100)  # Loop at 30 Hz
        
    def KFstate_callback(self, msg):
        self.pose = np.array([msg.x, msg.y, msg.yaw, msg.vb_x, msg.vb_y])
        # print("KFstate_callback: Pose set to", self.pose)
    
    def Target_callback(self, msg):
        self.goal[0] = msg.pose.position.x
        self.goal[1] = msg.pose.position.y


    def set_servo_angle(self):
        pulse = self.min_pulse_width + (self.max_pulse_width  - self.min_pulse_width) * (self.servo_angle + (self.angle_range*0.5)) / self.angle_range
        pulse = max(self.min_pulse_width, min(self.max_pulse_width , pulse))
        if self.pi is not None:
            self.pi.set_servo_pulsewidth(18, pulse)

        # self.pi.set_servo_pulsewidth(18, pulse)
        
    def enc_publish(self):
        rate = rospy.Rate(50)  # publish at 50 Hz (adjust)
        while not rospy.is_shutdown():
            enc_msg = motor_estimates()
                # publish estimates
            enc_msg.header.stamp = rospy.Time.now()

            def f(x, default=0.0):
                try: return float(x)
                except (TypeError, ValueError): 
                    return float(default)

            enc_msg.PWM = f(self.module.get("propeller_motor_control", "ctrl_pwm"))
            # print((enc_msg.PWM), (self.module.get("multi_turn_angle_control", "ctrl_pwm")))
            enc_msg.supply_volts = f(self.module.get("propeller_motor_control", "ctrl_volts"))
            enc_msg.encoder_vel = f(self.module.get("brushless_drive", "obs_velocity"))
            # enc_msg.encoder_pos = f(self.module.get("multi_turn_angle_control", "obs_angular_displacement"))
            enc_msg.voltage = f(self.module.get("power_monitor", "volts"))
            enc_msg.current = f(self.module.get("power_monitor", "amps"))
            enc_msg.power   = f(self.module.get("power_monitor", "watts"))

            enc_msg.servo_angle = float(self.servo_angle)
            enc_msg.A = float(self.A)
            enc_msg.omega = float(self.omega)

            self.pub_enc.publish(enc_msg)
            rate.sleep()

    def update_control(self):
        
        while not rospy.is_shutdown():
            rate = rospy.Rate(100)
            tc = rospy.Time.now()
            control_loop_time = tc.to_nsec()*1e-9
            if self.start_time is None:
                self.start_time = control_loop_time
                
            self.t = control_loop_time - self.start_time
            self.dt = control_loop_time -self.t0
            # if self.dt > 0:
                # print("Control loop rate (Hz):", 1 / self.dt)
            self.t0 = control_loop_time
            
            # Get lookahead point
            pos = self.pose[:2]
            U_vel = np.sqrt(self.pose[3]**2 + self.pose[4]**2)

            # geometric error P control Set servo angle
            yaw_target = np.degrees(np.arctan2(self.goal[1] - pos[1], self.goal[0] - pos[0]))
            error_yaw = yaw_target-self.pose[2]
            error_yaw = ((error_yaw+180) %360)-180
            Kp = 1
            Ki = 0
            Kd = 0
            self.I = self.I + error_yaw*self.dt
            target_servo_angle = -Kp*error_yaw - Ki*self.I + Kd*(error_yaw - self.prev_error_yaw)/self.dt
            self.prev_error_yaw = error_yaw
            self.servo_angle = target_servo_angle
            # print(target_servo_angle,error_yaw,self.dt)
            self.set_servo_angle()
            # self.omega = 2
            # self.A = 6
            w = self.omega * 2 * np.pi
            Ramp = 1 #min((self.t)/2,1)
            target_vel = Ramp * self.A * w * cos(w * self.t)

            print(1/self.dt) 
            self.module.set("propeller_motor_control", "ctrl_velocity", target_vel)
            # self.module.set("multi_turn_angle_control", "ctrl_pwm", max(min(target_PWM,0.4),-0.4))
            rate.sleep()

    def shutdown_hook(self):
        try:
            self.set_servo_angle(0)
            self.module.set("propeller_motor_control", "ctrl_cosat")
            rospy.loginfo("Motor stopped and set to coast.")
        except Exception as e:
            rospy.logwarn(f"Shutdown motor command failed: {e}")

        os.system('sudo killall pigpiod')
        rospy.loginfo("pigpiod daemon stopped.")
        time.sleep(1)

if __name__ == '__main__':
    try:
        controller = Fish_PurePursuitController()
        key_listener_thread = threading.Thread(target=controller.enc_publish)
        key_listener_thread.daemon = True
        key_listener_thread.start()
        controller.update_control()
    except rospy.ROSInterruptException:
        pass
