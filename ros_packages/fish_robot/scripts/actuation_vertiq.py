#!/usr/bin/env python3

import os
import time
import subprocess
import rospy
import pigpio
import iqmotion as iq
from geometry_msgs.msg import Vector3
from custom_msgs.msg import motor_estimates,Fish_ControlCmd


class MotorNode:
    def __init__(self):
        # --- IQ module ---
        self.com = iq.SerialCommunicator("/dev/ttyS0", baudrate=115200)
        self.module = iq.ServoModule(self.com, 0)
        self.module.set("multi_turn_angle_control", "angular_speed_max", 300)
        self.module.save("multi_turn_angle_control", "angular_speed_max")
        self.module.set("multi_turn_angle_control", "timeout",60)
        self.module.save("multi_turn_angle_control", "timeout")
        print(self.module.get("power_monitor", "volts"))
        print(self.module.get("multi_turn_angle_control", "angular_speed_max"))
        print(self.module.get("multi_turn_angle_control", "timeout"))
        # --- pigpio ---
        rospy.loginfo("Starting pigpiod...")
        subprocess.Popen(['sudo', 'pigpiod'])
        time.sleep(1)

        self.servo_pin = 18
        self.pig = pigpio.pi()
        self.pig.set_PWM_frequency(self.servo_pin, 333)
        if not self.pig.connected:
            rospy.logwarn("Failed to connect to pigpio daemon. Servo will not work")

        self.min_pulse_width = 700
        self.max_pulse_width = 2200
        self.angle_range = 148

        # --- command variables (NO GLOBALS) ---
        self.target_vel = 0.0
        self.target_pos = 0.0
        self.servo_angle = 0.0

        # optional fields you were publishing (define so it doesn't crash)
        self.A = 0.0
        self.omega = 0.0

        # --- ROS pub/sub ---
        self.pub_enc = rospy.Publisher('/encoder_estimates', motor_estimates, queue_size=10)
        self.sub_cmd = rospy.Subscriber('/cmd_control', Fish_ControlCmd, self.cmd_cb, queue_size=10)

        self.set_servo_angle(self.servo_angle)
        rospy.on_shutdown(self.on_shutdown)

    def cmd_cb(self, msg: Vector3):
        # /cmd_control: x=target_vel, y=target_pos, z=servo_angle
        self.target_vel = float(msg.target_vel)
        self.target_pos = float(msg.target_pos)
        self.servo_angle = float(msg.servo_angle)
        self.A = float(msg.A)
        self.omega = float(msg.omega)

    def set_servo_angle(self, angle):
        pulse_width = self.min_pulse_width + (self.max_pulse_width - self.min_pulse_width) * (
            angle + self.angle_range * 0.5
        ) / self.angle_range

        if self.pig.connected:
            self.pig.set_servo_pulsewidth(self.servo_pin, pulse_width)

    def spin(self):
        rate = rospy.Rate(3000)
        while not rospy.is_shutdown():
            # motor velocity command (same as your original)
            self.module.set("multi_turn_angle_control", "ctrl_velocity", self.target_vel)

            # servo command
            self.set_servo_angle(self.servo_angle)

            # publish estimates
            enc_msg = motor_estimates()
            enc_msg.header.stamp = rospy.Time.now()
            enc_msg.encoder_vel = self.module.get("multi_turn_angle_control", "obs_angular_velocity")
            enc_msg.encoder_pos = self.module.get("multi_turn_angle_control", "obs_angular_displacement")
            enc_msg.A = self.A
            enc_msg.omega = self.omega
            enc_msg.voltage = self.module.get("power_monitor", "volts")
            enc_msg.current = self.module.get("power_monitor", "amps")
            enc_msg.power = self.module.get("power_monitor", "watts")
            enc_msg.servo_angle = self.servo_angle

            self.pub_enc.publish(enc_msg)
            rate.sleep()

    def on_shutdown(self):
        try:
            self.module.set("multi_turn_angle_control", "ctrl_velocity", 0)
            self.module.set("multi_turn_angle_control", "ctrl_coast")
            rospy.loginfo("Motor stopped and set to coast.")
        except Exception as e:
            rospy.logwarn(f"Shutdown motor command failed: {e}")

        os.system('sudo killall pigpiod')
        rospy.loginfo("pigpiod daemon stopped.")
        time.sleep(1)


if __name__ == '__main__':
    rospy.init_node('motors', anonymous=True)
    node = MotorNode()
    node.spin()

