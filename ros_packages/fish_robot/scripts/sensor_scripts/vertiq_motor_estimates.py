#!/usr/bin/env python3

import rospy
import iqmotion as iq
from custom_msgs.msg import motor_estimates, purepursuit_control

class Motor_publisher:
    def __init__(self):
        rospy.init_node('vertiq_motor_publisher')
        self.A = 0
        self.omega = 0
        self.servo_angle = 0
        self.start_time = None
        self.prev_t = 0
        
        # Hardware setup
        com = iq.SerialCommunicator("/dev/ttyS0", baudrate=115200)
        self.module = iq.Vertiq4006(com)
        voltage_now = self.module.get("power_monitor", "volts")
        print(voltage_now)
        # Publisher
        self.pub_enc = rospy.Publisher('/encoder_estimates', motor_estimates, queue_size=10)
        
        # subscriber
        rospy.Subscriber('/purepursuit_target', purepursuit_control, self.controller_callback)

        rospy.on_shutdown(self.shutdown_hook)

        # Fast loop (increase if needed)
        rospy.loginfo(rospy.get_caller_id() + "  vertiq motor publisher node launched.")
        self.rate = rospy.Rate(50)   # 500 Hz
    
    def controller_callback(self,msg):
        self.A = msg.A
        self.omega = msg.omega
        self.servo_angle = msg.servo_angle

    def publish(self):
        while not rospy.is_shutdown():

            voltage_now = self.module.get("power_monitor", "volts")
            # print(voltage_now)
            tc = rospy.Time.now()
            loop_time = tc.to_nsec()*1e-9
            if self.start_time is None:
                self.start_time = loop_time
                    
            self.t = loop_time - self.start_time
            dt = loop_time -self.prev_t
            # if dt > 0:
            #     print("motor loop rate (Hz):", 1 / dt)
            self.prev_t = loop_time
                
            # Only run when motor is ON
            if isinstance(voltage_now, float):

                supply_volts = self.module.get("propeller_motor_control", "ctrl_volts")
                PWM = self.module.get("propeller_motor_control", "ctrl_pwm")
                e_vel = self.module.get("brushless_drive", "obs_velocity")
                e_pos = self.module.get("brushless_drive", "obs_angle")
                current = self.module.get("power_monitor", "amps")
                power = self.module.get("power_monitor","amps")
                temperature = self.module.get("temperature_monitor_uc","uc_temp")
                # print(self.module.get_all("temperature_monitor_uc"))
                
                if all(isinstance(x, float) for x in
                       [PWM, supply_volts, e_vel, e_pos, voltage_now, current, power, temperature]):

                    enc_msg = motor_estimates()
                    enc_msg.header.stamp = rospy.Time.now()
                    enc_msg.temperature = temperature
                    enc_msg.PWM = PWM
                    enc_msg.supply_volts = supply_volts
                    enc_msg.encoder_vel = e_vel
                    enc_msg.encoder_pos = e_pos
                    enc_msg.voltage = voltage_now
                    enc_msg.current = current
                    enc_msg.power = power
                    enc_msg.A = self.A
                    enc_msg.omega = self.omega
                    enc_msg.servo_angle = self.servo_angle

                    self.pub_enc.publish(enc_msg)

            # Always sleep to maintain high stable rate
            self.rate.sleep()

    def shutdown_hook(self):
        
        rospy.loginfo("Shutdown initiated!")
        rospy.loginfo(rospy.get_caller_id() + "  vertiq motor publisher node shutdown.")


if __name__ == '__main__':
    try:
        motor_pub = Motor_publisher()
        motor_pub.publish()
    except rospy.ROSInterruptException:
        pass
