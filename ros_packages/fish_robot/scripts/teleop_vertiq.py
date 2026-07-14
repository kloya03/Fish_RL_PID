#!/usr/bin/env python3

import subprocess
import os
import can
import datetime
import iqmotion as iq
from math import sin,cos
import numpy as np
import pigpio
import time
import rospy
import readchar
from custom_msgs.msg import motor_estimates
from std_msgs.msg import Bool
import threading
import subprocess
import os
import board
import adafruit_bno08x
from digitalio import DigitalInOut
import board
import busio
from adafruit_bno08x.i2c import BNO08X_I2C

i2c = busio.I2C(board.SCL, board.SDA)
reset_pin = DigitalInOut(board.D5)
bno = BNO08X_I2C(i2c, reset=reset_pin, debug=False)

# bno.begin_calibration()
# TODO: UPDATE UART/SPI
bno.enable_feature(adafruit_bno08x.BNO_REPORT_ROTATION_VECTOR)
bno.enable_feature(adafruit_bno08x.BNO_REPORT_LINEAR_ACCELERATION)
bno.enable_feature(adafruit_bno08x.BNO_REPORT_GYROSCOPE)

com = iq.SerialCommunicator("/dev/ttyS0",baudrate=115200)
#module = iq.Vertiq2306(com,module_idn=0)
module = iq.Vertiq4006(com,0)
voltage_now = module.get("power_monitor","volts")

print(voltage_now)

# Initialize pigpio
print("Starting pigpiod...")
subprocess.Popen(['sudo', 'pigpiod'])
# Wait a bit to ensure pigpiod starts up
time.sleep(1)
servo_pin = 12  # Replace with your GPIO pin number
pig = pigpio.pi()
pig.set_PWM_frequency(servo_pin, 333)
if not pig.connected:
    print("Failed to connect to pigpio daemon. Is it running? Servo will not work")

min_pulse_width = 700  # Minimum pulse width in microseconds
max_pulse_width = 2200  # Maximum pulse width in microseconds
angle_range = 160  # Range of servo angle in degrees
key_pressed = None

global_imuTime=[]
time_data=[]
current_datetime = datetime.datetime.now()
timestamped_filename_imu = f"Fish_data_test{current_datetime.strftime('%Y%m%d_%H%M')}.csv"

angular_velocity_data = {'x': [], 'y': [], 'z': []}
linear_acceleration_data = {'x': [], 'y': [], 'z': []}
quat_data = {'i': [], 'j': [], 'k': [], 'real':[]}

def set_servo_angle(angle):
    """Sets the servo to a specific angle."""
    pulse_width = min_pulse_width + (max_pulse_width - min_pulse_width) * (angle+angle_range*0.5) / angle_range
    pig.set_servo_pulsewidth(servo_pin, pulse_width)

def key_listener():
    global key_pressed
    try:
        while True:
            key_pressed = readchar.readkey()
    except KeyboardInterrupt:
        print("Key listener thread interrupted.")
        rospy.signal_shutdown("Key listener interrupted")

# Teleoperation control Servo Motor
servo_angle = 0
set_servo_angle(servo_angle)

global_imuTime=[]
time_data=[]
current_datetime = datetime.datetime.now()

rad2tick=8192/(2*np.pi)
deg2rad=np.pi/180
Hz2w=2*np.pi
A=0
omega = 0
w=omega*Hz2w
dt=0
ts = 0
t0 = time.time()
prev_ts = 0
State = False
def main():
    global key_pressed
# try:
    while not rospy.is_shutdown():
        global A, w, dt, ts, prev_ts, t0, servo_angle, angle_range, omega, State
        pub_enc = rospy.Publisher('/encoder_estimates', motor_estimates, queue_size=10)
        pub_cam_cmd = rospy.Publisher('/camera', Bool, queue_size=10)
        rate = rospy.Rate(1000)
        
        if key_pressed:
            if key_pressed == 'a':
                servo_angle = max(servo_angle - 10, -angle_range*0.5)
                set_servo_angle(servo_angle)
                print(f"Set servo angle to {servo_angle} degrees.")
            elif key_pressed == 'd':
                servo_angle = min(servo_angle + 10, angle_range*0.5)
                set_servo_angle(servo_angle)
                print(f"Set servo angle to {servo_angle} degrees.")
            elif key_pressed == 'w':
                A = min(A+1,13)
                print('A =',A)
            elif key_pressed == 's':
                A = max(A-1,0)
                print('A =',A)
            elif key_pressed == 'x':
                omega = min(omega+0.5,3)
                print('Omega =',omega)
            elif key_pressed == 'z':
                omega = max(omega-0.5,0)
                print('Omega =',omega)
            elif key_pressed == 'q':
                module.set("propeller_motor_control","ctrl_velocity",0)
                module.set("propeller_motor_control","ctrl_coast")
                print("Vertiq Motor set to coast")
            elif key_pressed == 'c':
                cam_msg = Bool()
                State = not State
                cam_msg = Bool(data=State)
                pub_cam_cmd.publish(cam_msg)
                rospy.loginfo("camera capture set to %s",cam_msg.data)
                
            key_pressed = None  # Reset the key press
                
        ts=time.time()- t0
        dt = ts-prev_ts
        prev_ts = ts
        w=omega*Hz2w
        gg = min((ts)/5,1)
        v=gg*A*w*cos(w*ts)
        target_vel=v
        p=gg*A*sin(w*ts)
        # target_pos=p
        module.set("propeller_motor_control","ctrl_velocity",target_vel)

        enc_msg = motor_estimates()
        enc_msg.header.stamp = rospy.Time.now()
        # enc_msg.PWM = self.module.get("propeller_motor_control", "ctrl_pwm")
        # enc_msg.supply_volts = self.module.get("propeller_motor_control", "ctrl_volts")
        enc_msg.encoder_vel = module.get("brushless_drive", "obs_velocity")   # Using x to store velocity
        enc_msg.encoder_pos = module.get("brushless_drive","obs_angle")  # Using x to store velocity
        enc_msg.A = A
        enc_msg.omega = omega
        enc_msg.voltage = module.get("power_monitor", "volts") #tm.Vbus.magnitude
        enc_msg.current = module.get("power_monitor","amps") #tm.Iq["estimate"].magnitude
        enc_msg.power = module.get("power_monitor","amps")
        enc_msg.servo_angle = servo_angle
        pub_enc.publish(enc_msg)

        rate.sleep()

if __name__ == '__main__':
    try:
        rospy.init_node('motors', anonymous=True)
        # Start the key listener thread
        key_listener_thread = threading.Thread(target=key_listener)
        key_listener_thread.daemon = True
        key_listener_thread.start()
        main()


    except rospy.ROSInterruptException:
        pass
    finally:
        module.set("propeller_motor_control","ctrl_velocity",0)
        module.set("propeller_motor_control","ctrl_coast")
        print("Motor stopped and set to coast.")
        os.system('sudo killall pigpiod')
        print("pigpiod daemon stopped.")
        time.sleep(1)
        

