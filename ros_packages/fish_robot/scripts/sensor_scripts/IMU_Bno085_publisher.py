#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import MagneticField,Imu
from std_msgs.msg import Float64
from diagnostic_msgs.msg import DiagnosticStatus, DiagnosticArray
import time
import board
import busio
from digitalio import DigitalInOut
import adafruit_bno08x
from adafruit_bno08x.i2c import BNO08X_I2C


def bno08x_node():
    # calibration_good_at = None
    # Initialize ROS node
    raw_pub = rospy.Publisher('IMU_bno08x/raw', Imu, queue_size=50)
    # mag_pub = rospy.Publisher('IMU_bno08x/mag', MagneticField, queue_size=10)
    # status_pub = rospy.Publisher('IMU_bno08x/status', DiagnosticArray, queue_size=30)
    rospy.init_node('Imu_Bno085_publisher')
    rate = rospy.Rate(100) # frequency in Hz
    rospy.loginfo(rospy.get_caller_id() + "  bno08x node launched.")
    # reset_pin = DigitalInOut(board.D5)
    i2c = busio.I2C(board.SCL, board.SDA)
    bno = BNO08X_I2C(i2c,address=0x4a) # BNO080 (0x4b) BNO085 (0x4a)


    bno.enable_feature(adafruit_bno08x.BNO_REPORT_LINEAR_ACCELERATION)
    bno.enable_feature(adafruit_bno08x.BNO_REPORT_GYROSCOPE)
    bno.enable_feature(adafruit_bno08x.BNO_REPORT_MAGNETOMETER)
    # bno.enable_feature(adafruit_bno08x.BNO_REPORT_ROTATION_VECTOR)
    bno.enable_feature(adafruit_bno08x.BNO_REPORT_GAME_ROTATION_VECTOR)
    
    ## Calibration
    # bno.begin_calibration()
    # while True:
    #     if not calibration_good_at and bno.calibration_status >= 2:
    #         calibration_good_at = time.monotonic()
    #         print(
    #         "Magnetometer Calibration quality:",
    #         adafruit_bno08x.REPORT_ACCURACY_STATUS[bno.calibration_status],
    #         " (%d)" % bno.calibration_status)
    #     if calibration_good_at and (time.monotonic() - calibration_good_at > 5.0):
    #         print("calibration done")
    #         bno.save_calibration_data()
    #         break
    time.sleep(0.1) # ensure IMU is initialized
    t0=0

    while not rospy.is_shutdown():
        raw_msg = Imu()
        tc = rospy.Time.now()
        raw_msg.header.stamp = tc
        dt = tc.to_nsec()*1e-9 -t0
        # print("IMU freq:", 1/dt)
        accel_x, accel_y, accel_z = bno.linear_acceleration
        raw_msg.linear_acceleration.x = accel_x
        raw_msg.linear_acceleration.y = accel_y
        raw_msg.linear_acceleration.z = accel_z

        gyro_x, gyro_y, gyro_z = bno.gyro
        raw_msg.angular_velocity.x = gyro_x
        raw_msg.angular_velocity.y = gyro_y
        raw_msg.angular_velocity.z = gyro_z
        
        quat_i, quat_j, quat_k, quat_real = bno.game_quaternion
        raw_msg.orientation.w = quat_real
        raw_msg.orientation.x = quat_i
        raw_msg.orientation.y = quat_j
        raw_msg.orientation.z = quat_k
        # print(1/dt)
        
        # raw_msg.orientation_covariance[0] = -1
        # raw_msg.linear_acceleration_covariance[0] = -1
        # raw_msg.angular_velocity_covariance[0] = -1

        raw_pub.publish(raw_msg)
        t0 = tc.to_nsec()*1e-9
        # mag_msg = MagneticField()
        # mag_x, mag_y, mag_z = bno.magnetic
        # mag_msg.header.stamp = rospy.Time.now()
        # mag_msg.magnetic_field.x = mag_x
        # mag_msg.magnetic_field.y = mag_y
        # mag_msg.magnetic_field.z = mag_z
        # mag_msg.magnetic_field_covariance[0] = -1
        # mag_pub.publish(mag_msg)
        
        # status_msg = DiagnosticArray()
        # status_msg.header.stamp = raw_msg.header.stamp
        # status = DiagnosticStatus()
        # status.level = bno.calibration_status
        # status.name = "bno08x IMU"
        # status.message = ""
        # status_msg.status.append(status)
        # status_pub.publish(status_msg)

        rate.sleep()   
        
            # rospy.loginfo(rospy.get_caller_id() + "  bno08x node finished")

if __name__ == '__main__':
    try:
        
        bno08x_node()
    except rospy.ROSInterruptException:
        rospy.loginfo(rospy.get_caller_id() + "  bno08x node exited with exception.")
