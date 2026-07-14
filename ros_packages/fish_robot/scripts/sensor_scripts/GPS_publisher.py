#!/usr/bin/env python3
import rospy
import serial
import time
from std_msgs.msg import Bool
from custom_msgs.msg import GpsTracking

class GPS_Tracker:
    
    def __init__(self):
        rospy.init_node("gps_tracking_node")
        self.PORT = "/dev/ttyACM0"   # often /dev/ttyUSB0 or /dev/ttyACM0 on Linux
        self.BAUD = 115200

        self.pub = rospy.Publisher("/gps_tracking", GpsTracking, queue_size=10)
        self.sub = rospy.Subscriber("/camera", Bool, self.camera_cb)
        self.ser = serial.Serial(self.PORT, self.BAUD)

        self.rate = rospy.Rate(100)
        
    def main(self):
        while not rospy.is_shutdown():

            # Read whatever the ESP32 sends back (line-based)
            line = self.ser.readline().decode('utf-8',errors='replace').strip()
            # print(line)
            if line.startswith('#GPS:'):
                data=line[5:]
                parts = data.split(',')
                msg = GpsTracking()

                msg.latitude = float(parts[0])
                msg.longitude = float(parts[1])
                msg.speed = float(parts[2])
                msg.heading_deg = float(parts[3])
                msg.hdop = float(parts[4])
                msg.satellites = int(parts[5])
                msg.year = int(parts[6])
                msg.month = int(parts[7])
                msg.day = int(parts[8])

                msg.hour = int(parts[9])
                msg.minute = int(parts[10])
                msg.second = int(parts[11])
                msg.img_count = int(parts[12])

                self.pub.publish(msg)

            self.rate.sleep()
                    
                    
    def camera_cb(self, msg):
        while not rospy.is_shutdown():
            self.cmd = msg.data
            self.ser.write(self.cmd.encode("ascii"))
            self.ser.flush()
            

if __name__ == '__main__':
    try:
        node = GPS_Tracker()
        node.main()
    except rospy.ROSInterruptException:
        pass
    
    
    
    
    
# with serial.Serial(PORT, BAUD) as ser:
#     # time.sleep()  # give ESP32 time to reboot after opening port (common)
#     while True:
#         cmd = "30 6.28\n"
#         ser.write(cmd.encode("ascii"))
#         ser.flush()

#         # Read whatever the ESP32 sends back (line-based)
#         line = ser.readline().decode(errors="replace").strip()
#         if line:
#             print("ESP32:", line)

#         time.sleep(10)



#!/usr/bin/env python3


# from micropyGPS import MicropyGPS
# from custom_msgs.msg import GpsTracking

# def main():
#     rospy.init_node("gps_tracking_node")
#     PORT = "/dev/ttyUSB0"   # often /dev/ttyUSB0 or /dev/ttyACM0 on Linux
#     BAUD = 9600

#     pub = rospy.Publisher("/gps_tracking", GpsTracking, queue_size=10)

#     ser = serial.Serial(PORT, BAUD, timeout=1)
#     gps = MicropyGPS(location_formatting='dd')

#     rate = rospy.Rate(100)

#     while not rospy.is_shutdown():
#         while ser.in_waiting:
#             for c in ser.read().decode('ascii', errors='ignore'):
#                 gps.update(c)

#         if gps.fix_stat > 0:
#             msg = GpsTracking()

#             msg.latitude = gps.latitude[0]
#             msg.longitude = gps.longitude[0]
#             msg.speed_mps = gps.speed['mps']
#             msg.heading_deg = gps.course
#             msg.hdop = gps.hdop

#             msg.year = gps.date[2]
#             msg.month = gps.date[1]
#             msg.day = gps.date[0]

#             msg.hour = gps.timestamp[0]
#             msg.minute = gps.timestamp[1]
#             msg.second = int(gps.timestamp[2])

#             pub.publish(msg)
#             print()

#         rate.sleep()

# if __name__ == "__main__":
#     main()
