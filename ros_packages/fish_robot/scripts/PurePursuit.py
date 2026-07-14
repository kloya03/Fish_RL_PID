#!/usr/bin/env python3

import rospy
import numpy as np
from math import sin, cos
from nav_msgs.msg import Path
from custom_msgs.msg import KFstate, purepursuit_control

class FishPurePursuitController:
    def __init__(self):
        rospy.init_node('fish_pure_pursuit_controller')

        # Parameters
        self.Ld = rospy.get_param('~lookahead_distance', 0.30)
        # self.use_vector_pursuit = rospy.get_param('~use_vector_pursuit', False)
        self.t = 0
        self.start_time = None
        self.t0 = 0
        self.forward = True
        self.vKp, self.vKi, self.vKd = 1, 2, 3
        self.aKp, self.aKi, self.aKd = 0.5, 1.5, 2.5
        # Subscribers
        self.pose = None
        self.path = []
        rospy.Subscriber('/KF_state_estimate', KFstate, self.KFstate_callback)
        rospy.Subscriber('/runcam_path', Path, self.TargetPath_callback)

        # Publisher
        self.pub_PP = rospy.Publisher('/purepursuit_target', purepursuit_control, queue_size=10)

        # Register shutdown hook
        rospy.on_shutdown(self.shutdown_hook)
        rospy.loginfo(rospy.get_caller_id() + "  Lookahead calculation launched.")

        self.rate = rospy.Rate(100)  # Loop at 30 Hz
        
    def KFstate_callback(self, msg):
        self.pose = np.array([msg.x, msg.y, msg.yaw])
        # print("KFstate_callback: Pose set to", self.pose)
    
    def TargetPath_callback(self, msg):
        self.path = [np.array([p.pose.position.x, p.pose.position.y]) for p in msg.poses]
        print("TargetPath_callback: Path received with", len(self.path), "points")

    def find_lookahead_point(self):
        if self.pose is None or len(self.path) < 2:
            return None

        if not hasattr(self, 'last_index'):
            self.last_index = 0

        currentX, currentY = self.pose[0], self.pose[1]
        pos = np.array([currentX, currentY])
        lookAheadDis = self.Ld
        path_len = len(self.path)
            
         # Reverse direction when near end or beginning
        if self.forward:
            dist_to_end = np.linalg.norm(pos - self.path[-1])
            if dist_to_end < self.Ld:
                rospy.loginfo_throttle(5.0, "Reached end — reversing direction.")
                self.forward = False
                self.last_index = path_len - 1  # second last point
        else:
            dist_to_start = np.linalg.norm(pos - self.path[0])
            if dist_to_start < self.Ld:
                rospy.loginfo_throttle(5.0, "Reached start — reversing direction.")
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
            discriminant = (lookAheadDis ** 2) * (dr ** 2) - D ** 2

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

    def publish_target(self):
        t0 = 0
        while not rospy.is_shutdown():
            if self.pose is not None and self.path:
                tc = rospy.Time.now()
                dt = tc.to_nsec()*1e-9 - t0
                t0 = tc.to_nsec()*1e-9
                dt = np.array(dt)
                # print("PP_freq:", 1/dt)
                # Get lookahead point
                goal = self.find_lookahead_point()
                msg_pp = purepursuit_control()
                msg_pp.header.frame_id = "PurePursuit_controller"
                msg_pp.header.stamp = rospy.Time.now()
                msg_pp.goal_x = goal[0]
                msg_pp.goal_y = goal[1]
                msg_pp.Ld = self.Ld  # Lookahead distance
                msg_pp.curvature_des = 1 #k_des
                msg_pp.velocity_x_des = 2 #vb_des
                msg_pp.throttle = 3 #throttle #kappa  # Curvature
                msg_pp.error_yaw = 0 # error_yaw
                msg_pp.error_vel = 0 #error_vel
                msg_pp.yaw_Kp = self.aKp
                msg_pp.yaw_Ki = self.aKi
                msg_pp.yaw_Kd = self.aKd
                msg_pp.vel_Kp = self.vKp
                msg_pp.vel_Ki = self.vKi
                msg_pp.vel_Kd = self.vKd
                self.pub_PP.publish(msg_pp)    
            self.rate.sleep()      

    def shutdown_hook(self):
        rospy.loginfo("Shutdown initiated! No Target points.")

if __name__ == '__main__':
    try:
        controller = FishPurePursuitController()
        controller.publish_target()
    except rospy.ROSInterruptException:
        rospy.loginfo(rospy.get_caller_id() + "  Lookahead calculation exited with exception.")
        pass


## different code for finding Look ahead target point

    # def find_lookahead_point(self):
    #     if self.pose is None or len(self.path) < 2:
    #         return None

    #     if not hasattr(self, 'last_index'):
    #         self.last_index = 0

    #     currentX, currentY = self.pose[0], self.pose[1]
    #     pos = np.array([currentX, currentY])
    #     lookAheadDis = self.Ld
    #     lastFoundIndex = self.last_index
    #     path_len = len(self.path)

        # for offset in range(path_len):
        #     i = (lastFoundIndex + offset) % (path_len - 1)
        #     x1, y1 = self.path[i][0] - currentX, self.path[i][1] - currentY
        #     x2, y2 = self.path[i+1][0] - currentX, self.path[i+1][1] - currentY
        #     dx, dy = x2 - x1, y2 - y1
        #     dr = np.hypot(dx, dy)
        #     D = x1 * y2 - x2 * y1
        #     discriminant = (lookAheadDis**2) * (dr**2) - D**2

        #     if discriminant < 0:
        #         continue

        #     sqrt_disc = np.sqrt(discriminant)
        #     sol_x1 = (D * dy + np.sign(dy) * dx * sqrt_disc) / dr**2
        #     sol_x2 = (D * dy - np.sign(dy) * dx * sqrt_disc) / dr**2
        #     sol_y1 = (-D * dx + abs(dy) * sqrt_disc) / dr**2
        #     sol_y2 = (-D * dx - abs(dy) * sqrt_disc) / dr**2

        #     pt1 = [sol_x1 + currentX, sol_y1 + currentY]
        #     pt2 = [sol_x2 + currentX, sol_y2 + currentY]

        #     minX = min(self.path[i][0], self.path[i+1][0])
        #     maxX = max(self.path[i][0], self.path[i+1][0])
        #     minY = min(self.path[i][1], self.path[i+1][1])
        #     maxY = max(self.path[i][1], self.path[i+1][1])

        #     in_range_1 = (minX <= pt1[0] <= maxX) and (minY <= pt1[1] <= maxY)
        #     in_range_2 = (minX <= pt2[0] <= maxX) and (minY <= pt2[1] <= maxY)

        #     if in_range_1 or in_range_2:
        #         # Choose better point (closer to next path point)
        #         if in_range_1 and in_range_2:
        #             goalPt = pt1 if np.linalg.norm(np.array(pt1) - self.path[(i+1)%path_len]) < np.linalg.norm(np.array(pt2) - self.path[(i+1)%path_len]) else pt2
        #         elif in_range_1:
        #             goalPt = pt1
        #         else:
        #             goalPt = pt2

        #         if np.linalg.norm(np.array(goalPt) - self.path[(i+1)%path_len]) < np.linalg.norm(pos - self.path[(i+1)%path_len]):
        #             self.last_index = i
        #             return np.array(goalPt)
        #         else:
        #             self.last_index = (i + 1) % path_len
        #             continue

        # # Optional lap complete detection
        # if np.linalg.norm(pos - self.path[0]) < self.Ld:
        #     rospy.loginfo_throttle(5.0, "Lap complete!")

        # # Fallback: return closest known path point
        # return self.path[self.last_index]