#!/usr/bin/env python3
import rospy
import numpy as np
import tf.transformations as tft
from custom_msgs.msg import imu_euler, cam_tracking_data, KFstate


# ==============================
# Helpers
# ==============================

def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi


def quat_normalize(q):
    return q / np.linalg.norm(q)


def quat_mul(q1, q2):  # wxyz
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def exp_quat_from_rotvec(dtheta):
    th = np.linalg.norm(dtheta)
    if th < 1e-12:
        return quat_normalize(np.array([1.0, 0.5*dtheta[0], 0.5*dtheta[1], 0.5*dtheta[2]]))
    axis = dtheta/th
    half = 0.5*th
    return np.array([np.cos(half), *(axis*np.sin(half))])


def quat_to_R(q):
    q_xyzw = [q[1], q[2], q[3], q[0]]
    return tft.quaternion_matrix(q_xyzw)[0:3,0:3]


def quat_to_euler(q):
    q_xyzw = [q[1], q[2], q[3], q[0]]
    return np.array(tft.euler_from_quaternion(q_xyzw))


def quat_yaw(q):
    return quat_to_euler(q)[2]


# ==============================
# EKF Node
# ==============================

class FishEKF:

    def __init__(self):

        rospy.init_node("fish_ekf")

        # State: [px py pz vx vy vz qw qx qy qz]
        self.x = np.zeros((10,1))
        self.x[6] = 1.0

        self.P = np.eye(10)*0.5

        self.sigma_a = 0.6
        self.sigma_g = 0.05

        self.R_rp = np.diag([0.02,0.02])
        self.R_cam = np.diag([0.05,0.05,0.2,0.2,0.05])

        self.last_imu_time = None

        self.pub = rospy.Publisher("/KF_state", KFstate, queue_size=10)

        rospy.Subscriber("/IMU_bno08x/raw", imu_euler, self.imu_callback, queue_size=200)
        rospy.Subscriber("/runcam_tracking", cam_tracking_data, self.cam_callback)

        rospy.loginfo("Fish EKF running...")

    # ==============================
    # IMU PREDICT
    # ==============================
    def imu_callback(self, msg):

        t = msg.header.stamp.to_sec()

        if self.last_imu_time is None:
            self.last_imu_time = t
            return

        dt = t - self.last_imu_time
        self.last_imu_time = t

        if dt <= 0:
            return

        p = self.x[0:3,0]
        v = self.x[3:6,0]
        q = quat_normalize(self.x[6:10,0])

        a_b = np.array([msg.linear_acceleration.x,
                        msg.linear_acceleration.y,
                        msg.linear_acceleration.z])

        w_b = np.array([msg.angular_velocity.x,
                        msg.angular_velocity.y,
                        msg.angular_velocity.z])

        dq = exp_quat_from_rotvec(w_b*dt)
        q = quat_normalize(quat_mul(q,dq))

        R = quat_to_R(q)
        a_w = R @ a_b  # assumes gravity removed

        p = p + v*dt + 0.5*a_w*dt*dt
        v = v + a_w*dt

        self.x[0:3,0] = p
        self.x[3:6,0] = v
        self.x[6:10,0] = q

        # Covariance
        F = np.eye(10)
        F[0:3,3:6] = np.eye(3)*dt

        Qa = (self.sigma_a**2)*np.eye(3)
        Q = np.zeros((10,10))
        Q[0:3,0:3] = 0.25*(dt**4)*Qa
        Q[3:6,3:6] = (dt**2)*Qa
        Q[6:10,6:10] = (dt**2)*(self.sigma_g**2)*np.eye(4)

        self.P = F @ self.P @ F.T + Q

        # Roll/Pitch update
        self.update_roll_pitch(msg)

        self.publish_state(msg.header.stamp)

    # ==============================
    # ROLL/PITCH UPDATE
    # ==============================
    def update_roll_pitch(self,msg):

        q_meas = np.array([
            msg.orientation.w,
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z])

        q_meas = quat_normalize(q_meas)

        rpy_meas = quat_to_euler(q_meas)[0:2].reshape((2,1))

        q = quat_normalize(self.x[6:10,0])
        rpy_pred = quat_to_euler(q)[0:2].reshape((2,1))

        y = rpy_meas - rpy_pred
        y[0] = wrap_pi(y[0])
        y[1] = wrap_pi(y[1])

        H = np.zeros((2,10))
        H[:,6:10] = self.numeric_rp_jacobian(q)

        S = H @ self.P @ H.T + self.R_rp
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(10) - K @ H) @ self.P

        self.x[6:10,0] = quat_normalize(self.x[6:10,0])

    def numeric_rp_jacobian(self,q):

        eps = 1e-6
        J = np.zeros((2,4))
        rp0 = quat_to_euler(q)[0:2]

        for i in range(4):
            dq = np.zeros(4)
            dq[i] = eps
            q2 = quat_normalize(q+dq)
            rp2 = quat_to_euler(q2)[0:2]
            J[:,i] = (rp2-rp0)/eps

        return J

    # ==============================
    # CAMERA UPDATE
    # ==============================
    def cam_callback(self,msg):

        z = np.array([
            msg.x_position,
            msg.y_position,
            msg.Vx,
            msg.Vy,
            np.deg2rad(msg.head_angle)
        ]).reshape((5,1))

        p = self.x[0:3,0]
        v = self.x[3:6,0]
        q = quat_normalize(self.x[6:10,0])
        yaw = quat_yaw(q)

        h = np.array([p[0],p[1],v[0],v[1],yaw]).reshape((5,1))

        y = z-h
        y[4] = wrap_pi(y[4])

        H = np.zeros((5,10))
        H[0,0]=1
        H[1,1]=1
        H[2,3]=1
        H[3,4]=1
        H[4,6:10]=self.numeric_yaw_jacobian(q)

        S = H @ self.P @ H.T + self.R_cam
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(10)-K@H)@self.P

        self.x[6:10,0] = quat_normalize(self.x[6:10,0])

    def numeric_yaw_jacobian(self,q):

        eps = 1e-6
        J = np.zeros((1,4))
        y0 = quat_yaw(q)

        for i in range(4):
            dq = np.zeros(4)
            dq[i]=eps
            q2 = quat_normalize(q+dq)
            J[0,i]=(wrap_pi(quat_yaw(q2)-y0))/eps

        return J

    # ==============================
    # Publish
    # ==============================
    def publish_state(self,stamp):

        msg = KFstate()
        msg.header.stamp = stamp

        p = self.x[0:3,0]
        v = self.x[3:6,0]
        q = quat_normalize(self.x[6:10,0])

        R = quat_to_R(q)
        vb = R.T @ v

        rpy = quat_to_euler(q)

        msg.x, msg.y, msg.z = p
        msg.Vx, msg.Vy, msg.Vz = v
        msg.vb_x, msg.vb_y, msg.vb_z = vb
        msg.roll, msg.pitch, msg.yaw = rpy

        self.pub.publish(msg)


if __name__ == "__main__":
    FishEKF()
    rospy.spin()

