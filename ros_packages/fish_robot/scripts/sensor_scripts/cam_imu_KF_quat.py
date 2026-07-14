#!/usr/bin/env python3
import rospy
import numpy as np
from custom_msgs.msg import cam_tracking_data, KFstate
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion

def wrap_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def quat_normalize(q):
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n

def quat_mul(q1, q2):
    # q = q1 * q2, quaternion as [w, x, y, z]
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def rotvec_to_quat(dtheta):
    th = np.linalg.norm(dtheta)
    if th < 1e-12:
        # small angle: [1, 0.5*dtheta]
        return quat_normalize(np.array([1.0, 0.5*dtheta[0], 0.5*dtheta[1], 0.5*dtheta[2]]))
    axis = dtheta / th
    half = 0.5 * th
    return np.array([np.cos(half), *(axis*np.sin(half))])

def quat_to_R(q):
    # world-from-body rotation matrix, q=[w,x,y,z]
    w,x,y,z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)],
    ])

def quat_yaw(q):
    # yaw from quaternion (XYZ roll-pitch-yaw convention)
    w,x,y,z = q
    siny_cosp = 2*(w*z + x*y)
    cosy_cosp = 1 - 2*(y*y + z*z)
    return np.arctan2(siny_cosp, cosy_cosp)

def yaw_jacobian_numeric(q, eps=1e-6):
    # dyaw/dq (1x4)
    y0 = quat_yaw(q)
    J = np.zeros((1,4))
    for i in range(4):
        dq = np.zeros(4); dq[i] = eps
        q2 = quat_normalize(q + dq)
        J[0,i] = wrap_pi(quat_yaw(q2) - y0) / eps
    return J

def rp_jacobian_numeric(q, eps=1e-6):
    # d[roll,pitch]/dq (2x4)
    r0,p0,_ = euler_from_quaternion([q[1],q[2],q[3],q[0]])
    J = np.zeros((2,4))
    for i in range(4):
        dq = np.zeros(4); dq[i] = eps
        q2 = quat_normalize(q + dq)
        r2,p2,_ = euler_from_quaternion([q2[1],q2[2],q2[3],q2[0]])
        dr = wrap_pi(r2 - r0)
        dp = wrap_pi(p2 - p0)
        J[:,i] = np.array([dr, dp]) / eps
    return J

class SensorFusionKF:
    """
    Quaternion EKF:
      - class with imu_callback (predict + optional tilt update)
      - cam_callback (x,y,yaw update)
      - publish_state (publish X,Y,Z,Vx,Vy,Vz + Euler + body vel)

    State (10):
      x = [px,py,pz, vx,vy,vz, qw,qx,qy,qz]^T

    Notes:
      - yaw is corrected ONLY from camera (no magnetometer)
      - roll/pitch can be corrected from IMU orientation (tilt) if you want
      - IMU dt computed ONLY from IMU stamps.
    """

    def __init__(self):
        rospy.init_node("sensor_fusion_kf_quat", anonymous=True)

        self.n = 10
        self.x = np.zeros((self.n,1))
        self.x[6,0] = 1.0  # qw

        self.P = np.eye(self.n) * 1.0

        # Tunable Parameters
        self.sigma_a = 1.0   # m/s^2
        self.sigma_g = 0.10  # rad/s

        self.R_cam = 1e-6*np.diag([0.05, 0.05, 0.10])  # camera [x,y,yaw]
        self.R_rp  = np.diag([(2*np.pi/180)**2, (2*np.pi/180)**2])  # roll/pitch meas

        self.USE_IMU_TILT_UPDATE = True  # roll/pitch only (NOT yaw)

        self.last_imu_time = None

        self.max_dt = 0.2  # clamp big dt spikes

        self.imu_sub = rospy.Subscriber("/IMU_bno08x/raw", Imu, self.imu_callback, queue_size=50)
        self.cam_sub = rospy.Subscriber("/runcam_tracking", cam_tracking_data, self.cam_callback, queue_size=50)

        self.pub = rospy.Publisher("/KF_state_estimate", KFstate, queue_size=10)
        rospy.loginfo(rospy.get_caller_id() + " quat EKF cam-imu fusion launched.")

    # -------------------------- EKF core --------------------------
    def ekf_predict(self, F, Q, x_pred):
        self.x = x_pred
        self.P = F @ self.P @ F.T + Q

    def ekf_update(self, z, h, H, R, angle_wrap_idx=None):
        y = z - h
        if angle_wrap_idx:
            for i in angle_wrap_idx:
                y[i,0] = wrap_pi(float(y[i,0]))

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        I = np.eye(self.n)
        # Joseph form
        self.P = (I - K@H) @ self.P @ (I - K@H).T + K @ R @ K.T

        # normalize quaternion after any update
        q = self.x[6:10,0]
        self.x[6:10,0] = quat_normalize(q)

    # -------------------------- IMU callback --------------------------
    def imu_callback(self, msg: Imu):
        t = msg.header.stamp.to_sec()
        if self.last_imu_time is None:
            self.last_imu_time = t
            return

        dt = t - self.last_imu_time
        self.last_imu_time = t
        if not np.isfinite(dt) or dt <= 0.0:
            return
        if dt > self.max_dt:
            dt = self.max_dt

        # Read IMU
        a_b = np.array([msg.linear_acceleration.x,
                        msg.linear_acceleration.y,
                        msg.linear_acceleration.z], dtype=float)
        w_b = np.array([msg.angular_velocity.x,
                        msg.angular_velocity.y,
                        msg.angular_velocity.z], dtype=float)

        # Current state
        p = self.x[0:3,0]
        v = self.x[3:6,0]
        q = quat_normalize(self.x[6:10,0])

        # --- propagate quaternion with gyro ---
        dq = rotvec_to_quat(w_b * dt)
        q_pred = quat_normalize(quat_mul(q, dq))

        # --- propagate p,v using accel rotated to world ---
        Rwb = quat_to_R(q_pred)
        a_w = Rwb @ a_b  # assumes gravity removed

        p_pred = p + v*dt + 0.5*a_w*(dt**2)
        v_pred = v + a_w*dt

        x_pred = np.zeros((self.n,1))
        x_pred[0:3,0] = p_pred
        x_pred[3:6,0] = v_pred
        x_pred[6:10,0] = q_pred

        # --- F (simple) ---
        F = np.eye(self.n)
        F[0:3,3:6] = np.eye(3)*dt

        # --- Q (simple) ---
        Qa = (self.sigma_a**2) * np.eye(3)
        Q = np.zeros((self.n,self.n))
        Q[0:3,0:3] = 0.25*(dt**4)*Qa
        Q[0:3,3:6] = 0.5*(dt**3)*Qa
        Q[3:6,0:3] = 0.5*(dt**3)*Qa
        Q[3:6,3:6] = (dt**2)*Qa
        Q[6:10,6:10] = (dt**2)*(self.sigma_g**2)*np.eye(4)

        self.ekf_predict(F, Q, x_pred)

        # --- Optional: update roll/pitch from IMU orientation (tilt), ignore yaw ---
        if self.USE_IMU_TILT_UPDATE:
            qimu = msg.orientation
            q_meas = np.array([qimu.w, qimu.x, qimu.y, qimu.z], dtype=float)
            if np.all(np.isfinite(q_meas)) and np.linalg.norm(q_meas) > 1e-6:
                q_meas = quat_normalize(q_meas)
                roll_m, pitch_m, _ = euler_from_quaternion([q_meas[1],q_meas[2],q_meas[3],q_meas[0]])

                z = np.array([[roll_m],[pitch_m]])

                q_now = quat_normalize(self.x[6:10,0])
                roll_p, pitch_p, _ = euler_from_quaternion([q_now[1],q_now[2],q_now[3],q_now[0]])
                h = np.array([[roll_p],[pitch_p]])

                H = np.zeros((2,self.n))
                H[:,6:10] = rp_jacobian_numeric(q_now)

                self.ekf_update(z, h, H, self.R_rp, angle_wrap_idx=[0,1])

        self.publish_state(msg.header.stamp)

    # -------------------------- Camera callback --------------------------
    def cam_callback(self, msg: cam_tracking_data):
        if np.isnan(msg.x_position) or np.isnan(msg.y_position) or np.isnan(msg.head_angle):
            rospy.logwarn("NaN camera data, skipping update.")
            return

        z = np.array([[msg.x_position],
                      [msg.y_position],
                      [msg.head_angle]], dtype=float)

        # predicted measurement h(x) = [px, py, yaw(q)]
        p = self.x[0:3,0]
        q = quat_normalize(self.x[6:10,0])
        yaw = quat_yaw(q)

        h = np.array([[p[0]],
                      [p[1]],
                      [yaw]], dtype=float)

        H = np.zeros((3,self.n))
        H[0,0] = 1.0
        H[1,1] = 1.0
        H[2,6:10] = yaw_jacobian_numeric(q)

        # wrap only yaw innovation
        self.ekf_update(z, h, H, self.R_cam, angle_wrap_idx=[2])

        self.publish_state(msg.header.stamp)

    # -------------------------- Publish --------------------------
    def publish_state(self, stamp):
        q = quat_normalize(self.x[6:10,0])
        Rwb = quat_to_R(q)
        v_world = self.x[3:6,0]
        v_body = Rwb.T @ v_world

        roll, pitch, yaw = euler_from_quaternion([q[1],q[2],q[3],q[0]])
        yaw = wrap_pi(yaw)

        state_msg = KFstate()
        state_msg.header.stamp = stamp

        state_msg.x  = float(self.x[0,0])
        state_msg.y  = float(self.x[1,0])
        state_msg.z  = float(self.x[2,0])

        state_msg.Vx = float(self.x[3,0])
        state_msg.Vy = float(self.x[4,0])
        state_msg.Vz = float(self.x[5,0])

        state_msg.roll  = float(roll)
        state_msg.pitch = float(pitch)
        state_msg.yaw   = float(yaw)

        state_msg.vb_x = float(v_body[0])
        state_msg.vb_y = float(v_body[1])
        state_msg.vb_z = float(v_body[2])

        self.pub.publish(state_msg)

if __name__ == "__main__":
    try:
        SensorFusionKF()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
