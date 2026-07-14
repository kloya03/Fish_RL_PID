#!/usr/bin/env python3
import time
import rospy
from smbus2 import SMBus
from custom_msgs.msg import ms5837_pressure

# -------------------------------
# I2C / MUX settings
# -------------------------------
I2C_BUS = 1
MUX_ADDR = 0x70
SENSOR_ADDR = 0x76
MUX_CHANNELS = [0, 1, 2, 3]

# -------------------------------
# ms5837 constants (matching your ms5837.py)
# -------------------------------
CMD_RESET = 0x1E
CMD_ADC_READ = 0x00
CMD_PROM_READ = 0xA0
CMD_CONV_D1_256 = 0x40
CMD_CONV_D2_256 = 0x50

# Oversampling: keep default behavior (DO NOT change)
OSR_8192 = 5

# From your lib: sleep(2.5e-6 * 2**(8+oversampling))
CONV_SLEEP_S = 2.5e-6 * (2 ** (8 + OSR_8192))  # 0.02048s

# Fluid density
DENSITY_FRESHWATER = 997

# -------------------------------
# Globals
# -------------------------------
bus = None
_current_mux_channel = None


def mux_select(channel: int):
    """Select a TCA9548A mux channel. Only writes if channel changes."""
    global _current_mux_channel
    if _current_mux_channel == channel:
        return
    bus.write_byte(MUX_ADDR, 1 << channel)
    _current_mux_channel = channel
    time.sleep(0.0005)


def read_word_swapped(addr, reg):
    """Match your library's word read + endian swap."""
    w = bus.read_word_data(addr, reg)
    return ((w & 0xFF) << 8) | (w >> 8)


def crc4(n_prom7):
    """
    Same CRC4 routine style as your ms5837.py (but without mutating caller list).
    Expects 7 PROM words: C0..C6.
    CRC nibble is in high nibble of C0: (C0 & 0xF000)>>12
    """
    n_prom = list(n_prom7)
    n_rem = 0
    n_prom[0] = n_prom[0] & 0x0FFF
    n_prom.append(0)

    for i in range(16):
        if i % 2 == 1:
            n_rem ^= (n_prom[i >> 1] & 0x00FF)
        else:
            n_rem ^= (n_prom[i >> 1] >> 8)

        for _ in range(8):
            if n_rem & 0x8000:
                n_rem = (n_rem << 1) ^ 0x3000
            else:
                n_rem = (n_rem << 1)
            n_rem &= 0xFFFF

    n_rem = (n_rem >> 12) & 0x000F
    return n_rem ^ 0x00


def init_channel(channel: int):
    """Reset + read PROM (C0..C6) + CRC check. Returns calibration list C[0..6]."""
    mux_select(channel)
    bus.write_byte(SENSOR_ADDR, CMD_RESET)
    time.sleep(0.01)

    C = []
    for i in range(7):
        C.append(read_word_swapped(SENSOR_ADDR, CMD_PROM_READ + 2 * i))

    crc_read = (C[0] & 0xF000) >> 12
    crc_calc = crc4(C)
    if crc_read != crc_calc:
        rospy.logwarn(f"PROM CRC failed on mux channel {channel}: read={crc_read} calc={crc_calc}")
    return C


def start_d1(channel: int):
    mux_select(channel)
    bus.write_byte(SENSOR_ADDR, CMD_CONV_D1_256 + 2 * OSR_8192)


def start_d2(channel: int):
    mux_select(channel)
    bus.write_byte(SENSOR_ADDR, CMD_CONV_D2_256 + 2 * OSR_8192)


def read_adc(channel: int) -> int:
    mux_select(channel)
    d = bus.read_i2c_block_data(SENSOR_ADDR, CMD_ADC_READ, 3)
    return (d[0] << 16) | (d[1] << 8) | d[2]


def calculate_30ba(D1: int, D2: int, C):
    """
    Copy of your MS5837._calculate() for MODEL_30BA.
    Returns: (pressure_mbar, temp_C)
    """
    dT = D2 - C[5] * 256
    SENS = C[1] * 32768.0 + (C[3] * dT) / 256.0
    OFF  = C[2] * 65536.0 + (C[4] * dT) / 128.0
    TEMP = 2000.0 + dT * C[6] / 8388608.0  # centi-degC

    Ti = 0.0
    OFFi = 0.0
    SENSi = 0.0

    if (TEMP / 100.0) < 20.0:
        Ti = (3.0 * dT * dT) / 8589934592.0
        OFFi = (3.0 * (TEMP - 2000.0) * (TEMP - 2000.0)) / 2.0
        SENSi = (5.0 * (TEMP - 2000.0) * (TEMP - 2000.0)) / 8.0
        if (TEMP / 100.0) < -15.0:
            OFFi = OFFi + 7.0 * (TEMP + 1500.0) * (TEMP + 1500.0)
            SENSi = SENSi + 4.0 * (TEMP + 1500.0) * (TEMP + 1500.0)
    else:
        Ti = 2.0 * (dT * dT) / 137438953472.0
        OFFi = (1.0 * (TEMP - 2000.0) * (TEMP - 2000.0)) / 16.0
        SENSi = 0.0

    OFF2 = OFF - OFFi
    SENS2 = SENS - SENSi
    TEMP2 = TEMP - Ti  # centi-degC

    P_mbar = (((D1 * SENS2) / 2097152.0 - OFF2) / 8192.0) / 10.0
    temp_C = TEMP2 / 100.0
    return float(P_mbar), float(temp_C)


def pressure_to_depth_m(pressure_mbar: float):
    """Same as your lib depth(): (P(Pa)-101300)/(rho*g)"""
    P_pa = pressure_mbar * 100.0
    return (P_pa - 101300.0) / (DENSITY_FRESHWATER * 9.80665)


def main():
    global bus
    rospy.init_node("ms5837_publisher")

    pub = rospy.Publisher("/ms5837_PressureSensor", ms5837_pressure, queue_size=10)

    rate_hz = 100.0  # loop target; actual limited by sensor conversion timing
    depth_bias = 0.270

    bus = SMBus(I2C_BUS)

    rospy.loginfo("Initializing 4 pressure sensors (pipelined)...")

    # Read calibration for each channel
    calib = {ch: init_channel(ch) for ch in MUX_CHANNELS}

    rate = rospy.Rate(rate_hz)
    t0 = rospy.Time.now().to_sec()

    try:
        rospy.loginfo("ms5837 pressure sensors node is publishing.")
        while not rospy.is_shutdown():
            # ---------------------------
            # Pipeline D1 (pressure)
            # ---------------------------
            for ch in MUX_CHANNELS:
                start_d1(ch)
            time.sleep(CONV_SLEEP_S)

            D1 = {ch: read_adc(ch) for ch in MUX_CHANNELS}

            # ---------------------------
            # Pipeline D2 (temperature)
            # ---------------------------
            for ch in MUX_CHANNELS:
                start_d2(ch)
            time.sleep(CONV_SLEEP_S)

            # ---------------------------
            # Read D2 + compute
            # ---------------------------
            results = {}
            for ch in MUX_CHANNELS:
                D2 = read_adc(ch)
                p_mbar, t_C = calculate_30ba(D1[ch], D2, calib[ch])
                depth_m = pressure_to_depth_m(p_mbar) + depth_bias
                results[ch] = (p_mbar, depth_m, t_C)

            # ---------------------------
            # Fill custom ROS message
            # ---------------------------
            msg = ms5837_pressure()
            msg.header.stamp = rospy.Time.now()
            msg.header.frame_id = "ms5837_pressure"

            msg.pressure_0, msg.depth_0, msg.temp_0, msg.bias_0 = map(float, (*results[0], depth_bias))
            msg.pressure_1, msg.depth_1, msg.temp_1, msg.bias_1 = map(float, (*results[1], depth_bias))
            msg.pressure_2, msg.depth_2, msg.temp_2, msg.bias_2 = map(float, (*results[2], depth_bias))
            msg.pressure_3, msg.depth_3, msg.temp_3, msg.bias_3 = map(float, (*results[3], depth_bias))

            pub.publish(msg)

            # ---------------------------
            # Loop frequency print
            # ---------------------------
            ct = rospy.Time.now().to_sec()
            dt = ct - t0
            t0 = ct
            # if dt > 0:
                # print("ms_pressure loop freq:", 1.0 / dt)

            rate.sleep()

    finally:
        try:
            bus.close()
        except Exception:
            rospy.loginfo("ms5837 pressure sensors has stopped publishing.")
            pass


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo(rospy.get_caller_id() + "  ms5837 pressure sensors node exited with exception.")
