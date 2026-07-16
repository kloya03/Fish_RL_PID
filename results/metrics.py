from rosbags.highlevel import AnyReader
from pathlib import Path
import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
base_dir = Path("/home/ravi/iros_video")

paths = ["I", "R", "O", "S"]
velocities = ["1.0", "1.5", "2.0", "2.5"]

# =========================
# Storage
# =========================
results = []

# =========================
# Loop
# =========================
for p in paths:
    for v in velocities:

        folder = base_dir / p / v
        bag_files = list(folder.glob("*.bag"))

        if not bag_files:
            print(f"⚠ No bags found for {p} at {v}")
            continue

        ct_err_vals = []
        hd_err_vals = []
        speed_err_vals = []

        with AnyReader(bag_files) as reader:
            connections = [
                c for c in reader.connections
                if c.topic == "/purepursuit_target"
            ]

            for connection, timestamp, rawdata in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, connection.msgtype)

                ct_err_vals.append(msg.e_fa)
                hd_err_vals.append(msg.error_yaw)
                speed_err_vals.append(msg.error_vel)

        ct_err_vals = np.array(ct_err_vals)
        hd_err_vals = np.array(hd_err_vals)
        speed_err_vals = np.array(speed_err_vals)

        # Compute RMS
        ct_rmse = np.sqrt(np.mean(ct_err_vals**2))
        hd_rmse = np.sqrt(np.mean(hd_err_vals**2))
        speed_rmse = np.sqrt(np.mean(speed_err_vals**2))

        results.append({
            "Path": p,
            "Velocity": float(v)/10.0,
            "CT_RMSE": ct_rmse,
            "Heading_RMSE": hd_rmse,
            "Speed_RMSE": speed_rmse
        })

# =========================
# Convert to table
# =========================
df = pd.DataFrame(results)

print("\nFinal Metrics Table:\n")
print(df.sort_values(["Path", "Velocity"]))
pivot_ct = df.pivot(index="Path", columns="Velocity", values="CT_RMSE")
print("\nCross-track RMSE:\n", pivot_ct)
print(pivot_ct.to_latex(float_format="%.4f"))