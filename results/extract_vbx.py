from rosbags.highlevel import AnyReader
from pathlib import Path
import numpy as np

# =========================
# CONFIG
# =========================
base_dir = Path("/home/ravi/iros_video")
save_dir = Path("/home/ravi/iros_video/processed_npz")
save_dir.mkdir(exist_ok=True)

paths = ["I", "R", "O", "S"]
velocities = ["1.0", "1.5", "2.0", "2.5"]

# =========================
# LOOP
# =========================
for p in paths:
    for v in velocities:

        folder = base_dir / p / v
        bag_files = list(folder.glob("*.bag"))

        if not bag_files:
            print(f"⚠ No bags found for {p} at {v}")
            continue

        vb_x_vals = []
        time_vals = []

        with AnyReader(bag_files) as reader:
            connections = [
                c for c in reader.connections
                if c.topic == "/KF_state_estimate"
            ]

            for connection, timestamp, rawdata in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, connection.msgtype)

                vb_x_vals.append(msg.vb_x)
                time_vals.append(timestamp * 1e-9)

        vb_x_vals = np.array(vb_x_vals)
        time_vals = np.array(time_vals)

        # Normalize time to start at 0
        time_vals = time_vals - time_vals[0]

        # =========================
        # Save
        # =========================
        save_path = save_dir / f"{p}_{v}.npz"

        np.savez(
            save_path,
            vb_x=vb_x_vals,
            time=time_vals,
            path=p,
            velocity=float(v)/10.0
        )

        print(f"Saved: {save_path}")