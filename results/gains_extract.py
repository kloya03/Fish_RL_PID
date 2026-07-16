import numpy as np
from rosbags.highlevel import AnyReader
from pathlib import Path

# --------- CONFIG ---------
root_dir = Path("/home/ravi/iros_video")
save_dir = Path("/home/ravi/iros_video/gains_processed_npz")
topic_name = "/purepursuit_target"

save_dir.mkdir(parents=True, exist_ok=True)

# ALL paths
paths = ["I", "R", "O", "S"]
velocity_folder = "2.0"

for p in paths:

    folder = root_dir / p / velocity_folder

    if not folder.exists():
        print(f"⚠ Folder not found: {folder}")
        continue

    bag_files = list(folder.glob("*.bag"))

    if len(bag_files) == 0:
        print(f"⚠ No bag files in {folder}")
        continue

    for bag_path in bag_files:

        print(f"Processing: {p} | {bag_path.name}")

        t_list = []
        yaw_kp_list = []
        yaw_kd_list = []
        yaw_ki_list = []
        vel_kp_list = []
        vel_kd_list = []
        vel_ki_list = []

        with AnyReader([bag_path]) as reader:
            connections = [x for x in reader.connections if x.topic == topic_name]

            if len(connections) == 0:
                print("   ⚠ Topic not found")
                continue

            for connection, timestamp, rawdata in reader.messages(connections=connections):
                msg = reader.deserialize(rawdata, connection.msgtype)

                t = timestamp * 1e-9  # ns → sec

                t_list.append(t)

                yaw_kp_list.append(msg.yaw_Kp)
                yaw_kd_list.append(msg.yaw_Kd)
                yaw_ki_list.append(msg.yaw_Ki)

                vel_kp_list.append(msg.vel_Kp)
                vel_kd_list.append(msg.vel_Kd)
                vel_ki_list.append(msg.vel_Ki)

        if len(t_list) == 0:
            print("   ⚠ No messages extracted")
            continue

        t = np.array(t_list)
        t = t - t[0]

        save_name = f"{p}_2.0_{bag_path.stem}_pid.npz"
        save_path = save_dir / save_name

        np.savez(
            save_path,
            time=t,
            yaw_kp=np.array(yaw_kp_list),
            yaw_kd=np.array(yaw_kd_list),
            yaw_ki=np.array(yaw_ki_list),
            vel_kp=np.array(vel_kp_list),
            vel_kd=np.array(vel_kd_list),
            vel_ki=np.array(vel_ki_list),
        )

        print(f"   Saved → {save_path.name}")

print("Done extracting all paths @ 2.0 m/s.")