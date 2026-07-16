import cv2
import glob
import numpy as np
import os

# ---- Load data ----
frame_paths = sorted(glob.glob("i_frames/frame_*.png"))
points = np.load("fish_trajectory_2.5.npy")

os.makedirs("traj_frames", exist_ok=True)

for i, path in enumerate(frame_paths):
    img = cv2.imread(path)

    if i < len(points):
        # Draw trajectory so far
        for j in range(1, i+1):
            pt1 = tuple(points[j-1].astype(int))
            pt2 = tuple(points[j].astype(int))
            cv2.line(img, pt1, pt2, (0, 165, 255), 3)

        # Draw current position
        cx, cy = points[i].astype(int)
        cv2.circle(img, (cx, cy), 6, (0, 0, 255), -1)

    save_path = os.path.join("traj_frames", os.path.basename(path))
    cv2.imwrite(save_path, img)

print("Trajectory frames saved.")

#ffmpeg -framerate 10 -i traj_frames/frame_%05d.png -c:v libx264 -pix_fmt yuv420p i_traj2.5.mp4