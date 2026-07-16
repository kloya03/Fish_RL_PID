import cv2
import glob
import numpy as np
import os
import matplotlib.pyplot as plt

# ---- Paths ----
frame_paths = sorted(glob.glob("s_frames/frame_*.png"))
os.makedirs("marked_s_frames", exist_ok=True)

points = []
current_frame = 0

def mouse_callback(event, x, y, flags, param):
    global current_frame

    if event == cv2.EVENT_LBUTTONDOWN:
        img = cv2.imread(frame_paths[current_frame])

        # Save point
        points.append((x, y))
        print(f"Frame {current_frame}: ({x}, {y})")

        # Draw marker
        cv2.circle(img, (x, y), 8, (0, 0, 255), -1)

        # Save marked frame
        save_path = os.path.join(
            "marked_s_frames",
            os.path.basename(frame_paths[current_frame])
        )
        cv2.imwrite(save_path, img)

        current_frame += 1
        show_next_frame()

def show_next_frame():
    global current_frame

    if current_frame >= len(frame_paths):
        cv2.destroyAllWindows()
        print("Finished marking.")

        # Save trajectory
        pts = np.array(points)
        np.save("fish_trajectory_s1.5.npy", pts)

        # Plot trajectory
        if len(pts) > 0:
            plt.figure()
            plt.plot(pts[:,0], pts[:,1])
            plt.gca().invert_yaxis()
            plt.axis("equal")
            plt.title("Fish Trajectory (Manual Marking)")
            plt.show()
        return

    img = cv2.imread(frame_paths[current_frame])
    cv2.imshow("Click Fish Center", img)

cv2.namedWindow("Click Fish Center")
cv2.setMouseCallback("Click Fish Center", mouse_callback)

show_next_frame()
cv2.waitKey(0)