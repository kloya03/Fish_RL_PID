import cv2
import numpy as np

# =========================
# Load background
# =========================
img = cv2.imread("otest.png")

# Make copy for drawing
overlay = img.copy()

# =========================
# Load trajectories
# =========================
traj_10 = np.load("fish_trajectory_o1.0.npy")  # Not used in final drawing, but loaded for completeness
traj_15 = np.load("fish_trajectory_o1.5.npy")
traj_20 = np.load("fish_trajectory_o2.0.npy")
traj_25 = np.load("fish_trajectory_o2.5.npy")

# Convert to int32 for OpenCV
traj_10 = traj_10.astype(np.int32)
traj_15 = traj_15.astype(np.int32)
traj_20 = traj_20.astype(np.int32)
traj_25 = traj_25.astype(np.int32)

# Reshape for polylines
traj_10 = traj_10.reshape((-1,1,2))
traj_15 = traj_15.reshape((-1,1,2))
traj_20 = traj_20.reshape((-1,1,2))
traj_25 = traj_25.reshape((-1,1,2))

# =========================
# Colors (BGR)
# =========================
colors = {
    "1.0": (255, 255, 255),  # White
    "1.5": (0, 255, 0),      # Green
    "2.0": (255, 0, 255),    # Magenta
    "2.5": (0, 165, 255),    # Orange
}

thickness = {
    "1.0": 3,
    "1.5": 3,
    "2.0": 3,
    "2.5": 3,
}

# =========================
# Drawing function
# =========================
def draw_traj(img, traj, color, thickness_val):
    # Draw thick white outline
    # cv2.polylines(img, [traj], False, (255,255,255), thickness=thickness_val+1, lineType=cv2.LINE_AA)
        # Black outline (slightly thicker)
    cv2.polylines(img, [traj], False, (0,0,0),
                  thickness=thickness_val+2,
                  lineType=cv2.LINE_AA)

    # Draw colored line
    # cv2.polylines(img, [traj], False, color, thickness=thickness_val-1, lineType=cv2.LINE_AA)
    cv2.polylines(img, [traj], False, color,
                  thickness=thickness_val,
                  lineType=cv2.LINE_AA)

    # White outline (thick)
    # cv2.polylines(img, [traj], False, (255,255,255), thickness=1, lineType=cv2.LINE_AA)

    # Colored line (thin)
    # cv2.polylines(img, [traj], False, color, thickness=2, lineType=cv2.LINE_AA)

    # Start marker (small circle)
    start_pt = tuple(traj[0][0])
    cv2.circle(img, start_pt, 6, (255,255,255), -1, lineType=cv2.LINE_AA)

    # End marker (filled colored circle with black border)
    end_pt = tuple(traj[-1][0])
    # cv2.circle(img, end_pt, 10, (0,0,0), -1, lineType=cv2.LINE_AA)   # black border
    # cv2.circle(img, end_pt, 7, color, -1, lineType=cv2.LINE_AA)     # colored fill


# =========================
# Draw all
# =========================
draw_traj(overlay, traj_10, colors["1.0"], thickness["1.0"])
draw_traj(overlay, traj_15, colors["1.5"], thickness["1.5"])
draw_traj(overlay, traj_20, colors["2.0"], thickness["2.0"])
draw_traj(overlay, traj_25, colors["2.5"], thickness["2.5"])

# =========================
# Save result
# =========================
cv2.imwrite("overlay_result.png", overlay)

print("Saved as overlay_result.png")