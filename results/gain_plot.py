import numpy as np
import matplotlib.pyplot as plt
# import scienceplots
from pathlib import Path

# plt.style.use(['science', 'ieee'])

data_dir = Path("/home/ravi/iros_video/gains_processed_npz")

paths = ["R", "S"]
velocity_tag = "vel_2"

def rgb(r, g, b):
    return (r/255, g/255, b/255)


path_colors = {
    "I": rgb(255, 127, 14),
    "R": rgb(31, 119, 180),
    "O": rgb(148, 103, 189),
    "S": rgb(44, 160, 44),
}
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Times", "Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",

    "font.size": 20,
    "axes.labelsize": 18,   # ← 14 only here
    "axes.titlesize": 18,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

def ema(signal, beta=0.08):
    smoothed = np.zeros_like(signal)
    smoothed[0] = signal[0]
    for i in range(1, len(signal)):
        smoothed[i] = beta * signal[i] + (1 - beta) * smoothed[i-1]
    return smoothed


fig, axes = plt.subplots(3, 2, figsize=(7.2, 6.0),dpi=300, sharex=True)

gain_map = {
    0: ("yaw_kp", "vel_kp", r"$K_p$"),
    1: ("yaw_kd", "vel_kd", r"$K_d$"),
    2: ("yaw_ki", "vel_ki", r"$K_i$")
}

for p in paths:

    files = list(data_dir.glob(f"{p}_*{velocity_tag}*_pid.npz"))
    if not files:
        continue

    data = np.load(files[0])
    time = data["time"]

    mask = time <= 20.0
    time = time[mask]

    color = path_colors[p]

    for row in range(3):

        yaw_key, vel_key, _ = gain_map[row]

        yaw_signal = ema(data[yaw_key][mask])
        vel_signal = ema(data[vel_key][mask])

        axes[row, 0].plot(
            time, yaw_signal,
            color=color,
            linewidth=1.5,
            linestyle="-"
        )

        axes[row, 1].plot(
            time, vel_signal,
            color=color,
            linewidth=1.5,
            linestyle="-"
        )


# Column titles
axes[0, 0].set_title("Yaw Gains" )
axes[0, 1].set_title("Velocity Gains")

# Row labels
axes[0, 0].set_ylabel(r"$K_p$", )
axes[1, 0].set_ylabel(r"$K_d$")
axes[2, 0].set_ylabel(r"$K_i$")

# X labels
axes[2, 0].set_xlabel("Time (s)")
axes[2, 1].set_xlabel("Time (s)")

# # Legend
# legend_lines = [
#     plt.Line2D([0], [0], color=path_colors["R"], lw=1.5),
#     plt.Line2D([0], [0], color=path_colors["S"], lw=1.5),
# ]

# fig.legend(
#     legend_lines,
#     ["R", "S"],
#     loc="lower center",
#     ncol=2,
#     frameon=False,
#     fontsize=8
# )
axes[0, 0].legend(
    ["R", "S"],
    loc="upper right",
    frameon=False,

)
# for ax in axes.flatten():
#     ymin, ymax = ax.get_ylim()
#     ax.set_yticks([ymin, ymax])
for ax in axes.flatten():
    ax.grid(True, linestyle='-', linewidth=0.3, alpha=0.4)
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig("pid_gains_col.pdf")
plt.show()