import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from pathlib import Path

plt.style.use(['science', 'ieee', 'grid'])

data_dir = Path("/home/ravi/iros_video/processed_npz")

paths = ["I", "R", "O", "S"]
velocities = ["1.0", "1.5", "2.0", "2.5"]

def rgb(r, g, b):
    return (r/255, g/255, b/255)

path_colors = {
    "I": rgb(255, 127, 14),
    "R": rgb(31, 119, 180),
    "O": rgb(148, 103, 189),
    "S": rgb(44, 160, 44),
}

def ema(signal, beta=0.08):
    smoothed = np.zeros_like(signal)
    smoothed[0] = signal[0]
    for i in range(1, len(signal)):
        smoothed[i] = beta * signal[i] + (1 - beta) * smoothed[i-1]
    return smoothed

# Single IEEE column size
fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.6), sharex=True, sharey=True)
axes = axes.flatten()

for idx, v in enumerate(velocities):
    ax = axes[idx]
    commanded_speed = float(v) / 10.0

    for p in paths:
        file_path = data_dir / f"{p}_{v}.npz"
        if not file_path.exists():
            continue

        data = np.load(file_path)
        time = data["time"]
        vb_x = np.abs(data["vb_x"])

        # ---- Cut to first 25 seconds ----
        if p == 'I':

            mask = time <= 10.0
            time = time[mask]
            vb_x = vb_x[mask]

        color = path_colors[p]

        # # Light raw signal (optional)
        # ax.plot(time, vb_x,
        #         color=color,
        #         alpha=0.12,
        #         linestyle="-",
        #         linewidth=0.5)

        # Solid trend line
        ax.plot(time,
                ema(vb_x),
                color=color,
                linestyle="-",
                linewidth=1.0)

    # Command reference line
    ax.axhline(commanded_speed,
               linestyle="--",
               linewidth=0.9,
               color="black")

    ax.set_title(f"desired speed={commanded_speed:.2f} m/s", fontsize=8)

# Axis labels only outer
axes[2].set_xlabel("Time (s)")
axes[3].set_xlabel("Time (s)")
axes[0].set_ylabel(r"$vb_x$ (m/s)")
axes[2].set_ylabel(r"$vb_x$ (m/s)")

for ax in axes:
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 0.35)

# Single compact legend
# handles = [plt.Line2D([0], [0], color=path_colors[p], lw=0.7) for p in paths]
# fig.legend(handles, paths,
#            loc="lower center",
#            ncol=4,
#            frameon=False,
#            fontsize=7)
axes[0].legend(
    paths,
    loc="upper right",
    bbox_to_anchor=(0.98, 0.98),
    frameon=False,
    fontsize=6
)
for ax in axes.flatten():
    ax.grid(True, linestyle='-', linewidth=0.3, alpha=0.4)
plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.savefig("speed_tracking_all_velocities_20s.pdf")
plt.show()