import numpy as np
import matplotlib.pyplot as plt
import scienceplots
from pathlib import Path

plt.style.use(['science', 'ieee', 'grid'])

data_dir = Path("/home/ravi/iros_video/processed_npz")
paths = ["I", "R", "O", "S"]
velocity = "2.5"

path_colors = {
    "I": (31, 119, 180),
    "R": (255, 127, 14),
    "O": (44, 160, 44),
    "S": (148, 103, 189),
}

def ema(signal, beta=0.08):
    smoothed = np.zeros_like(signal)
    smoothed[0] = signal[0]
    for i in range(1, len(signal)):
        smoothed[i] = beta * signal[i] + (1 - beta) * smoothed[i-1]
    return smoothed

fig, ax = plt.subplots(figsize=(7.16, 3.0))

for p in paths:
    file_path = data_dir / f"{p}_{velocity}.npz"
    if not file_path.exists():
        continue

    data = np.load(file_path)
    time = data["time"]
    vb_x = data["vb_x"]
    # if p == "O":
    #   mask = time <= (time[-1] - 10.0)
    #   time = time[mask]
    #   vb_x = vb_x[mask]
    # # Trim last 3 seconds
    # else:
    mask = time <= (time[-1] - 5.0)
    time = time[mask]
    vb_x = vb_x[mask]

    color = path_colors[p]

    # Raw
    ax.plot(time, np.abs(vb_x),
            color=color,
            linestyle="-",
            alpha=0.2)

    # Trend
    ax.plot(time,
            ema(np.abs(vb_x)),
            color=color,
            linestyle="-",
            label=p)

# Reference line
ax.axhline(0.25, linestyle="--", linewidth=1)

ax.set_xlabel("Time (s)")
ax.set_ylabel(r"$vb_x$ (m/s)")

ax.set_xlim(left=0)
ax.set_ylim(0, 0.28)

ax.legend(title="Path")
for ax in ax.flatten():
    ax.grid(True, linestyle='-', linewidth=0.3, alpha=0.4)
plt.tight_layout()
plt.savefig("speed_tracking_R_0p2.5.pdf")
plt.show()