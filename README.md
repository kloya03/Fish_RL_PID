# Differentiable RL-PID Control for a Fish-Like Robot

<p align="center">
  <em>Physics-based simulation, BPTT-trained adaptive PID gains, and sim-to-real path tracking.</em>
</p>

This repository contains the modeling, learning, control, and
experimental evaluation code for **“Differentiable Reinforcement Learning for
Path Tracking by an Agile Fish-Like Robot.”** The method learns time-varying
PID gains in a differentiable simulator and transfers the resulting policy to
a physical internally actuated fish robot.

This repository accompanies:

> P. Chivkula, K. Loya, V. R. R. Varikuti and P. Tallapragada, “Differentiable Reinforcement Learning for Path Tracking by an Agile Fish-Like Robot,” *arxiv*, 2026. ](https://arxiv.org/abs/2607.16508)


The repository covers the complete research pipeline:

1. derive and validate a reduced-order modified Chaplygin-sleigh model;
2. train a neural policy through backpropagation through time (BPTT);
3. use the policy to adapt the heading and speed PID gains online;
4. deploy the controller through ROS with camera/IMU state estimation; and
5. evaluate path and speed tracking on `I`, `R`, `O`, and `S` trajectories.

<p align="center">
  <img src="output.gif" width="480" alt="Fish-like robot swimming during an experimental trial">
  <br>
  <sub>Physical fish-like robot during closed-loop swimming.</sub>
</p>

## Method at a glance

The robot is a 250 mm long, 535 g planar swimmer. An internal balanced rotor
provides propulsion, while a servo-actuated flexible tail controls steering.
The platform reaches up to 2 body lengths per second and can turn with a
minimum radius of approximately 0.3 body lengths.

The controller retains the structure and interpretability of two PID loops:

- a **heading PID** maps line-of-sight heading error to rudder angle;
- a **speed PID** maps forward-velocity error to rotor throttle; and
- a small multilayer perceptron outputs the six PID gains from normalized
  heading, velocity, desired velocity, heading error, speed error, and
  cross-track error.

During training, gradients propagate through the policy, PID controller,
throttle mapping, and differentiable dynamics. A curriculum progressively
introduces faster reference speeds and higher-curvature sinusoidal and circular
paths. The learned NumPy policy is then run onboard the physical robot without
requiring PyTorch at deployment time.

```text
reference path -> line-of-sight planner -> adaptive PID gains -> robot inputs
                         ^                                      |
                         |---- camera + IMU state estimate <----|
```

## Repository structure

| Path | Purpose |
| --- | --- |
| [`BPTT_training/`](BPTT_training/) | Differentiable PyTorch model, policy network, pure-pursuit planner, curriculum, and main BPTT training script. |
| [`simulation_model/`](simulation_model/) | Symbolic/dynamic model development in MATLAB and a Python model/notebook for the four-link swimmer. |
| [`ros_packages/fish_robot/`](ros_packages/fish_robot/) | ROS nodes for sensing, state estimation, actuation, path following, fixed-gain PID, and learned RL-PID control. |
| [`ros_packages/fish_pc/`](ros_packages/fish_pc/) | Host-computer overhead-camera tracking and target-generation nodes. |
| [`ros_packages/custom_msgs/`](ros_packages/custom_msgs/) | Custom ROS message definitions used by the robot and tracking stack. |
| [`Camera_Calibrate_runcam/`](Camera_Calibrate_runcam/) | Intrinsic/extrinsic fisheye-camera calibration, undistortion, and pixel-to-metre mapping utilities. |
| [`results/`](results/) | Experimental bags, trajectories, videos/GIFs, processed gains, plots, and evaluation scripts. |
| [`2D tracking results/`](2D%20tracking%20results/) | Additional two-dimensional tracking outputs. |

The main training entry point is
[`BPTT_training/DPC_PID_PIDs_old3_100.py`](BPTT_training/DPC_PID_PIDs_old3_100.py).
The physical learned-gain controllers are under
[`ros_packages/fish_robot/scripts/RL_PID/`](ros_packages/fish_robot/scripts/RL_PID/),
with exported NumPy policy weights in the `RL_base_64` and `RL_base_128`
subdirectories.

## Quick start: train in simulation

### Requirements

- Python 3
- NumPy, SciPy, Matplotlib, SymPy, Jupyter, and tqdm
- PyTorch (CPU execution is supported and is how the reported training was run)

Create an isolated environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r BPTT_training/requirements.txt
python -m pip install torch tqdm
```

Install the PyTorch build appropriate for your platform if you require CUDA.
The current implementation detects CUDA, but its batched dynamics were designed
primarily for multi-core CPU execution.

Start training with:

```bash
python BPTT_training/DPC_PID_PIDs_old3_100.py
```

By default, training uses eight parallel environments. The script alternates
between sinusoidal and circular paths, increases curvature based on recent
cross-track performance, spans desired forward speeds from 0.1 to 0.3 m/s, and
periodically writes a PyTorch checkpoint. Training constants, horizon, loss
weights, and output location are configured directly in the training script.
See [`BPTT_training/Read2.md`](BPTT_training/Read2.md) for shorter training notes.

## Simulation and model development

The reduced-order model represents the head and flexible tail as a constrained
multi-link system with added mass, viscous drag, torsional tail stiffness, and a
nonholonomic lateral-velocity constraint at the tail. This captures the stable
limit-cycle propulsion needed for control while remaining fast enough to
differentiate through repeated closed-loop rollouts.

- Start with
  [`simulation_model/Python codes/CS_4link_dynamics.ipynb`](simulation_model/Python%20codes/CS_4link_dynamics.ipynb)
  for interactive Python exploration.
- Use
  [`simulation_model/Python codes/CS_4link_RL.py`](simulation_model/Python%20codes/CS_4link_RL.py)
  for the Python model implementation.
- Use
  [`simulation_model/Matlab Codes/Balanced_Cs_wAddedMass_4link_v1.mlx`](simulation_model/Matlab%20Codes/Balanced_Cs_wAddedMass_4link_v1.mlx)
  for the MATLAB derivation and simulation workflow.
- The differentiable training version is
  [`BPTT_training/model/CS_4link_v4_torch.py`](BPTT_training/model/CS_4link_v4_torch.py).

## Physical robot and ROS stack

The experimental system combines an overhead fisheye camera, colored body
markers, a BNO085 IMU, a Kalman-filtered pose/velocity estimate, Vertiq motor
feedback, a Raspberry Pi Zero 2W, and the learned controller. The most relevant
launch files are:

| Command | Function |
| --- | --- |
| `roslaunch fish_robot base_imu_kf.launch` | IMU, camera/IMU fusion, and motor estimates. |
| `roslaunch fish_robot PID_controller.launch` | Fixed-gain PID baseline and motor feedback. |
| `roslaunch fish_robot RL_controller.launch` | Learned RL-PID controller and motor feedback. |
| `roslaunch fish_robot PP_controller.launch` | Pure-pursuit controller with sensing. |

> **Hardware note:** this repository contains the ROS nodes, messages, and
> launch files, but it does not currently include catkin `package.xml` or
> `CMakeLists.txt` manifests. Integrate these directories into the configured
> ROS workspace used by the robot, provide the required device libraries, set
> camera/device parameters, build the workspace, and source it before using the
> commands above. Do not run actuation nodes on an unsecured robot.

The camera calibration workflow is in
[`Camera_Calibrate_runcam/`](Camera_Calibrate_runcam/). It includes frame
capture, intrinsic and extrinsic calibration, fisheye undistortion, saved
calibration parameters, and pixel-to-metre mapping. Calibration is specific to
the camera resolution and physical test area; select or regenerate the matching
parameter file before an experiment.

## Results

The trained policy was deployed directly on the physical robot in a 14 ft ×
7 ft indoor pool. Experiments used four paths with different curvature profiles
(`I`, `R`, `O`, and `S`) at commanded speeds of 0.10, 0.15, 0.20, and 0.25 m/s.
The paper reports stable closed-loop tracking for every tested path/speed pair.

### IROS trajectory gallery

The following experimental overlays show the four letter-shaped reference
paths stacked vertically. Each image combines trials at the tested commanded
speeds so that the path-following behavior can be compared directly.

<p align="center">
  <img src="results/I/combined/I.png" width="620" alt="Experimental I-path overlays at four commanded speeds">
  <br>
  <sub><strong>I path</strong> — predominantly straight-line tracking.</sub>
</p>

<p align="center">
  <img src="results/R/combined/R.png" width="620" alt="Experimental R-path overlays at four commanded speeds">
  <br>
  <sub><strong>R path</strong> — mixed straight and curved segments.</sub>
</p>

<p align="center">
  <img src="results/O/combined/O.png" width="620" alt="Experimental O-path overlays at four commanded speeds">
  <br>
  <sub><strong>O path</strong> — continuous closed-loop turning.</sub>
</p>

<p align="center">
  <img src="results/S/combined/s.png" width="620" alt="Experimental S-path overlays at four commanded speeds">
  <br>
  <sub><strong>S path</strong> — alternating-curvature tracking.</sub>
</p>

Across the experimental matrix, the reported cross-track RMSE ranges from
**0.011 m to 0.122 m**, and speed RMSE ranges from **0.015 m/s to 0.075 m/s**.
The nearly straight `I` path has the smallest lateral error, while the curved
paths are more challenging at low speed because steering authority and forward
propulsion are coupled. Simulation tests on sinusoidal and circular paths
achieve approximately **0.01 m mean cross-track error**.

<p align="center">
  <img src="results/R/2.5/r_trajmod.gif" width="360" alt="Fish robot tracking an R trajectory at 0.25 metres per second">
  &nbsp;
  <img src="results/S/2.5/S_trajmod.gif" width="360" alt="Fish robot tracking an S trajectory at 0.25 metres per second">
  <br>
  <sub>Physical R- and S-path trials at a commanded speed of 0.25 m/s.</sub>
</p>

The [`results/`](results/) directory is organized first by path letter and then
by speed. It contains the original ROS bags, extracted NumPy trajectories,
annotated recordings, combined overlays, learned-gain data, and plotting/
metric scripts. Useful summary files include:

- [`results/pid_gains_col.pdf`](results/pid_gains_col.pdf) — online PID-gain
  evolution during representative experiments;
- [`results/speed_tracking_all_velocities_20s.pdf`](results/speed_tracking_all_velocities_20s.pdf)
  — speed tracking across commands and paths;
- [`results/metrics.py`](results/metrics.py) — tracking-metric calculations;
- [`results/gain_plot.py`](results/gain_plot.py) and
  [`results/velocity_plot.py`](results/velocity_plot.py) — plot generation; and
- each `results/<path>/<speed>/draw_traj.py` — trajectory extraction/plotting
  for an individual trial.

Large `.bag` and video files are raw experimental artifacts; the `.npy`, `.npz`,
`.png`, `.gif`, and `.pdf` files are the lighter-weight processed outputs.

## Paper

If this repository supports your work, please cite the companion manuscript:

```bibtex
@misc{chivkula2026_diff_rl_fish,
      title={Differentiable Reinforcement Learning for Path Tracking by an Agile Fish-Like Robot}, 
      author={Prashanth Chivkula and Kartik Loya and Venkata Ravindhra Reddy Varikuti and Phanindra Tallapragada},
      year={2026},
      eprint={2607.16508},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2607.16508}, 
}
```

Update the BibTeX venue, pages, and DOI when the final publication metadata is
available.

## License

This project is released under the terms in [`LICENSE`](LICENSE).
