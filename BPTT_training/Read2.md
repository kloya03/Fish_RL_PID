# BPTT Training: PID gains for heading and velocity

This folder contains code for reinforcement learning of PID gains using backpropagation through time (BPTT) on our reduced-order modified Chaplygin sleigh model.
The script trains a NN-based controller to generate PID gains, while using a pure pursuit lookahead point for reference heading and a constant velocity target per episode.

## Contents

- `DPC_PID_PIDs_old3_100.py`: main training script
- `model/`: local model code, including dynamics and network definitions
- `controller/`: local controller code used by the training script (includes a pure pursuit planner)
- `utils/`: utility functions used during training
- `requirements.txt`: Python dependencies for the training project

## How to run training

1. Open a terminal in the repository root:

```bash
cd /workspace/Fish_RL_PID
```

2. Install dependencies:

```bash
pip install -r BPTT_training/requirements.txt
```

3. Run the training script:

```bash
python BPTT_training/DPC_PID_PIDs_old3_100.py
```

## Notes

- The script uses a batch size of 8 environments by default (`B = 8`).

- The code detects CUDA, but it is not optimized for GPU execution. Expect CPU-like performance unless the script is modified for efficient GPU batching.
- Keep the `BPTT_training` folder structure intact so local imports resolve correctly.
- If you run the code from the repo root, the local imports in the script should work without modifying Python paths.
- Use the `requirements.txt` file to reproduce the environment.

