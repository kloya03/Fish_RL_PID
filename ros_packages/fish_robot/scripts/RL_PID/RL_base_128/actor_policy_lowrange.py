import numpy as np

class ActorNumpy:
    def __init__(self, weight_path="policy_weights.npz"):

        data = np.load(weight_path)
        # PyTorch Linear: y = x @ W.T + b
        # Transpose ONCE at load
        self.fc1_w = data["net.0.weight"].T
        self.fc1_b = data["net.0.bias"]

        self.fc2_w = data["net.2.weight"].T
        self.fc2_b = data["net.2.bias"]

        self.fc3_w = data["net.4.weight"].T
        self.fc3_b = data["net.4.bias"]
        
        self.fc4_w = data["net.6.weight"].T
        self.fc4_b = data["net.6.bias"]

    # --------------------------------------------
    # Activations
    # --------------------------------------------
    def relu(self, x):
        return np.maximum(x, 0.0)

    def to_range_tanh(self, x, min_val, max_val):
        return min_val + (max_val - min_val) * (np.tanh(x) + 1.0) / 2.0

    # --------------------------------------------
    # Forward
    # --------------------------------------------
    def forward(self, obs):
        """
        obs: (obs_dim,) or (batch, obs_dim)
        returns: 6 scaled PID gains
        """

        if obs.ndim == 1:
            obs = obs.reshape(1, -1)

        x = self.relu(obs @ self.fc1_w + self.fc1_b)
        x = self.relu(x @ self.fc2_w + self.fc2_b)
        x = self.relu(x @ self.fc3_w + self.fc3_b)
        
        raw = (x @ self.fc4_w + self.fc4_b).squeeze()

        # ------------------------------------------------
        # APPLY EXACT SAME RANGES USED IN TRAINING
        # Replace bounds below if different
        # ------------------------------------------------

        # Steering PID (alpha)
        Kp_a = self.to_range_tanh(raw[0], 0, 1)
        Ki_a = self.to_range_tanh(raw[1], 0, 0.1)
        Kd_a = self.to_range_tanh(raw[2], 0, 0.1)

        # Speed PID (throttle)
        Kp_s = self.to_range_tanh(raw[3], 0.0, 2.5)
        Ki_s = self.to_range_tanh(raw[5], 0.0, 0.1)
        Kd_s = self.to_range_tanh(raw[4], 0.0, 0.1)

        return np.array([Kp_a, Ki_a, Kd_a,
                         Kp_s, Ki_s, Kd_s])

    # --------------------------------------------
    # Deployment call
    # --------------------------------------------
    def act(self, obs):
        return self.forward(obs)