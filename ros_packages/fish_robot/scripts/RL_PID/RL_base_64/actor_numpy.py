import numpy as np

class ActorNumpy:
    def __init__(self, weight_path="actor_weights.npz"):
        data = np.load(weight_path)

        self.fc1_w = data["fc1_w"]
        self.fc1_b = data["fc1_b"]

        self.fc2_w = data["fc2_w"]
        self.fc2_b = data["fc2_b"]

        self.mean_w = data["mean_w"]
        self.mean_b = data["mean_b"]

        self.log_std = data["log_std"]

    def relu(self, x):
        return np.maximum(x, 0.0)

    def forward(self, obs):
        """
        obs: (obs_dim,) numpy
        returns: mean, std
        """

        x = self.relu(obs @ self.fc1_w + self.fc1_b)
        x = self.relu(x @ self.fc2_w + self.fc2_b)

        mean = x @ self.mean_w + self.mean_b

        log_std = np.clip(self.log_std, -20, -2)
        std = np.exp(log_std)

        return mean, std

    def act(self, obs):
        mean, std = self.forward(obs)
        action = mean  # deterministic
        return action
