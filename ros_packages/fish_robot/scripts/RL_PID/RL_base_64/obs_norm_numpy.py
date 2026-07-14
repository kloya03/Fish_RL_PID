import numpy as np

class ObsNormalizer:
    def __init__(self, path="obs_norm.npz"):
        data = np.load(path)
        self.mean = data["mean"]
        self.var = data["var"]
        self.count = data["count"]

    def normalize(self, obs):
        return (obs - self.mean) / np.sqrt(self.var + 1e-8)
