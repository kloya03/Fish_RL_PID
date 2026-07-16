import torch
import torch.nn as nn


class Actor(nn.Module):
    """
    Standard deterministic MLP policy.
    No tanh squashing.
    No action scaling.
    """

    def __init__(self,
                 obs_dim=8,
                 act_dim=6,
                 hidden1=128,
                 hidden2=128,
                 hidden3=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.Linear(hidden3, act_dim)
        )

    def forward(self, obs):
        """
        obs: [B, obs_dim] or [obs_dim]
        returns: raw action output
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        action = self.net(obs)
        return action