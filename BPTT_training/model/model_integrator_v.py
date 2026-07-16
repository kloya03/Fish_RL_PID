import torch

class RK4Integrator:
    def __init__(self, model, dt: float):
        self.model = model
        self.dt = dt

    def rhs(self, state: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        state: (B, state_dim)
        tau:   (B,)
        returns: (B, state_dim)
        """
        return self.model.dynamics(0.0, state, tau)

    def step(self, state: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """
        state: (B, state_dim)
        tau:   (B,)
        """

        dt = self.dt

        k1 = self.rhs(state, tau)
        k2 = self.rhs(state + 0.5 * dt * k1, tau)
        k3 = self.rhs(state + 0.5 * dt * k2, tau)
        k4 = self.rhs(state + dt * k3, tau)

        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)