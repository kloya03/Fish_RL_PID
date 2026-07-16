import torch


class PathGeometry:
    @staticmethod
    def curvature_three_points(path_xy: torch.Tensor, i: int, eps: float = 1e-9):
        """
        Discrete curvature from 3 consecutive points.
        path_xy: (N,2) tensor
        """
        i = max(1, min(int(i), path_xy.shape[0] - 2))

        p0 = path_xy[i - 1]
        p1 = path_xy[i]
        p2 = path_xy[i + 1]

        a = torch.linalg.norm(p1 - p0)
        b = torch.linalg.norm(p2 - p1)
        c = torch.linalg.norm(p2 - p0)

        area2 = torch.abs(
            (p1[0] - p0[0]) * (p2[1] - p0[1])
            - (p1[1] - p0[1]) * (p2[0] - p0[0])
        )

        denom = torch.clamp(a * b * c, min=eps)
        kappa = 2.0 * area2 / denom
        return kappa

    @staticmethod
    def wrap_to_pi(a):
        a = torch.as_tensor(a)
        return torch.atan2(torch.sin(a), torch.cos(a))