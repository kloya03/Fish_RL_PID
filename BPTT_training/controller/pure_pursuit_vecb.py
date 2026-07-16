import torch


class PurePursuitClassic:

    def __init__(self):
        pass

    @staticmethod
    def _wrap_angle(theta):
        return (theta + torch.pi) % (2 * torch.pi) - torch.pi

    def __call__(self, state_aug, path_xy, lookahead):
        """
        state_aug : (B,3)
        path_xy   : (B,M,2)
        lookahead : (B,)
        """

        B, M, _ = path_xy.shape
        device = path_xy.device

        # -------------------------------------------------
        # Extract head positions
        # -------------------------------------------------
        p = state_aug[:, :2]                  # (B,2)
        x = p[:, 0]
        y = p[:, 1]

        # -------------------------------------------------
        # Closest waypoint (batched path)
        # -------------------------------------------------
        diff = path_xy - p.unsqueeze(1)       # (B,M,2)
        d2 = (diff ** 2).sum(dim=2)           # (B,M)
        i0 = torch.argmin(d2, dim=1)          # (B,)
        e_xy = torch.sqrt(d2.gather(1, i0.unsqueeze(1)).squeeze(1))

        # -------------------------------------------------
        # Build segments (batched)
        # -------------------------------------------------
        a = path_xy[:, :-1, :]                # (B,M-1,2)
        b = path_xy[:, 1:, :]                 # (B,M-1,2)
        seg = b - a                           # (B,M-1,2)

        p_exp = p.unsqueeze(1)                # (B,1,2)
        f = a - p_exp                         # (B,M-1,2)

        A = (seg * seg).sum(dim=2)            # (B,M-1)
        Bq = 2.0 * (f * seg).sum(dim=2)       # (B,M-1)
        C = (f * f).sum(dim=2) - lookahead.unsqueeze(1) ** 2

        disc = Bq * Bq - 4.0 * A * C          # (B,M-1)

        valid_disc = disc >= 0.0

        # Safe discriminant
        disc_safe = torch.where(
            torch.isfinite(disc),
            torch.clamp(disc, min=1e-6),
            torch.full_like(disc, 1e-6)
        )

        sqrt_disc = torch.sqrt(disc_safe)

        t1 = (-Bq - sqrt_disc) / (2.0 * A)
        t2 = (-Bq + sqrt_disc) / (2.0 * A)

        valid_t1 = (t1 >= 0.0) & (t1 <= 1.0) & valid_disc
        valid_t2 = (t2 >= 0.0) & (t2 <= 1.0) & valid_disc

        # -------------------------------------------------
        # Combine roots
        # -------------------------------------------------
        t_candidates = torch.stack([t1, t2], dim=2)      # (B,M-1,2)
        valid_mask = torch.stack([valid_t1, valid_t2], dim=2)

        seg_indices = torch.arange(M-1, device=device).view(1, -1, 1)
        progress = seg_indices + t_candidates

        progress[~valid_mask] = -1e6

        best_idx = torch.argmax(progress.view(B, -1), dim=1)

        seg_id = best_idx // 2
        root_id = best_idx % 2

        t_best = t_candidates[
            torch.arange(B, device=device),
            seg_id,
            root_id
        ]

        # -------------------------------------------------
        # Compute goal
        # -------------------------------------------------
        goal = a[
            torch.arange(B, device=device),
            seg_id
        ] + t_best.unsqueeze(1) * seg[
            torch.arange(B, device=device),
            seg_id
        ]

        # -------------------------------------------------
        # Fallback
        # -------------------------------------------------
        no_intersection = (progress.view(B, -1).max(dim=1).values < 0)

        fallback_idx = torch.clamp(i0 + 1, max=M-1)
        fallback_goal = path_xy[
            torch.arange(B, device=device),
            fallback_idx
        ]

        goal[no_intersection] = fallback_goal[no_intersection]

        # -------------------------------------------------
        # Heading
        # -------------------------------------------------
        dx = goal[:, 0] - x
        dy = goal[:, 1] - y

        theta_ref = torch.atan2(dy, dx)
        theta_ref = self._wrap_angle(theta_ref)

        Ld = torch.sqrt(dx * dx + dy * dy)

        return theta_ref, i0, seg_id, goal[:, 0], goal[:, 1], Ld, e_xy