import torch
import math
import numpy as np
# You MUST provide torch versions of these:
# mass_matrix_torch(...)
# coriolis_vector_torch(...)
# gravity_vector_torch(...)

class ChaplyginSleighModelTorch:
    def __init__(self, const_vals, inp):
        """
        const_vals: list/tuple/torch tensor of constants
        inp: callable inputs(t) -> torch tensor of inputs
        """
        self.Const = torch.as_tensor(const_vals, dtype=torch.float32).view(-1)
        self.inputs = inp

    def dynamics(self, t, states, tau=None):
        """
        Torch dynamics: returns dq (torch tensor)

        states:
            (n_state,)        → single env
            (B, n_state)      → batched envs

        returns:
            (B, n_state)
        """

        # -----------------------------
        # Ensure batched state
        # -----------------------------
        states = torch.as_tensor(states, dtype=torch.float32)

        if states.dim() == 1:
            states = states.unsqueeze(0)  # (1, n_state)

        B = states.shape[0]


        inputs = self.inputs(t)
        inputs = torch.as_tensor(inputs, dtype=states.dtype)

        if inputs.dim() == 1:
            inputs = inputs.unsqueeze(0)  # (1, n_input)

        l1, l2, ls, b = self.Const[:4]

                # -------------------------------------------------
        # Batched mass matrix and forces
        # -------------------------------------------------
        M    = self.Ms(states, inputs)      # (B, 4, 4)
        C_qd = self.C_qd(states, inputs)    # (B, 4)
        G    = self.G(states, inputs)       # (B, 4)

        # -------------------------------------------------
        # RHS for solve
        # -------------------------------------------------
        rhs = -(C_qd + G)                   # (B, 4)

        # Ensure rhs shape is (B, 4, 1)
        if rhs.dim() == 2:
            rhs = rhs.unsqueeze(-1)         # (B, 4, 1)

        # -------------------------------------------------
        # Batched linear solve
        # -------------------------------------------------
        # torch.linalg.solve supports batched solve
        ddq = torch.linalg.solve(M, rhs).squeeze(-1)   # (B, 4)

        # -------------------------------------------------
        # Extract states (batched!)
        # -------------------------------------------------
        u      = states[:, 0]
        v      = states[:, 1]
        theta  = states[:, 2]
        q3     = states[:, 3]
        q4     = states[:, 4]
        q5     = states[:, 5]

        # -------------------------------------------------
        # xdot, ydot (batched trig)
        # -------------------------------------------------
        xdot = u * torch.cos(theta) - 0.5 * l1 * v * torch.sin(theta)
        ydot = u * torch.sin(theta) + 0.5 * l1 * v * torch.cos(theta)

        # -------------------------------------------------
        # Assemble dq (B, n_state)
        # -------------------------------------------------
        dq = torch.stack([
            ddq[:, 0],
            ddq[:, 1],
            v,
            ddq[:, 2],
            q3,
            ddq[:, 3],
            q5,
            xdot,
            ydot
        ], dim=1)   # (B, 9)

        return dq
    def Ms(self, states, inputs):

        # states: (B, n_state)
        # inputs: (B, n_input)

        B = states.shape[0]
        consts = self.Const

        # Extract state columns
        s = [states[:, i] for i in range(states.shape[1])]
        u = [inputs[:, i] for i in range(inputs.shape[1])]

        # mass_matrix_torch must be rewritten to support batched tensors
        return mass_matrix_torch(*s, *u, *consts)   # must return (B,4,4)
    
    def C_qd(self, states, inputs):

        s = [states[:, i] for i in range(states.shape[1])]
        u = [inputs[:, i] for i in range(inputs.shape[1])]

        return coriolis_vector_torch(*s, *u, *self.Const)  # must return (B,4)
    
    def G(self, states, inputs):

        s = [states[:, i] for i in range(states.shape[1])]
        u = [inputs[:, i] for i in range(inputs.shape[1])]

        return gravity_vector_torch(*s, *u, *self.Const)   
    
def mass_matrix_torch(s1, s2, s3, s4, s5, s6, s7, s8, s9, al, d_al, dd_al, dd_ph, l1, l2, ls, b, c, ma_l1x, ma_l1y, ma_l2x, ma_l2y, ma_hx, ma_hy, ma_lsx, ma_lsy, Ia_h, Ia_l1, Ia_l2, Ia_ls, I_r, C_hx, C_hy, C_l1x, C_l1y, C_l2x, C_l2y, C_lsx, C_lsy, K_1, K_2):
    m00= (-1.0*ma_hx*torch.sin(s3)**2 + 1.0*ma_hx + 1.0*ma_hy*torch.sin(s3)**2 - 1.0*ma_l1x*torch.sin(s3)**2 
                    + 1.0*ma_l1x + 1.0*ma_l1y*torch.sin(s3)**2 - 1.0*ma_l2x*torch.sin(s3)**2 + 1.0*ma_l2x 
                    + 1.0*ma_l2y*torch.sin(s3)**2 - 1.0*ma_lsx*torch.sin(s3)**2 + 1.0*ma_lsx + 1.0*ma_lsy*torch.sin(s3)**2
    )
    m01 =(l1*(-0.5*ma_hx + 0.5*ma_hy - 0.25*ma_l1x + 0.25*ma_l1y - 0.5*ma_l2x + 0.5*ma_l2y - 0.5*ma_lsx + 0.5*ma_lsy)*torch.sin(2*s3))

    m02 =(l2*(-1.0*ma_hx*torch.sin(s5)*torch.cos(s3) + 1.0*ma_hy*torch.sin(s3)*torch.cos(s5) - 0.5*ma_l2x*torch.sin(s5)*torch.cos(s3) 
                        + 0.5*ma_l2y*torch.sin(s3)*torch.cos(s5) - 1.0*ma_lsx*torch.sin(s5)*torch.cos(s3) + 1.0*ma_lsy*torch.sin(s3)*torch.cos(s5)))
    m03 =(-1.0*b*ma_hx*torch.sin(s7)*torch.cos(s3) + 1.0*b*ma_hy*torch.sin(s3)*torch.cos(s7) - 1.0*ls*ma_hx*torch.sin(al + s7)*torch.cos(s3) 
        + 1.0*ls*ma_hy*torch.sin(s3)*torch.cos(al + s7) - 0.5*ls*ma_lsx*torch.sin(al + s7)*torch.cos(s3) + 0.5*ls*ma_lsy*torch.sin(s3)*torch.cos(al + s7))
    
    row0 = torch.stack([m00, m01, m02, m03], dim =1)
    
    m10 =(-0.25*l1*(2*ma_hx - 2*ma_hy + ma_l1x - ma_l1y + 2*ma_l2x - 2*ma_l2y + 2*ma_lsx - 2*ma_lsy)*torch.sin(2*s3))



    m11 =(1.0*Ia_l1 + 1.0*l1**2*ma_hx*torch.sin(s3)**2 
            - 1.0*l1**2*ma_hy*torch.sin(s3)**2 + 1.0*l1**2*ma_hy + 0.25*l1**2*ma_l1x*torch.sin(s3)**2 - 0.25*l1**2*ma_l1y*torch.sin(s3)**2 + 0.25*l1**2*ma_l1y 
            + 1.0*l1**2*ma_l2x*torch.sin(s3)**2 - 1.0*l1**2*ma_l2y*torch.sin(s3)**2 + 1.0*l1**2*ma_l2y + 1.0*l1**2*ma_lsx*torch.sin(s3)**2 - 1.0*l1**2*ma_lsy*torch.sin(s3)**2
            + 1.0*l1**2*ma_lsy)
    
    m12 =( l1*l2*(1.0*ma_hx*torch.sin(s3)*torch.sin(s5) + 1.0*ma_hy*torch.cos(s3)*torch.cos(s5) + 0.5*ma_l2x*torch.sin(s3)*torch.sin(s5)
            + 0.5*ma_l2y*torch.cos(s3)*torch.cos(s5) + 1.0*ma_lsx*torch.sin(s3)*torch.sin(s5) + 1.0*ma_lsy*torch.cos(s3)*torch.cos(s5)))
    
    m13 =(l1*(1.0*b*ma_hx*torch.sin(s3)*torch.sin(s7)+ 1.0*b*ma_hy*torch.cos(s3)*torch.cos(s7) + 1.0*ls*ma_hx*torch.sin(s3)*torch.sin(al + s7) + 1.0*ls*ma_hy*torch.cos(s3)*torch.cos(al + s7)
            + 0.5*ls*ma_lsx*torch.sin(s3)*torch.sin(al + s7) + 0.5*ls*ma_lsy*torch.cos(s3)*torch.cos(al + s7)))
    
    row1 = torch.stack([m10, m11, m12, m13], dim=1)

    m20 =(l2*(-1.0*ma_hx*torch.sin(s5)*torch.cos(s3) + 1.0*ma_hy*torch.sin(s3)*torch.cos(s5) - 0.5*ma_l2x*torch.sin(s5)*torch.cos(s3) + 0.5*ma_l2y*torch.sin(s3)*torch.cos(s5) 
            - 1.0*ma_lsx*torch.sin(s5)*torch.cos(s3)+ 1.0*ma_lsy*torch.sin(s3)*torch.cos(s5)))

    m21 =(l1*l2*(1.0*ma_hx*torch.sin(s3)*torch.sin(s5) + 1.0*ma_hy*torch.cos(s3)*torch.cos(s5) + 0.5*ma_l2x*torch.sin(s3)*torch.sin(s5) 
            + 0.5*ma_l2y*torch.cos(s3)*torch.cos(s5) + 1.0*ma_lsx*torch.sin(s3)*torch.sin(s5) + 1.0*ma_lsy*torch.cos(s3)*torch.cos(s5)))

    m22 =(1.0*Ia_l2 + 1.0*l2**2*ma_hx*torch.sin(s5)**2 + 1.0*l2**2*ma_hy*torch.cos(s5)**2 + 0.25*l2**2*ma_l2x*torch.sin(s5)**2 + 0.25*l2**2*ma_l2y*torch.cos(s5)**2 
            + 1.0*l2**2*ma_lsx*torch.sin(s5)**2 + 1.0*l2**2*ma_lsy*torch.cos(s5)**2)

    m23 =(l2*(1.0*b*ma_hx*torch.sin(s5)*torch.sin(s7) + 1.0*b*ma_hy*torch.cos(s5)*torch.cos(s7) 
        + 1.0*ls*ma_hx*torch.sin(s5)*torch.sin(al + s7) + 1.0*ls*ma_hy*torch.cos(s5)*torch.cos(al + s7) + 0.5*ls*ma_lsx*torch.sin(s5)*torch.sin(al + s7) 
        + 0.5*ls*ma_lsy*torch.cos(s5)*torch.cos(al + s7)))

    row2 = torch.stack([m20, m21, m22, m23], dim=1)

    m30 =(-1.0*b*ma_hx*torch.sin(s7)*torch.cos(s3) + 1.0*b*ma_hy*torch.sin(s3)*torch.cos(s7) - 1.0*ls*ma_hx*torch.sin(al + s7)*torch.cos(s3) + 1.0*ls*ma_hy*torch.sin(s3)*torch.cos(al + s7) 
        - 0.5*ls*ma_lsx*torch.sin(al + s7)*torch.cos(s3) + 0.5*ls*ma_lsy*torch.sin(s3)*torch.cos(al + s7))

    m31=(l1*(1.0*b*ma_hx*torch.sin(s3)*torch.sin(s7) + 1.0*b*ma_hy*torch.cos(s3)*torch.cos(s7) 
        + 1.0*ls*ma_hx*torch.sin(s3)*torch.sin(al + s7) + 1.0*ls*ma_hy*torch.cos(s3)*torch.cos(al + s7) + 0.5*ls*ma_lsx*torch.sin(s3)*torch.sin(al + s7) 
        + 0.5*ls*ma_lsy*torch.cos(s3)*torch.cos(al + s7)))

    m32=(l2*(1.0*b*ma_hx*torch.sin(s5)*torch.sin(s7) + 1.0*b*ma_hy*torch.cos(s5)*torch.cos(s7) + 1.0*ls*ma_hx*torch.sin(s5)*torch.sin(al + s7) 
            + 1.0*ls*ma_hy*torch.cos(s5)*torch.cos(al + s7) + 0.5*ls*ma_lsx*torch.sin(s5)*torch.sin(al + s7) + 0.5*ls*ma_lsy*torch.cos(s5)*torch.cos(al + s7)))

    m33 =( 1.0*I_r + 1.0*Ia_h + 1.0*Ia_ls + 1.0*b**2*ma_hx*torch.sin(s7)**2 
            + 1.0*b**2*ma_hy*torch.cos(s7)**2 + 2.0*b*ls*ma_hx*torch.sin(s7)*torch.sin(al + s7) + 2.0*b*ls*ma_hy*torch.cos(s7)*torch.cos(al + s7) + 1.0*ls**2*ma_hx*torch.sin(al + s7)**2 
            + 1.0*ls**2*ma_hy*torch.cos(al + s7)**2 + 0.25*ls**2*ma_lsx*torch.sin(al + s7)**2 + 0.25*ls**2*ma_lsy*torch.cos(al + s7)**2)

    row3 = torch.stack([m30, m31, m32, m33], dim=1)

    return torch.stack([row0, row1, row2, row3], dim=1)# must return (B,4)

def coriolis_vector_torch(s1, s2, s3, s4, s5, s6, s7, s8, s9, al, d_al, dd_al, dd_ph, l1, l2, ls, b, c, ma_l1x, ma_l1y, ma_l2x, ma_l2y, ma_hx, ma_hy, ma_lsx, ma_lsy, Ia_h, Ia_l1, Ia_l2, Ia_ls, I_r, C_hx, C_hy, C_l1x, C_l1y, C_l2x, C_l2y, C_lsx, C_lsy, K_1, K_2):
    cor_0 = (-0.5*C_hx*l1*s2*torch.sin(2*s3 - 2*s7) + 0.5*C_hx*l2*s4*torch.sin(s3 - s5) - 0.5*C_hx*l2*s4*torch.sin(s3 + s5 - 2*s7) - 
            0.5*C_hx*ls*s6*torch.sin(al - s3 + s7) - 0.5*C_hx*ls*s6*torch.sin(al + s3 - s7) + 0.5*C_hx*s1*torch.cos(2*s3 - 2*s7) + 0.5*C_hx*s1 + 1.0*C_hy*b*s6*torch.sin(s3 - s7) 
            + 0.5*C_hy*l1*s2*torch.sin(2*s3 - 2*s7) + 0.5*C_hy*l2*s4*torch.sin(s3 - s5) + 0.5*C_hy*l2*s4*torch.sin(s3 + s5 - 2*s7) - 0.5*C_hy*ls*s6*torch.sin(al - s3 + s7) 
            + 0.5*C_hy*ls*s6*torch.sin(al + s3 - s7) - 0.5*C_hy*s1*torch.cos(2*s3 - 2*s7) + 0.5*C_hy*s1 + 1.0*C_l1x*s1 - 0.5*C_l2x*l1*s2*torch.sin(2*s3 - 2*s5) 
            + 0.5*C_l2x*s1*torch.cos(2*s3 - 2*s5) + 0.5*C_l2x*s1 + 0.5*C_l2y*l1*s2*torch.sin(2*s3 - 2*s5) + 0.5*C_l2y*l2*s4*torch.sin(s3 - s5) 
            - 0.5*C_l2y*s1*torch.cos(2*s3 - 2*s5) + 0.5*C_l2y*s1 + 0.5*C_lsx*l1*s2*torch.sin(2*al - 2*s3 + 2*s7) + 0.5*C_lsx*l2*s4*torch.sin(s3 - s5) 
            + 0.5*C_lsx*l2*s4*torch.sin(2*al - s3 - s5 + 2*s7) + 0.5*C_lsx*s1*torch.cos(2*al - 2*s3 + 2*s7) + 0.5*C_lsx*s1 - 0.5*C_lsy*l1*s2*torch.sin(2*al - 2*s3 + 2*s7) 
            + 0.5*C_lsy*l2*s4*torch.sin(s3 - s5) - 0.5*C_lsy*l2*s4*torch.sin(2*al - s3 - s5 + 2*s7) - 0.5*C_lsy*ls*s6*torch.sin(al - s3 + s7) 
            - 0.5*C_lsy*s1*torch.cos(2*al - 2*s3 + 2*s7) + 0.5*C_lsy*s1 - 0.5*b*ma_hx*s6**2*torch.cos(s3 - s7) - 0.5*b*ma_hx*s6**2*torch.cos(s3 + s7) 
            - 0.5*b*ma_hy*s6**2*torch.cos(s3 - s7) + 0.5*b*ma_hy*s6**2*torch.cos(s3 + s7) - 1.0*d_al*ls*ma_hx*s6*torch.cos(al - s3 + s7) - 1.0*d_al*ls*ma_hx*s6*torch.cos(al + s3 + s7) 
            - 1.0*d_al*ls*ma_hy*s6*torch.cos(al - s3 + s7) + 1.0*d_al*ls*ma_hy*s6*torch.cos(al + s3 + s7) - 0.5*d_al*ls*ma_lsx*s6*torch.cos(al - s3 + s7) 
            - 0.5*d_al*ls*ma_lsx*s6*torch.cos(al + s3 + s7) - 0.5*d_al*ls*ma_lsy*s6*torch.cos(al - s3 + s7) + 0.5*d_al*ls*ma_lsy*s6*torch.cos(al + s3 + s7) 
            - 0.5*l1*ma_hx*s2**2*torch.cos(2*s3) - 0.5*l1*ma_hx*s2**2 + 0.5*l1*ma_hy*s2**2*torch.cos(2*s3) - 0.5*l1*ma_hy*s2**2 - 0.25*l1*ma_l1x*s2**2*torch.cos(2*s3) 
            - 0.25*l1*ma_l1x*s2**2 + 0.25*l1*ma_l1y*s2**2*torch.cos(2*s3) - 0.25*l1*ma_l1y*s2**2 - 0.5*l1*ma_l2x*s2**2*torch.cos(2*s3) - 0.5*l1*ma_l2x*s2**2 
            + 0.5*l1*ma_l2y*s2**2*torch.cos(2*s3) - 0.5*l1*ma_l2y*s2**2 - 0.5*l1*ma_lsx*s2**2*torch.cos(2*s3) - 0.5*l1*ma_lsx*s2**2 + 0.5*l1*ma_lsy*s2**2*torch.cos(2*s3) 
            - 0.5*l1*ma_lsy*s2**2 - 0.5*l2*ma_hx*s4**2*torch.cos(s3 - s5) - 0.5*l2*ma_hx*s4**2*torch.cos(s3 + s5) - 0.5*l2*ma_hy*s4**2*torch.cos(s3 - s5) 
            + 0.5*l2*ma_hy*s4**2*torch.cos(s3 + s5) - 0.25*l2*ma_l2x*s4**2*torch.cos(s3 - s5) - 0.25*l2*ma_l2x*s4**2*torch.cos(s3 + s5) - 0.25*l2*ma_l2y*s4**2*torch.cos(s3 - s5) 
            + 0.25*l2*ma_l2y*s4**2*torch.cos(s3 + s5) - 0.5*l2*ma_lsx*s4**2*torch.cos(s3 - s5) - 0.5*l2*ma_lsx*s4**2*torch.cos(s3 + s5) - 0.5*l2*ma_lsy*s4**2*torch.cos(s3 - s5) 
            + 0.5*l2*ma_lsy*s4**2*torch.cos(s3 + s5) - 0.5*ls*ma_hx*s6**2*torch.cos(al - s3 + s7) - 0.5*ls*ma_hx*s6**2*torch.cos(al + s3 + s7) 
            - 0.5*ls*ma_hy*s6**2*torch.cos(al - s3 + s7) + 0.5*ls*ma_hy*s6**2*torch.cos(al + s3 + s7) - 0.25*ls*ma_lsx*s6**2*torch.cos(al - s3 + s7) 
            - 0.25*ls*ma_lsx*s6**2*torch.cos(al + s3 + s7) - 0.25*ls*ma_lsy*s6**2*torch.cos(al - s3 + s7) + 0.25*ls*ma_lsy*s6**2*torch.cos(al + s3 + s7) 
            - 0.5*ma_hx*s1*s2*torch.sin(2*s3) + 0.5*ma_hy*s1*s2*torch.sin(2*s3) - 0.5*ma_l1x*s1*s2*torch.sin(2*s3) + 0.5*ma_l1y*s1*s2*torch.sin(2*s3) - 0.5*ma_l2x*s1*s2*torch.sin(2*s3)
            + 0.5*ma_l2y*s1*s2*torch.sin(2*s3) - 0.5*ma_lsx*s1*s2*torch.sin(2*s3) + 0.5*ma_lsy*s1*s2*torch.sin(2*s3))
    
    cor_1=(l1*(1.0*C_hx*l1*s2*torch.sin(s3)**2  - 2.0*C_hx*l1*s2*torch.sin(s3)*torch.sin(s7)*torch.cos(s3 - s7) + 1.0*C_hx*l1*s2*torch.sin(s7)**2 
            + 1.0*C_hx*l2*s4*torch.sin(s3)*torch.sin(s5) - 1.0*C_hx*l2*s4*torch.sin(s7)*torch.sin(s3 + s5 - s7) + 1.0*C_hx*ls*s6*torch.sin(al)*torch.sin(s3 - s7) 
            - 2.0*C_hx*s1*torch.sin(s3)*torch.sin(s7)*torch.sin(s3 - s7) - 0.5*C_hx*s1*torch.sin(2*s3) + 0.5*C_hx*s1*torch.sin(2*s7) + 1.0*C_hy*b*s6*torch.cos(s3 - s7) 
            - 1.0*C_hy*l1*s2*torch.sin(s3)**2 + 2.0*C_hy*l1*s2*torch.sin(s3)*torch.sin(s7)*torch.cos(s3 - s7) - 1.0*C_hy*l1*s2*torch.sin(s7)**2 + 1.0*C_hy*l1*s2 
            + 1.0*C_hy*l2*s4*torch.sin(s7)*torch.sin(s3 + s5 - s7) + 1.0*C_hy*l2*s4*torch.cos(s3)*torch.cos(s5) + 1.0*C_hy*ls*s6*torch.cos(al)*torch.cos(s3 - s7) 
            + 2.0*C_hy*s1*torch.sin(s3)*torch.sin(s7)*torch.sin(s3 - s7) + 0.5*C_hy*s1*torch.sin(2*s3) - 0.5*C_hy*s1*torch.sin(2*s7) + 0.25*C_l1y*l1*s2 + 1.0*C_l2x*l1*s2*torch.sin(s3)**2 
            - 2.0*C_l2x*l1*s2*torch.sin(s3)*torch.sin(s5)*torch.cos(s3 - s5) + 1.0*C_l2x*l1*s2*torch.sin(s5)**2 - 2.0*C_l2x*s1*torch.sin(s3)*torch.sin(s5)*torch.sin(s3 - s5) 
            - 0.5*C_l2x*s1*torch.sin(2*s3) + 0.5*C_l2x*s1*torch.sin(2*s5) - 1.0*C_l2y*l1*s2*torch.sin(s3)**2 + 2.0*C_l2y*l1*s2*torch.sin(s3)*torch.sin(s5)*torch.cos(s3 - s5) 
            - 1.0*C_l2y*l1*s2*torch.sin(s5)**2 + 1.0*C_l2y*l1*s2 + 0.5*C_l2y*l2*s4*torch.cos(s3 - s5) + 2.0*C_l2y*s1*torch.sin(s3)*torch.sin(s5)*torch.sin(s3 - s5) + 0.5*C_l2y*s1*torch.sin(2*s3) 
            - 0.5*C_l2y*s1*torch.sin(2*s5) + 1.0*C_lsx*l1*s2*torch.sin(al)**2 + 4.0*C_lsx*l1*s2*torch.sin(al)*torch.sin(s3)*torch.sin(s7)*torch.sin(al - s3 + s7) 
            - 2.0*C_lsx*l1*s2*torch.sin(al)*torch.sin(s3)*torch.cos(al - s3) + 2.0*C_lsx*l1*s2*torch.sin(al)*torch.sin(s7)*torch.cos(al + s7) + 1.0*C_lsx*l1*s2*torch.sin(s3)**2 
            - 2.0*C_lsx*l1*s2*torch.sin(s3)*torch.sin(s7)*torch.cos(s3 - s7) + 1.0*C_lsx*l1*s2*torch.sin(s7)**2 - 0.25*C_lsx*l2*s4*(torch.cos(s3 + s5 - 2*s7) - torch.cos(s3 + s5 + 2*s7)) 
            + 0.125*C_lsx*l2*s4*(torch.cos(-2*al + s3 + s5 + 2*s7) - torch.cos(2*al - s3 - s5 + 2*s7) + torch.cos(2*al + s3 + s5 - 2*s7) - torch.cos(2*al + s3 + s5 + 2*s7)) 
            - 2.0*C_lsx*l2*s4*torch.sin(al)**2*torch.sin(s7)**2*torch.cos(s3 + s5) + 2.0*C_lsx*l2*s4*torch.sin(al)*torch.sin(s7)*torch.sin(al + s7)*torch.sin(s3 + s5) 
            + 1.0*C_lsx*l2*s4*torch.sin(s7)**2*torch.cos(s3 + s5) - 1.0*C_lsx*l2*s4*torch.cos(al)*torch.cos(-al + s3 + s5) + 1.0*C_lsx*l2*s4*torch.cos(s3)*torch.cos(s5) 
            + 0.125*C_lsx*s1*(torch.sin(-2*al + 2*s3 + 2*s7) + torch.sin(2*al - 2*s3 + 2*s7) + torch.sin(2*al + 2*s3 - 2*s7) - torch.sin(2*al + 2*s3 + 2*s7)) 
            - 4.0*C_lsx*s1*torch.sin(al)**2*torch.sin(s3)*torch.sin(s7)**2*torch.cos(s3) + 4.0*C_lsx*s1*torch.sin(al)*torch.sin(s3)**2*torch.sin(s7)*torch.sin(al + s7) 
            - 1.5*C_lsx*s1*torch.sin(al)*torch.sin(s7)*torch.sin(al + s7) + 0.25*C_lsx*s1*torch.sin(2*al) - 2.0*C_lsx*s1*torch.sin(s3)**2*torch.sin(s7)*torch.cos(s7)
                + 2.0*C_lsx*s1*torch.sin(s3)*torch.sin(s7)**2*torch.cos(s3) - 2.0*C_lsx*s1*torch.sin(s3)*torch.cos(al)*torch.cos(al - s3) + 0.5*C_lsx*s1*torch.sin(2*s3) 
                + 0.25*C_lsx*s1*torch.sin(2*s7) + 0.5*C_lsx*s1*torch.sin(al + s7)*torch.cos(al)*torch.cos(s7) - 1.0*C_lsy*l1*s2*torch.sin(al)**2 
                - 4.0*C_lsy*l1*s2*torch.sin(al)*torch.sin(s3)*torch.sin(s7)*torch.sin(al - s3 + s7) + 2.0*C_lsy*l1*s2*torch.sin(al)*torch.sin(s3)*torch.cos(al - s3) 
                - 2.0*C_lsy*l1*s2*torch.sin(al)*torch.sin(s7)*torch.cos(al + s7) - 1.0*C_lsy*l1*s2*torch.sin(s3)**2 + 2.0*C_lsy*l1*s2*torch.sin(s3)*torch.sin(s7)*torch.cos(s3 - s7) 
                - 1.0*C_lsy*l1*s2*torch.sin(s7)**2 + 1.0*C_lsy*l1*s2 + 0.25*C_lsy*l2*s4*(torch.cos(s3 + s5 - 2*s7) - torch.cos(s3 + s5 + 2*s7)) 
                - 0.125*C_lsy*l2*s4*(torch.cos(-2*al + s3 + s5 + 2*s7) - torch.cos(2*al - s3 - s5 + 2*s7) + torch.cos(2*al + s3 + s5 - 2*s7) - torch.cos(2*al + s3 + s5 + 2*s7)) 
                + 2.0*C_lsy*l2*s4*torch.sin(al)**2*torch.sin(s7)**2*torch.cos(s3 + s5) - 2.0*C_lsy*l2*s4*torch.sin(al)*torch.sin(s7)*torch.sin(al + s7)*torch.sin(s3 + s5) 
                + 1.0*C_lsy*l2*s4*torch.sin(s3)*torch.sin(s5) - 1.0*C_lsy*l2*s4*torch.sin(s7)**2*torch.cos(s3 + s5) + 1.0*C_lsy*l2*s4*torch.cos(al)*torch.cos(-al + s3 + s5) 
                + 0.5*C_lsy*ls*s6*torch.cos(al - s3 + s7) - 0.125*C_lsy*s1*(torch.sin(-2*al + 2*s3 + 2*s7) + torch.sin(2*al - 2*s3 + 2*s7) + torch.sin(2*al + 2*s3 - 2*s7) 
            - torch.sin(2*al + 2*s3 + 2*s7)) + 4.0*C_lsy*s1*torch.sin(al)**2*torch.sin(s3)*torch.sin(s7)**2*torch.cos(s3) - 4.0*C_lsy*s1*torch.sin(al)*torch.sin(s3)**2*torch.sin(s7)*torch.sin(al + s7) 
            + 1.5*C_lsy*s1*torch.sin(al)*torch.sin(s7)*torch.sin(al + s7) - 0.25*C_lsy*s1*torch.sin(2*al) + 2.0*C_lsy*s1*torch.sin(s3)**2*torch.sin(s7)*torch.cos(s7) 
            - 2.0*C_lsy*s1*torch.sin(s3)*torch.sin(s7)**2*torch.cos(s3) + 2.0*C_lsy*s1*torch.sin(s3)*torch.cos(al)*torch.cos(al - s3) - 0.5*C_lsy*s1*torch.sin(2*s3) 
            - 0.25*C_lsy*s1*torch.sin(2*s7) - 0.5*C_lsy*s1*torch.sin(al + s7)*torch.cos(al)*torch.cos(s7) + 1.0*b*ma_hx*s6**2*torch.sin(s3)*torch.cos(s7) 
            - 1.0*b*ma_hy*s6**2*torch.sin(s7)*torch.cos(s3) + 2.0*d_al*ls*ma_hx*s6*torch.sin(s3)*torch.cos(al + s7) - 2.0*d_al*ls*ma_hy*s6*torch.sin(al + s7)*torch.cos(s3) 
            + 1.0*d_al*ls*ma_lsx*s6*torch.sin(s3)*torch.cos(al + s7) - 1.0*d_al*ls*ma_lsy*s6*torch.sin(al + s7)*torch.cos(s3) + 0.5*l1*ma_hx*s2**2*torch.sin(2*s3) 
            - 0.5*l1*ma_hy*s2**2*torch.sin(2*s3) + 0.125*l1*ma_l1x*s2**2*torch.sin(2*s3) - 0.125*l1*ma_l1y*s2**2*torch.sin(2*s3) + 0.5*l1*ma_l2x*s2**2*torch.sin(2*s3) 
            - 0.5*l1*ma_l2y*s2**2*torch.sin(2*s3) + 0.5*l1*ma_lsx*s2**2*torch.sin(2*s3) - 0.5*l1*ma_lsy*s2**2*torch.sin(2*s3) + 1.0*l2*ma_hx*s4**2*torch.sin(s3)*torch.cos(s5) 
            - 1.0*l2*ma_hy*s4**2*torch.sin(s5)*torch.cos(s3) + 0.5*l2*ma_l2x*s4**2*torch.sin(s3)*torch.cos(s5) - 0.5*l2*ma_l2y*s4**2*torch.sin(s5)*torch.cos(s3) 
            + 1.0*l2*ma_lsx*s4**2*torch.sin(s3)*torch.cos(s5) - 1.0*l2*ma_lsy*s4**2*torch.sin(s5)*torch.cos(s3) + 1.0*ls*ma_hx*s6**2*torch.sin(s3)*torch.cos(al + s7)
                - 1.0*ls*ma_hy*s6**2*torch.sin(al + s7)*torch.cos(s3) + 0.5*ls*ma_lsx*s6**2*torch.sin(s3)*torch.cos(al + s7) - 0.5*ls*ma_lsy*s6**2*torch.sin(al + s7)*torch.cos(s3) 
                + 1.0*ma_hx*s1*s2*torch.sin(s3)**2 - 1.0*ma_hy*s1*s2*torch.sin(s3)**2 + 1.0*ma_hy*s1*s2 + 0.5*ma_l1x*s1*s2*torch.sin(s3)**2 - 0.5*ma_l1y*s1*s2*torch.sin(s3)**2 
                + 0.5*ma_l1y*s1*s2 + 1.0*ma_l2x*s1*s2*torch.sin(s3)**2 - 1.0*ma_l2y*s1*s2*torch.sin(s3)**2 + 1.0*ma_l2y*s1*s2 + 1.0*ma_lsx*s1*s2*torch.sin(s3)**2 
                - 1.0*ma_lsy*s1*s2*torch.sin(s3)**2 + 1.0*ma_lsy*s1*s2))
    
    cor_2 =(l2*(1.0*C_hx*l1*s2*torch.sin(s3)*torch.sin(s5) - 1.0*C_hx*l1*s2*torch.sin(s7)*torch.sin(s3 + s5 - s7)
            + 1.0*C_hx*l2*s4*torch.sin(s5)**2 - 2.0*C_hx*l2*s4*torch.sin(s5)*torch.sin(s7)*torch.cos(s5 - s7) + 1.0*C_hx*l2*s4*torch.sin(s7)**2 
            + 1.0*C_hx*ls*s6*torch.sin(al)*torch.sin(s5 - s7) - 1.0*C_hx*s1*torch.sin(s5)*torch.cos(s3) + 1.0*C_hx*s1*torch.sin(s7)*torch.cos(s3 + s5 - s7) 
            + 1.0*C_hy*b*s6*torch.cos(s5 - s7) + 1.0*C_hy*l1*s2*torch.sin(s7)*torch.sin(s3 + s5 - s7) + 1.0*C_hy*l1*s2*torch.cos(s3)*torch.cos(s5) 
            - 1.0*C_hy*l2*s4*torch.sin(s5)**2 + 2.0*C_hy*l2*s4*torch.sin(s5)*torch.sin(s7)*torch.cos(s5 - s7) - 1.0*C_hy*l2*s4*torch.sin(s7)**2 
            + 1.0*C_hy*l2*s4 + 1.0*C_hy*ls*s6*torch.cos(al)*torch.cos(s5 - s7) + 1.0*C_hy*s1*torch.sin(s3)*torch.cos(s5) 
            - 1.0*C_hy*s1*torch.sin(s7)*torch.cos(s3 + s5 - s7) + 0.5*C_l2y*l1*s2*torch.cos(s3 - s5) + 0.25*C_l2y*l2*s4 + 0.5*C_l2y*s1*torch.sin(s3 - s5) 
            - 0.25*C_lsx*l1*s2*(torch.cos(s3 + s5 - 2*s7) - torch.cos(s3 + s5 + 2*s7)) + 0.125*C_lsx*l1*s2*(torch.cos(-2*al + s3 + s5 + 2*s7) 
            - torch.cos(2*al - s3 - s5 + 2*s7) + torch.cos(2*al + s3 + s5 - 2*s7) - torch.cos(2*al + s3 + s5 + 2*s7)) 
            - 2.0*C_lsx*l1*s2*torch.sin(al)**2*torch.sin(s7)**2*torch.cos(s3 + s5) + 2.0*C_lsx*l1*s2*torch.sin(al)*torch.sin(s7)*torch.sin(al + s7)*torch.sin(s3 + s5) 
            + 1.0*C_lsx*l1*s2*torch.sin(s7)**2*torch.cos(s3 + s5) - 1.0*C_lsx*l1*s2*torch.cos(al)*torch.cos(-al + s3 + s5) + 1.0*C_lsx*l1*s2*torch.cos(s3)*torch.cos(s5) 
            + 1.0*C_lsx*l2*s4*torch.sin(al)**2 + 4.0*C_lsx*l2*s4*torch.sin(al)*torch.sin(s5)*torch.sin(s7)*torch.sin(al - s5 + s7) 
            - 2.0*C_lsx*l2*s4*torch.sin(al)*torch.sin(s5)*torch.cos(al - s5) + 2.0*C_lsx*l2*s4*torch.sin(al)*torch.sin(s7)*torch.cos(al + s7) + 1.0*C_lsx*l2*s4*torch.sin(s5)**2 
            - 2.0*C_lsx*l2*s4*torch.sin(s5)*torch.sin(s7)*torch.cos(s5 - s7) + 1.0*C_lsx*l2*s4*torch.sin(s7)**2 + 0.25*C_lsx*s1*(torch.sin(2*al - s3 - s5 + 2*s7) 
            + torch.sin(2*al + s3 + s5 + 2*s7)) + 1.0*C_lsx*s1*torch.sin(al)**2*torch.sin(s3 + s5) + 2.0*C_lsx*s1*torch.sin(al)*torch.sin(s7)*torch.sin(s3 + s5)*torch.cos(al + s7) 
            - 1.0*C_lsx*s1*torch.sin(s5)*torch.cos(s3) + 1.0*C_lsx*s1*torch.sin(s7)**2*torch.sin(s3 + s5) + 0.25*C_lsy*l1*s2*(torch.cos(s3 + s5 - 2*s7) - torch.cos(s3 + s5 + 2*s7)) 
            - 0.125*C_lsy*l1*s2*(torch.cos(-2*al + s3 + s5 + 2*s7) - torch.cos(2*al - s3 - s5 + 2*s7) + torch.cos(2*al + s3 + s5 - 2*s7) - torch.cos(2*al + s3 + s5 + 2*s7)) 
            + 2.0*C_lsy*l1*s2*torch.sin(al)**2*torch.sin(s7)**2*torch.cos(s3 + s5) - 2.0*C_lsy*l1*s2*torch.sin(al)*torch.sin(s7)*torch.sin(al + s7)*torch.sin(s3 + s5) 
            + 1.0*C_lsy*l1*s2*torch.sin(s3)*torch.sin(s5) - 1.0*C_lsy*l1*s2*torch.sin(s7)**2*torch.cos(s3 + s5) + 1.0*C_lsy*l1*s2*torch.cos(al)*torch.cos(-al + s3 + s5) 
            - 1.0*C_lsy*l2*s4*torch.sin(al)**2 - 4.0*C_lsy*l2*s4*torch.sin(al)*torch.sin(s5)*torch.sin(s7)*torch.sin(al - s5 + s7) + 2.0*C_lsy*l2*s4*torch.sin(al)*torch.sin(s5)*torch.cos(al - s5) 
            - 2.0*C_lsy*l2*s4*torch.sin(al)*torch.sin(s7)*torch.cos(al + s7) - 1.0*C_lsy*l2*s4*torch.sin(s5)**2 + 2.0*C_lsy*l2*s4*torch.sin(s5)*torch.sin(s7)*torch.cos(s5 - s7) 
            - 1.0*C_lsy*l2*s4*torch.sin(s7)**2 + 1.0*C_lsy*l2*s4 + 0.5*C_lsy*ls*s6*torch.cos(al - s5 + s7) - 0.25*C_lsy*s1*(torch.sin(2*al - s3 - s5 + 2*s7) + torch.sin(2*al + s3 + s5 + 2*s7)) 
            - 1.0*C_lsy*s1*torch.sin(al)**2*torch.sin(s3 + s5) - 2.0*C_lsy*s1*torch.sin(al)*torch.sin(s7)*torch.sin(s3 + s5)*torch.cos(al + s7) + 1.0*C_lsy*s1*torch.sin(s3)*torch.cos(s5) 
            - 1.0*C_lsy*s1*torch.sin(s7)**2*torch.sin(s3 + s5) + 1.0*b*ma_hx*s6**2*torch.sin(s5)*torch.cos(s7) - 1.0*b*ma_hy*s6**2*torch.sin(s7)*torch.cos(s5) 
            + 2.0*d_al*ls*ma_hx*s6*torch.sin(s5)*torch.cos(al + s7) - 2.0*d_al*ls*ma_hy*s6*torch.sin(al + s7)*torch.cos(s5) + 1.0*d_al*ls*ma_lsx*s6*torch.sin(s5)*torch.cos(al + s7) 
            - 1.0*d_al*ls*ma_lsy*s6*torch.sin(al + s7)*torch.cos(s5) + 1.0*l1*ma_hx*s2**2*torch.sin(s5)*torch.cos(s3) - 1.0*l1*ma_hy*s2**2*torch.sin(s3)*torch.cos(s5) 
            + 0.5*l1*ma_l2x*s2**2*torch.sin(s5)*torch.cos(s3) - 0.5*l1*ma_l2y*s2**2*torch.sin(s3)*torch.cos(s5) + 1.0*l1*ma_lsx*s2**2*torch.sin(s5)*torch.cos(s3) 
            - 1.0*l1*ma_lsy*s2**2*torch.sin(s3)*torch.cos(s5) + 0.5*l2*ma_hx*s4**2*torch.sin(2*s5) - 0.5*l2*ma_hy*s4**2*torch.sin(2*s5) + 0.125*l2*ma_l2x*s4**2*torch.sin(2*s5) 
            - 0.125*l2*ma_l2y*s4**2*torch.sin(2*s5) + 0.5*l2*ma_lsx*s4**2*torch.sin(2*s5) - 0.5*l2*ma_lsy*s4**2*torch.sin(2*s5) + 1.0*ls*ma_hx*s6**2*torch.sin(s5)*torch.cos(al + s7) 
            - 1.0*ls*ma_hy*s6**2*torch.sin(al + s7)*torch.cos(s5) + 0.5*ls*ma_lsx*s6**2*torch.sin(s5)*torch.cos(al + s7) - 0.5*ls*ma_lsy*s6**2*torch.sin(al + s7)*torch.cos(s5) 
            + 1.0*ma_hx*s1*s2*torch.sin(s3)*torch.sin(s5) + 1.0*ma_hy*s1*s2*torch.cos(s3)*torch.cos(s5) + 0.5*ma_l2x*s1*s2*torch.sin(s3)*torch.sin(s5) + 0.5*ma_l2y*s1*s2*torch.cos(s3)*torch.cos(s5) 
            + 1.0*ma_lsx*s1*s2*torch.sin(s3)*torch.sin(s5) + 1.0*ma_lsy*s1*s2*torch.cos(s3)*torch.cos(s5)))
    
    cor_3 = (0.5*C_hx*l1*ls*s2*torch.cos(al - s3 + s7) - 0.5*C_hx*l1*ls*s2*torch.cos(al + s3 - s7) + 0.5*C_hx*l2*ls*s4*torch.cos(al - s5 + s7) 
            - 0.5*C_hx*l2*ls*s4*torch.cos(al + s5 - s7) - 0.5*C_hx*ls**2*s6*torch.cos(2*al) + 0.5*C_hx*ls**2*s6 - 0.5*C_hx*ls*s1*torch.sin(al - s3 + s7) 
            - 0.5*C_hx*ls*s1*torch.sin(al + s3 - s7) + 1.0*C_hy*b**2*s6 + 1.0*C_hy*b*l1*s2*torch.cos(s3 - s7) + 1.0*C_hy*b*l2*s4*torch.cos(s5 - s7) 
            + 2.0*C_hy*b*ls*s6*torch.cos(al) + 1.0*C_hy*b*s1*torch.sin(s3 - s7) + 0.5*C_hy*l1*ls*s2*torch.cos(al - s3 + s7) + 0.5*C_hy*l1*ls*s2*torch.cos(al + s3 - s7) 
            + 0.5*C_hy*l2*ls*s4*torch.cos(al - s5 + s7) + 0.5*C_hy*l2*ls*s4*torch.cos(al + s5 - s7) + 0.5*C_hy*ls**2*s6*torch.cos(2*al) + 0.5*C_hy*ls**2*s6 
            - 0.5*C_hy*ls*s1*torch.sin(al - s3 + s7) + 0.5*C_hy*ls*s1*torch.sin(al + s3 - s7) + 0.5*C_lsy*l1*ls*s2*torch.cos(al - s3 + s7) 
            + 0.5*C_lsy*l2*ls*s4*torch.cos(al - s5 + s7) + 0.25*C_lsy*ls**2*s6 - 0.5*C_lsy*ls*s1*torch.sin(al - s3 + s7) + 0.5*b**2*ma_hx*s6**2*torch.sin(2*s7) 
            - 0.5*b**2*ma_hy*s6**2*torch.sin(2*s7) - 1.0*b*d_al*ls*ma_hx*s6*torch.sin(al) + 1.0*b*d_al*ls*ma_hx*s6*torch.sin(al + 2*s7) 
            - 1.0*b*d_al*ls*ma_hy*s6*torch.sin(al) - 1.0*b*d_al*ls*ma_hy*s6*torch.sin(al + 2*s7) - 0.5*b*l1*ma_hx*s2**2*torch.sin(s3 - s7) + 0.5*b*l1*ma_hx*s2**2*torch.sin(s3 + s7) 
            - 0.5*b*l1*ma_hy*s2**2*torch.sin(s3 - s7) - 0.5*b*l1*ma_hy*s2**2*torch.sin(s3 + s7) - 0.5*b*l2*ma_hx*s4**2*torch.sin(s5 - s7) + 0.5*b*l2*ma_hx*s4**2*torch.sin(s5 + s7) 
            - 0.5*b*l2*ma_hy*s4**2*torch.sin(s5 - s7) - 0.5*b*l2*ma_hy*s4**2*torch.sin(s5 + s7) + 1.0*b*ls*ma_hx*s6**2*torch.sin(al + 2*s7) - 1.0*b*ls*ma_hy*s6**2*torch.sin(al + 2*s7)
            + 0.5*b*ma_hx*s1*s2*torch.cos(s3 - s7) - 0.5*b*ma_hx*s1*s2*torch.cos(s3 + s7) + 0.5*b*ma_hy*s1*s2*torch.cos(s3 - s7) + 0.5*b*ma_hy*s1*s2*torch.cos(s3 + s7) 
            + 1.0*d_al*ls**2*ma_hx*s6*torch.sin(2*al + 2*s7) - 1.0*d_al*ls**2*ma_hy*s6*torch.sin(2*al + 2*s7) + 0.25*d_al*ls**2*ma_lsx*s6*torch.sin(2*al + 2*s7) 
            - 0.25*d_al*ls**2*ma_lsy*s6*torch.sin(2*al + 2*s7) + 0.5*l1*ls*ma_hx*s2**2*torch.sin(al - s3 + s7) + 0.5*l1*ls*ma_hx*s2**2*torch.sin(al + s3 + s7) 
            + 0.5*l1*ls*ma_hy*s2**2*torch.sin(al - s3 + s7) - 0.5*l1*ls*ma_hy*s2**2*torch.sin(al + s3 + s7) + 0.25*l1*ls*ma_lsx*s2**2*torch.sin(al - s3 + s7) 
            + 0.25*l1*ls*ma_lsx*s2**2*torch.sin(al + s3 + s7) + 0.25*l1*ls*ma_lsy*s2**2*torch.sin(al - s3 + s7) - 0.25*l1*ls*ma_lsy*s2**2*torch.sin(al + s3 + s7) 
            + 0.5*l2*ls*ma_hx*s4**2*torch.sin(al - s5 + s7) + 0.5*l2*ls*ma_hx*s4**2*torch.sin(al + s5 + s7) + 0.5*l2*ls*ma_hy*s4**2*torch.sin(al - s5 + s7) 
            - 0.5*l2*ls*ma_hy*s4**2*torch.sin(al + s5 + s7) + 0.25*l2*ls*ma_lsx*s4**2*torch.sin(al - s5 + s7) + 0.25*l2*ls*ma_lsx*s4**2*torch.sin(al + s5 + s7) 
            + 0.25*l2*ls*ma_lsy*s4**2*torch.sin(al - s5 + s7) - 0.25*l2*ls*ma_lsy*s4**2*torch.sin(al + s5 + s7) + 0.5*ls**2*ma_hx*s6**2*torch.sin(2*al + 2*s7) 
            - 0.5*ls**2*ma_hy*s6**2*torch.sin(2*al + 2*s7) + 0.125*ls**2*ma_lsx*s6**2*torch.sin(2*al + 2*s7) - 0.125*ls**2*ma_lsy*s6**2*torch.sin(2*al + 2*s7) 
            + 0.5*ls*ma_hx*s1*s2*torch.cos(al - s3 + s7) - 0.5*ls*ma_hx*s1*s2*torch.cos(al + s3 + s7) + 0.5*ls*ma_hy*s1*s2*torch.cos(al - s3 + s7) 
            + 0.5*ls*ma_hy*s1*s2*torch.cos(al + s3 + s7) + 0.25*ls*ma_lsx*s1*s2*torch.cos(al - s3 + s7) - 0.25*ls*ma_lsx*s1*s2*torch.cos(al + s3 + s7) 
            + 0.25*ls*ma_lsy*s1*s2*torch.cos(al - s3 + s7) + 0.25*ls*ma_lsy*s1*s2*torch.cos(al + s3 + s7))
    
    return torch.stack([cor_0, cor_1, cor_2, cor_3],dim=1)

def gravity_vector_torch(s1, s2, s3, s4, s5, s6, s7, s8, s9, al, d_al, dd_al, dd_ph, l1, l2, ls, b, c, ma_l1x, ma_l1y, ma_l2x, ma_l2y, ma_hx, ma_hy, ma_lsx, ma_lsy, Ia_h, Ia_l1, Ia_l2, Ia_ls, I_r, C_hx, C_hy, C_l1x, C_l1y, C_l2x, C_l2y, C_lsx, C_lsy, K_1, K_2):
    gv_0= (ls*(-1.0*C_hx*d_al*torch.sin(al)*torch.cos(s3 - s7) + 1.0*C_hy*d_al*torch.sin(s3 - s7)*torch.cos(al) - 0.5*C_lsy*d_al*torch.sin(al - s3 + s7) 
            - 1.0*d_al**2*ma_hx*torch.cos(s3)*torch.cos(al + s7) - 1.0*d_al**2*ma_hy*torch.sin(s3)*torch.sin(al + s7) - 0.5*d_al**2*ma_lsx*torch.cos(s3)*torch.cos(al + s7) 
            - 0.5*d_al**2*ma_lsy*torch.sin(s3)*torch.sin(al + s7) - 1.0*dd_al*ma_hx*torch.sin(al + s7)*torch.cos(s3) + 1.0*dd_al*ma_hy*torch.sin(s3)*torch.cos(al + s7) 
            - 0.5*dd_al*ma_lsx*torch.sin(al + s7)*torch.cos(s3) + 0.5*dd_al*ma_lsy*torch.sin(s3)*torch.cos(al + s7)))
    
    gv_1= (1.0*C_hx*d_al*l1*ls*torch.sin(al)*torch.sin(s3 - s7) 
            + 1.0*C_hy*d_al*l1*ls*torch.cos(al)*torch.cos(s3 - s7) + 0.5*C_lsy*d_al*l1*ls*torch.cos(al - s3 + s7) + 1.0*K_1*s3 
            - 1.0*K_1*s5 + 1.0*d_al**2*l1*ls*ma_hx*torch.sin(s3)*torch.cos(al + s7) - 1.0*d_al**2*l1*ls*ma_hy*torch.sin(al + s7)*torch.cos(s3)
            + 0.5*d_al**2*l1*ls*ma_lsx*torch.sin(s3)*torch.cos(al + s7) - 0.5*d_al**2*l1*ls*ma_lsy*torch.sin(al + s7)*torch.cos(s3) 
            + 1.0*dd_al*l1*ls*ma_hx*torch.sin(s3)*torch.sin(al + s7) + 1.0*dd_al*l1*ls*ma_hy*torch.cos(s3)*torch.cos(al + s7) 
            + 0.5*dd_al*l1*ls*ma_lsx*torch.sin(s3)*torch.sin(al + s7) + 0.5*dd_al*l1*ls*ma_lsy*torch.cos(s3)*torch.cos(al + s7))
    
    gv_2= (1.0*C_hx*d_al*l2*ls*torch.sin(al)*torch.sin(s5 - s7) + 1.0*C_hy*d_al*l2*ls*torch.cos(al)*torch.cos(s5 - s7) + 0.5*C_lsy*d_al*l2*ls*torch.cos(al - s5 + s7) 
            - 1.0*K_1*s3 + 1.0*K_1*s5 - 1.0*K_2*al + 1.0*K_2*s5 - 1.0*K_2*s7 + 1.0*d_al**2*l2*ls*ma_hx*torch.sin(s5)*torch.cos(al + s7) 
            - 1.0*d_al**2*l2*ls*ma_hy*torch.sin(al + s7)*torch.cos(s5) + 0.5*d_al**2*l2*ls*ma_lsx*torch.sin(s5)*torch.cos(al + s7) 
            - 0.5*d_al**2*l2*ls*ma_lsy*torch.sin(al + s7)*torch.cos(s5) + 1.0*dd_al*l2*ls*ma_hx*torch.sin(s5)*torch.sin(al + s7) 
            + 1.0*dd_al*l2*ls*ma_hy*torch.cos(s5)*torch.cos(al + s7) + 0.5*dd_al*l2*ls*ma_lsx*torch.sin(s5)*torch.sin(al + s7) 
            + 0.5*dd_al*l2*ls*ma_lsy*torch.cos(s5)*torch.cos(al + s7))
    
    gv_3= (-0.25*C_hx*d_al*ls**2*(torch.cos(2*al) - torch.cos(2*al + 4*s7)) 
            + 1.0*C_hx*d_al*ls**2*torch.sin(s7)**2*torch.cos(al + s7)**2 + 1.0*C_hx*d_al*ls**2*torch.sin(al + s7)**2*torch.cos(s7)**2 
            + 1.0*C_hy*b*d_al*ls*torch.sin(s7)**3*torch.sin(al + s7) + 1.0*C_hy*b*d_al*ls*torch.sin(s7)**2*torch.cos(s7)*torch.cos(al + s7) 
            + 1.0*C_hy*b*d_al*ls*torch.sin(s7)*torch.sin(al + s7)*torch.cos(s7)**2 + 1.0*C_hy*b*d_al*ls*torch.cos(s7)**3*torch.cos(al + s7) 
            + 0.25*C_hy*d_al*ls**2*(torch.cos(2*al) - torch.cos(2*al + 4*s7)) + 1.0*C_hy*d_al*ls**2*torch.sin(s7)**2*torch.sin(al + s7)**2 
            + 1.0*C_hy*d_al*ls**2*torch.cos(s7)**2*torch.cos(al + s7)**2 + 0.25*C_lsy*d_al*ls**2*torch.sin(al + s7)**4 + 0.5*C_lsy*d_al*ls**2*torch.sin(al + s7)**2*torch.cos(al + s7)**2 
            + 0.25*C_lsy*d_al*ls**2*torch.cos(al + s7)**4 + 1.0*I_r*dd_ph + 1.0*Ia_ls*dd_al + 1.0*K_2*al - 1.0*K_2*s5 + 1.0*K_2*s7 + 1.0*b*d_al**2*ls*ma_hx*torch.sin(s7)*torch.cos(al + s7) 
            - 1.0*b*d_al**2*ls*ma_hy*torch.sin(al + s7)*torch.cos(s7) + 1.0*b*dd_al*ls*ma_hx*torch.sin(s7)*torch.sin(al + s7) + 1.0*b*dd_al*ls*ma_hy*torch.cos(s7)*torch.cos(al + s7) 
            + 0.5*d_al**2*ls**2*ma_hx*torch.sin(2*al + 2*s7) - 0.5*d_al**2*ls**2*ma_hy*torch.sin(2*al + 2*s7) + 0.125*d_al**2*ls**2*ma_lsx*torch.sin(2*al + 2*s7) 
            - 0.125*d_al**2*ls**2*ma_lsy*torch.sin(2*al + 2*s7) + 1.0*dd_al*ls**2*ma_hx*torch.sin(al + s7)**2 + 1.0*dd_al*ls**2*ma_hy*torch.cos(al + s7)**2 
            + 0.25*dd_al*ls**2*ma_lsx*torch.sin(al + s7)**2 + 0.25*dd_al*ls**2*ma_lsy*torch.cos(al + s7)**2)
    
    return torch.stack([gv_0, gv_1, gv_2, gv_3],dim=1)
    