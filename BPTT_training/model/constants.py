# constants.py
#!usr/bin/env python3
import numpy as np

class Constants:
    # You can keep defaults here, override if needed
    rho: float = 997.0
    AM: float = 1.0

    # geometry / mass
    b: float = 0.075
    bs: float = 0.035
    m_h: float = 0.44

    l1: float = 0.048
    l2: float = 0.048
    ls: float = 0.015

    m_l1: float = 0.01 / 2
    m_l2: float = 0.01 / 2
    m_ls: float = 0.01

    c: float = 0.015
    m_r: float = 0.1

    # dissipation
    C_hx: float = 0.46
    C_hy: float = 10
    C_l1x: float =0# 0.085
    C_l1y: float = 10#7.0
    C_l2x: float = 0#0.085
    C_l2y: float = 10#1.0
    C_lsx: float = 0#0.085
    C_lsy: float = 0#1.0

    # C_hx = 0.46
# C_hy = 10
# C_lx = 0
# C_ly = 10
# K_1 = 0.4           # stiffness (Nm/rad)
# K_2 = 0.7 

    # stiffness
    K_1: float = 0.4
    K_2: float = 0.7

    def as_tuple(self):
        rho, AM = self.rho, self.AM
        b, bs, m_h = self.b, self.bs, self.m_h

        # head inertia + added mass/inertia
        I_h  = m_h * (b**2 + bs**2) / 4
        ma_hx = m_h + AM * np.pi * rho * (bs**2) * b
        ma_hy = m_h + AM * np.pi * rho * (b**3)
        Ia_h  = I_h + AM * (1/8) * np.pi * rho * b * (b**2 - bs**2)**2

        # link 1
        l1, m_l1 = self.l1, self.m_l1
        I_l1  = (1/12) * m_l1 * l1**2
        ma_l1x = m_l1 + AM * 0.0
        ma_l1y = m_l1 + AM * np.pi * rho * 0.075 * (l1/2)**2
        Ia_l1  = I_l1 + AM * (1/8) * np.pi * rho * 0.075 * (l1/2)**4

        # link 2
        l2, m_l2 = self.l2, self.m_l2
        I_l2  = (1/12) * m_l2 * l2**2
        ma_l2x = m_l2 + AM * 0.0
        ma_l2y = m_l2 + AM * np.pi * rho * 0.075 * (l2/2)**2
        Ia_l2  = I_l2 + AM * (1/8) * np.pi * rho * 0.075 * (l2/2)**4

        # short link
        ls, m_ls = self.ls, self.m_ls
        I_ls  = (1/12) * m_ls * ls**2
        ma_lsx = m_ls + AM * 0.0
        ma_lsy = m_ls + AM * np.pi * rho * 0.075 * (ls/2)**2
        Ia_ls  = I_ls + AM * (1/8) * np.pi * rho * 0.075 * (ls/2)**4

        # rotor inertia
        c, m_r = self.c, self.m_r
        I_r = m_r * 0.027**2

        Length = (l1, l2, ls, b, c)
        Stiffness = (self.K_1, self.K_2)
        AddedMass = (ma_l1x, ma_l1y, ma_l2x, ma_l2y, ma_hx, ma_hy, ma_lsx, ma_lsy)
        AddedInertia = (Ia_h, Ia_l1, Ia_l2, Ia_ls, I_r)
        Dissipation = (self.C_hx, self.C_hy, self.C_l1x, self.C_l1y,
                       self.C_l2x, self.C_l2y, self.C_lsx, self.C_lsy)

        return (*Length, *AddedMass, *AddedInertia, *Dissipation, *Stiffness)
