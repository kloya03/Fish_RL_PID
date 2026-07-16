# %%

# %%
# --- standard library ---
import os
from datetime import datetime
import copy
import math
import random
import time
import csv

# --- third-party ---
import numpy as np
from tqdm import tqdm

# from tqdm import tqdm

# --- torch ---
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# --- local project ---
from model.CS_4link_v4_torch import ChaplyginSleighModelTorch

from model.constants import Constants
from model.model_integrator_v import RK4Integrator
from controller.pure_pursuit_vecb import PurePursuitClassic
from utils.geometry_curvature import PathGeometry as ptg

from model.networks import Actor
torch.set_num_threads(8)

# %% [markdown]
# ### CTE

# %%
def compute_signed_cteb(x, y, ref_path):
    """
    x, y : (B,)
    ref_path : (B, M, 2)

    Returns:
        e_cte : (B,)
    """

    B, M, _ = ref_path.shape

    # -------------------------------------------------
    # Extract path components
    # -------------------------------------------------
    x_path = ref_path[:, :, 0]      # (B, M)
    y_path = ref_path[:, :, 1]      # (B, M)

    # -------------------------------------------------
    # Distance to all path points (per env)
    # -------------------------------------------------
    dx = x.unsqueeze(1) - x_path    # (B, M)
    dy = y.unsqueeze(1) - y_path    # (B, M)

    dist2 = dx**2 + dy**2           # (B, M)

    # -------------------------------------------------
    # Closest point per environment
    # -------------------------------------------------
    idx = torch.argmin(dist2, dim=1)    # (B,)

    batch_idx = torch.arange(B, device=x.device)

    x_ref = x_path[batch_idx, idx]     # (B,)
    y_ref = y_path[batch_idx, idx]     # (B,)

    # -------------------------------------------------
    # Tangent using next path point
    # -------------------------------------------------
    idx_next = torch.clamp(idx + 1, max=M-1)

    dx_path = x_path[batch_idx, idx_next] - x_ref
    dy_path = y_path[batch_idx, idx_next] - y_ref

    # Left normal
    normal_x = -dy_path
    normal_y = dx_path

    norm = torch.sqrt(normal_x**2 + normal_y**2) + 1e-6

    normal_x = normal_x / norm
    normal_y = normal_y / norm

    # -------------------------------------------------
    # Signed cross-track error
    # -------------------------------------------------
    error_x = x - x_ref
    error_y = y - y_ref

    e_cte = error_x * normal_x + error_y * normal_y

    return e_cte

# %% [markdown]
# ### Range 

# %%
def to_range_tanh(p, lo, hi):
    return lo + (hi - lo) * (torch.tanh(p) + 1) / 2


# %% [markdown]
# ### Throttle from s

# %%
def throttle_from_s(
    s,
    A_min=5.0, A_max=10.0,
    f_min_hz=1.5, f_max_hz=3.0,
    pA=1.0,
    pW2=1.0
):
    # s: (B,) or (B,1)

    sA  = s.pow(pA)
    sW2 = s.pow(pW2)

    A_des = A_min + (A_max - A_min) * sA

    w_min = 2.0 * torch.pi * f_min_hz
    w_max = 2.0 * torch.pi * f_max_hz

    w2_min = w_min**2
    w2_max = w_max**2

    w2_des = w2_min + (w2_max - w2_min) * sW2
    w_des  = torch.sqrt(w2_des)

    return A_des, w_des


# %% [markdown]
# ### Rate limit

# %%
def rate_limit(x, x_prev, rate_max, dt):
    delta = x - x_prev
    max_delta = rate_max * dt

    delta = torch.minimum(delta,  max_delta)
    delta = torch.maximum(delta, -max_delta)

    return x_prev + delta

# %% [markdown]
# ### PID step

# %%
def pid_step(e, e_int, e_prev, dt, Kp, Ki, Kd, alpha_min=None, alpha_max=None):
    # e, e_int, e_prev shape: (B,)

    e_int_new = e_int + dt * e

    de = ptg.wrap_to_pi(e - e_prev)
    e_dot = de / dt

    ctrl_alpha = Kp * e + Ki * e_int_new + Kd * e_dot

    # Remove NaNs safely (batched)
    ctrl_alpha = torch.where(
        torch.isfinite(ctrl_alpha),
        ctrl_alpha,
        torch.zeros_like(ctrl_alpha)
    )

    # Saturation (vectorized)
    if alpha_max is not None:
        ctrl_alpha = torch.clamp(ctrl_alpha, -alpha_max, alpha_max)

    if alpha_min is not None:
        ctrl_alpha = torch.clamp(ctrl_alpha, min=alpha_min)

    return ctrl_alpha, e_int_new, e

# %% [markdown]
# ### New clamp PID step

# %%
def pid_step_c(e, e_int, e_prev, dt, Kp, Ki, Kd, alpha_min=None, alpha_max=None):
    # e, e_int, e_prev shape: (B,)

    e_int_new = torch.clamp(e_int +  dt * e, min=alpha_min, max=alpha_max)

    de = ptg.wrap_to_pi(e - e_prev)
    e_dot = de / dt

    ctrl_alpha = Kp * e + Ki * e_int_new + Kd * e_dot

    # Remove NaNs safely (batched)
    ctrl_alpha = torch.where(
        torch.isfinite(ctrl_alpha),
        ctrl_alpha,
        torch.zeros_like(ctrl_alpha)
    )

    # Saturation (vectorized)
    if alpha_max is not None:
        ctrl_alpha = torch.clamp(ctrl_alpha, -alpha_max, alpha_max)

    if alpha_min is not None:
        ctrl_alpha = torch.clamp(ctrl_alpha, min=alpha_min)

    return ctrl_alpha, e_int_new, e

# %% [markdown]
# ### speed PD

# %%
def speed_pd_step(e_u, e_prev, dt, Kp, Kd):



    # Derivative (discrete)
    e_dot = (e_u - e_prev) / dt

    # PD control
    s = Kp * e_u + Kd * e_dot

    return s

# %% [markdown]
# ### speed PID

# %%
def speed_pid_step(e_u, e_int, e_prev, dt, Kp, Ki, Kd,
                   s_min=None, s_max=None):

    # --- Derivative ---
    e_dot = (e_u - e_prev) / dt

    # --- Candidate integral update ---
    e_int_candidate = e_int + dt * e_u

    # --- Unsaturated control ---
    s_unsat = Kp * e_u + Ki * e_int_candidate + Kd * e_dot

    # Remove NaNs safely
    s_unsat = torch.where(
        torch.isfinite(s_unsat),
        s_unsat,
        torch.zeros_like(s_unsat)
    )

    # --- Apply saturation ---
    s_sat = s_unsat

    if s_max is not None:
        s_sat = torch.clamp(s_sat, max=s_max)

    if s_min is not None:
        s_sat = torch.clamp(s_sat, min=s_min)

    # --- Anti-windup (freeze integrator if saturated) ---
    is_saturated = (s_sat != s_unsat)

    e_int_new = torch.where(
        is_saturated,
        e_int,            # freeze integrator
        e_int_candidate  # allow integration
    )

    return s_sat, e_int_new, e_u

# %% [markdown]
# ### compute head traj

# %%
def compute_head_vel_torch(stsN, alpha, d_alpha, xh, yh, dt, const_vals):
    """
    stsN : (B, state_dim)
    alpha, d_alpha, xh, yh : (B,)
    dt : scalar
    """

    # Constants (scalars)
    l1, l2, ls, b, c = const_vals[0:5]

    # Extract state components (B,)
    u  = stsN[:, 0]
    v  = stsN[:, 1]
    th = stsN[:, 2]
    q1 = stsN[:, 3]
    q2 = stsN[:, 4]
    q3 = stsN[:, 5]
    psi = stsN[:, 6]

    # Body velocity of chassis
    xdot = u * torch.cos(th) - 0.5 * l1 * v * torch.sin(th)
    ydot = u * torch.sin(th) + 0.5 * l1 * v * torch.cos(th)

    # Build vec (B,6)
    vec = torch.stack([
        xdot,
        ydot,
        v,
        q1,
        q3 + d_alpha,
        q3
    ], dim=1)  # (B,6)

    # Build Jacobian rows (B,6)

    row1 = torch.stack([
        torch.ones_like(u),
        torch.zeros_like(u),
        -0.5 * l1 * torch.sin(th),
        -l2 * torch.sin(q2),
        -ls * torch.sin(psi + alpha),
        -b  * torch.sin(psi),
    ], dim=1)

    row2 = torch.stack([
        torch.zeros_like(u),
        torch.ones_like(u),
         0.5 * l1 * torch.cos(th),
         l2 * torch.cos(q2),
         ls * torch.cos(psi + alpha),
         b  * torch.cos(psi),
    ], dim=1)

    # Compute head velocities (B,)
    xh_dot = torch.sum(row1 * vec, dim=1)
    yh_dot = torch.sum(row2 * vec, dim=1)

    # Integrate head position
    xh = xh + dt * xh_dot
    yh = yh + dt * yh_dot

    # Head frame velocities
    uh = xh_dot * torch.cos(psi) + yh_dot * torch.sin(psi)
    vh = -xh_dot * torch.sin(psi) + yh_dot * torch.cos(psi)

    return uh, vh, xh, yh

# %% [markdown]
# ### bptt environment

# %%
def environment_map_alpha_pid(ic, ref_path, tp, N,A, w,Kp_a, Ki_a, Kd_a,alpha_max,lookahead,policy_net,optimizer,H,B=16, chunk_loss_history=None, is_difficult=False):

    # -------------------------------------------------
    # Time
    # -------------------------------------------------
    T = torch.linspace(0, tp, N)
    dt = T[1] - T[0]

    # -------------------------------------------------
    # Batched Initial State
    # -------------------------------------------------
    ic = torch.as_tensor(ic, dtype=torch.float32)
    state = ic.unsqueeze(0).repeat(B, 1)   # (B, state_dim)

    state_dim = state.shape[1]

    # -------------------------------------------------
    # Logging (batched)
    # -------------------------------------------------
    state_history = torch.zeros(N, B, 12)
    loss_history  = torch.zeros(N, B, 2)
    loss_term     = torch.zeros(N, B, 3)
    gain_history  = torch.zeros(N, B, 3)
    state_aug_history = torch.zeros(N-1, B, 2)
    u_des_history = torch.zeros(N, B, 2)

    throttle_history = torch.zeros(N, B, 5)
    goal_history = torch.zeros(N, B, 2)

    # -------------------------------------------------
    # Constants + Model
    # -------------------------------------------------
    Const_vals = Constants().as_tuple()

    holder = {
        "alpha": torch.zeros(B),
        "dd_phi": torch.zeros(B)
    }

    def input_funcs(tt):
        z = torch.zeros(B)
        return torch.stack([
            holder["alpha"],
            z,
            z,
            holder["dd_phi"]
        ], dim=1)   # (B, 4)

    model = ChaplyginSleighModelTorch(Const_vals, input_funcs)

    # -------------------------------------------------
    # Head position states (batched)
    # -------------------------------------------------
    xh = torch.zeros(B)
    yh = torch.zeros(B)

    # -------------------------------------------------
    # Alpha PID states (batched)
    # -------------------------------------------------
    e_int  = torch.zeros(B)
    e_prev = torch.zeros(B)
    alpha  = torch.zeros(B)
    alpha_prev = torch.zeros(B)
    Kp_a_prev = torch.zeros(B)
    Ki_a_prev = torch.zeros(B)
    Kd_a_prev = torch.zeros(B)
    Kp_s_prev = torch.zeros(B)
    Kd_s_prev = torch.zeros(B)

    alpha_rate_max = torch.full((B,), 1.0)

    # -------------------------------------------------
    # Speed control states (batched)
    # -------------------------------------------------
    u_prev   = torch.zeros(B)
    e_u_int  = torch.zeros(B)
    e_u_prev = torch.zeros(B)

    u_des_f = torch.zeros(B)

    # -------------------------------------------------
    # Lookahead filter state
    # -------------------------------------------------
    Ld_f = torch.full((B,), lookahead)

    # -------------------------------------------------
    # Throttle initial values (batched)
    # -------------------------------------------------
    A_des = torch.full((B,), A)
    w_des = torch.full((B,), w)
    A_prev = torch.full((B,), A)   # initial motor amplitude
    w_prev = torch.full((B,), w)   # initial motor frequency
    A_des_f = torch.full((B,), A)
    w_des_f = torch.full((B,), w)

    # A_rate_max = torch.full((B,), 2.0)
    # w_rate_max = torch.full((B,), 0.5 * 2 * torch.pi)

    phase = torch.zeros(B)

    # -------------------------------------------------
    # Speed decision parameters (scalars OK)
    # -------------------------------------------------
    c_kappa = 0.1
    c_e     = 5.0
    u_min   = 0.1
    u_max   = 0.5

    # -------------------------------------------------
    # Loss accumulation
    # -------------------------------------------------
    total_loss = torch.zeros(B)
    chunk_loss = None
    step_loss  = None

    # -------------------------------------------------
    # Integrator + Pure Pursuit
    # -------------------------------------------------
    integrator = RK4Integrator(model, dt)
    pp_obj = PurePursuitClassic()
    noise = torch.randn(B)
    
    Ld_f = torch.full((B,), 0.3)
    # (Time loop will go here...)
    N_path = ref_path.shape[1]
    # Evenly spaced base speeds
    # if is_difficult:
    #     base_speeds = torch.full((B,), 0.2)
    #     print ("Using difficult scenario with fixed low speed of 0.2 m/s for all environments.")
    # else:
    base_speeds = torch.linspace(u_min, u_max, B)

    # Small noise
    noise = 0.02 * torch.randn(B)

    # Final desired speed per env
    u_des_f = torch.clamp(base_speeds + noise, u_min, u_max)
    for k, t in enumerate(T[:-1]):

        phase = torch.remainder(phase + w_des * dt, 2 * torch.pi) # (B,)

        dd_phi = -A_des* (w_des ** 2) * torch.sin(phase)   # (B,)
        holder["dd_phi"] = dd_phi

        state = integrator.step(state, dd_phi) 
        # print("state requires grad:", state.requires_grad)  # (B, state_dim)
        # print("state grad_fn:", state.grad_fn)

        theta_h = state[:, 6]
  # (B,)

        if torch.isnan(theta_h).any():
            bad_idx = torch.isnan(theta_h)

            print("NaN detected at t =", t.item())
            print("kp_a:",Kp_a[bad_idx], "ki_a:", Ki_a[bad_idx], "kd_a:", Kd_a[bad_idx])
            print("kp_s:",Kp_s[bad_idx], "kd_s:", Kd_s[bad_idx])

        


            # if k > 0:
            #     print("Previous State (bad envs):", state_history[k-1, bad_idx])

            # print("Current State (bad envs):", state[bad_idx])
            # print("Time:", t.item())
            final_k = k-1
            break

        d_alpha = torch.zeros_like(alpha)

        uh, vh, xh, yh = compute_head_vel_torch(state,alpha,d_alpha,xh,yh,dt, Const_vals)
        # print("uh grad_fn:", uh.grad_fn, "xh grad_fn:", xh.grad_fn)

        theta_h = state[:, 6]   # (B,)

        state_aug = torch.stack([xh, yh, theta_h], dim=1)  # (B,3)

        # lookahead distance update
 
        
        # pure pursuit to get reference heading
        theta_ref, i0, iL, xL,yL,*__ = pp_obj(state_aug, ref_path, Ld_f)
        theta_h_w= ptg.wrap_to_pi(theta_h)
        # print("theta_h_w grad_fn:", theta_h_w.grad_fn)
        theta_ref_w= ptg.wrap_to_pi(theta_ref)
        e_th = ptg.wrap_to_pi(theta_ref_w - theta_h_w)


        # Desired speed
         # fixed per env

        e_u = u_des_f - uh
      
        e_cte = compute_signed_cteb(xh, yh, ref_path)
        # print("e_cte grad_fn:", e_cte.grad_fn)



        # Build batched NN input
        inp = torch.stack([
            torch.zeros_like(uh),  # placeholder for future use
            theta_h_w / (torch.pi / 2),
            uh / u_max,
            u_des_f / u_max,
            e_th / (torch.pi / 2),
            e_u / u_max,
            e_cte / Ld_f,
            torch.zeros_like(uh),
        ], dim=1)   # (B, features)

        # Forward pass (batched)
        action = policy_net(inp)   # (B, action_dim)
        # print("action req grad", action.requires_grad)
        # --------------------------------------
        # Map NN outputs to gains (batched)
        # --------------------------------------
        r_sat= 0.01 * torch.clamp(torch.abs(action) - 0.9, min=0.0).pow(2).mean()
        Kp_a = to_range_tanh(action[:, 0], -1.0, 0.0)
        Ki_a = to_range_tanh(action[:, 1], -0.1, 0.0)
        Kd_a = to_range_tanh(action[:, 2], -0.1, 0.0)

        # --------------------------------------
        # Throttle scalar s (batched)
        # --------------------------------------
        # s = to_range_tanh(action[3], 0.0, 1.0)
        Kp_s = to_range_tanh(action[:, 3], 0.0, 2.5)
        Kd_s = to_range_tanh(action[:, 4], 0.0, 0.1)
        Ki_s = to_range_tanh(action[:, 5], 0.0, 0.1)

        s_raw, e_u_int, e_u_prev = speed_pid_step(e_u,e_u_int,e_u_prev,dt,Kp_s,Ki_s,Kd_s,s_min=0.0,s_max=1.0)
        # s_raw = speed_pd_step(e_u, e_u_prev, dt, Kp_s, Kd_s)
        # e_u_prev = e_u
        s = to_range_tanh(s_raw, 0.0, 1.0)

        A_des, w_des = throttle_from_s(s)
        # A_des_f = rate_limit(A_des, A_prev, A_rate_max, dt)
        # w_des_f = rate_limit(w_des, w_prev, w_rate_max, dt)
        # A_prev = A_des_f
        # w_prev = w_des_f
           # both (B,)

        # --------------------------------------
        # PID (batched)
        # --------------------------------------
        alpha_des, e_int, e_prev = pid_step_c(e_th,e_int,e_prev,
                dt,Kp_a,Ki_a,Kd_a,alpha_min=-alpha_max,alpha_max=alpha_max)

        # --------------------------------------
        # Rate limit (batched)
        # --------------------------------------
        alpha = rate_limit(alpha_des, alpha_prev, alpha_rate_max, dt)
        alpha_prev = alpha

        holder["alpha"] = alpha

        du = (uh - u_prev) / dt
        u_prev = uh
      
        e_th_n  = e_th / (torch.pi / 2)
        e_u_n   = e_u / u_max
        e_cte_n = e_cte / Ld_f



        rh  = 20.0 * e_th_n**2
        ru  = 20.0 * e_u_n**2
        # ru2 = 100.0 * torch.relu(0.2 - uh)**2
        rK = 0.0001 * (
        Kp_a**2 + Ki_a**2 + Kd_a**2 +
        Kp_s**2 + Kd_s**2
        ).mean()
        r_cte = 20*e_cte_n**2
        r_du = 0.01 * du**2
        dK = torch.stack([Kp_a-Kp_a_prev, Ki_a-Ki_a_prev, Kd_a-Kd_a_prev,
                  Kp_s-Kp_s_prev, Kd_s-Kd_s_prev], dim=1)
        Kp_a_prev = Kp_a
        Ki_a_prev = Ki_a
        Kd_a_prev = Kd_a
        Kp_s_prev = Kp_s
        Kd_s_prev = Kd_s
        r_dK = 0.001 * (dK**2).mean()

        step_loss = rh + ru + r_cte + r_du + rK + r_dK + r_sat   # (B,)

        # print("step_loss req/grad_fn:", step_loss.requires_grad, step_loss.grad_fn)
        # print("rh req/grad_fn:", rh.requires_grad, rh.grad_fn)
        # print("uh req/grad_fn:", uh.requires_grad, uh.grad_fn)
        # print("A_des req/grad_fn:", A_des.requires_grad, A_des.grad_fn)
        # print("w_des req/grad_fn:", w_des.requires_grad, w_des.grad_fn)

        if chunk_loss is None:
            chunk_loss = step_loss
        else:
            chunk_loss = chunk_loss + step_loss

        total_loss = total_loss + step_loss
        # ---------------------------------
        # Loss logging
        # ---------------------------------
        loss_history[k, :, 0] = rh.detach()
        loss_history[k, :, 1] = ru.detach()

        # ---------------------------------
        # State logging
        # ---------------------------------
        state_history[k, :, 0:7] = state[:, :7].detach()
        state_history[k, :, 7]   = dd_phi.detach()
        state_history[k, :, 8]   = alpha.detach()
        state_history[k, :, 9]   = uh.detach()
        state_history[k, :, 10]  = xh.detach()
        state_history[k, :, 11]  = yh.detach()

        # ---------------------------------
        # Desired speed logging
        # ---------------------------------
        u_des_history[k, :, 0] = u_des_f.detach()

        # ---------------------------------
        # Throttle logging
        # ---------------------------------
        throttle_history[k, :, 0] = s.detach()
        throttle_history[k, :, 1] = A_des.detach()
        throttle_history[k, :, 2] = w_des.detach()
        throttle_history[k, :, 3] = Ld_f.detach()

        # ---------------------------------
        # Gain logging
        # ---------------------------------
        gain_history[k, :, 0] = Kp_a.detach()
        gain_history[k, :, 1] = Ki_a.detach()
        gain_history[k, :, 2] = Kd_a.detach()

        # ---------------------------------
        # Error terms
        # ---------------------------------
        loss_term[k, :, 0] = e_th.detach()
        loss_term[k, :, 1] = theta_ref.detach()
        loss_term[k, :, 2] = e_cte.detach()

        # ---------------------------------
        # Head position logging
        # ---------------------------------
        state_aug_history[k, :, 0] = xh.detach()
        state_aug_history[k, :, 1] = yh.detach()

        # ---------------------------------
        # Goal logging
        # ---------------------------------
        goal_history[k, :, 0] = xL.detach()
        goal_history[k, :, 1] = yL.detach()

        # -------------------------------------------------
# TBPTT update
# -------------------------------------------------
        if torch.all(iL >= N_path - 2):
            break
    
        if (k + 1) % H == 0 and optimizer is not None:
            # print("k", k, "dd_phi req/grad_fn:", dd_phi.requires_grad, dd_phi.grad_fn)
            # print("A_des req/grad_fn:", A_des.requires_grad, A_des.grad_fn)
            # print("w_des req/grad_fn:", w_des.requires_grad, w_des.grad_fn)

            loss_scalar = chunk_loss.mean()   # reduce (B,) → scalar
            # print("loss requires grad:",loss_scalar.requires_grad)
            if chunk_loss_history is not None:
                chunk_loss_history.append(loss_scalar.detach().cpu().item())

            if torch.isfinite(loss_scalar):
                optimizer.zero_grad(set_to_none=True)
                loss_scalar.backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            else:
                optimizer.zero_grad(set_to_none=True)
                print("Non-finite loss detected, skipping update at k =", k  )

            # ---------------------------------------------
            # Truncate graph (detach batched states)
            # ---------------------------------------------
            state = state.detach()
            xh    = xh.detach()
            yh    = yh.detach()

            alpha_prev = alpha_prev.detach()
            alpha      = alpha_prev

            e_int  = e_int.detach()
            e_prev = e_prev.detach()
            e_u_int  = e_u_int.detach()

            u_prev  = u_prev.detach()
            u_des_f = u_des_f.detach()
            phase   = phase.detach()
            Ld_f    = Ld_f.detach()
            A_des   = A_des.detach()
            w_des   = w_des.detach()
            A_prev  = A_prev.detach()
            w_prev  = w_prev.detach()
            A_des_f = A_des_f.detach()
            w_des_f = w_des_f.detach()
            e_u_prev = e_u_prev.detach()
            e_cte_n = e_cte_n.detach()
            e_th_n = e_th_n.detach()
            e_u_n = e_u_n.detach()
            Kp_a_prev = Kp_a_prev.detach()
            Ki_a_prev = Ki_a_prev.detach()
            Kd_a_prev = Kd_a_prev.detach()
            Kp_s_prev = Kp_s_prev.detach()
            Kd_s_prev = Kd_s_prev.detach()

            # A_prev = A_prev.detach()
            # w_prev = w_prev.detach()

            holder["alpha"] = alpha_prev

            chunk_loss = None


    # -------------------------------------------------
    # Final leftover chunk
    # -------------------------------------------------
    if (chunk_loss is not None) and (optimizer is not None):
        loss_scalar = chunk_loss.mean()
        loss_scalar.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    final_k = k+1

    return ( state_history[:final_k], T[:final_k], 
                loss_history[:final_k], loss_term[:final_k], total_loss, state_aug_history[:final_k], 
                gain_history[:final_k], goal_history[:final_k], throttle_history[:final_k], u_des_history[:final_k], )



# %% [markdown]
# ### generate ref path

# %%

def make_sine_path(N=600, x_end=4.2, amp=-0.25, cycles=1.0, device="cpu", dtype=torch.float32):
    x = torch.linspace(0.0, x_end, N, device=device, dtype=dtype)
    y = amp * torch.sin(2.0 * torch.pi * cycles * (x / x_end))
    return torch.stack([x, y], dim=1)

def make_circle_path(N=600, radius=2.0, device="cpu"):
    theta = torch.linspace(0.0, 2 * torch.pi, N, device=device)

    x = radius * torch.sin(theta)
    y = radius * (1.0 - torch.cos(theta))  # shifted so it starts at (0,0)

    return torch.stack([x, y], dim=1)

def make_three_quarter_circle_path(N=600, radius=2.0, device="cpu", start_angle=0.0):
    """
    3/4 circle arc (3*pi/2 radians).
    start_angle=0 -> goes from 0 to 3*pi/2.
    Returns (N,2). With the y-shift, it starts at (0,0) when start_angle=0.
    """
    theta = torch.linspace(start_angle, start_angle + 1.5 * torch.pi, N, device=device)
    x = radius * torch.sin(theta)
    y = radius * (1.0 - torch.cos(theta))  # shifted so theta=start_angle starts at (0,0) if start_angle=0
    return torch.stack([x, y], dim=1)

# %% [markdown]
# ### Train 

# %%
def train_alpha_throttle(save_path=None):

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

        # -------------------------------------------------
    # Save path setup (cluster-safe)
    # -------------------------------------------------
    if save_path is None:
        scratch_dir = os.environ.get("SCRATCH", ".")
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")

        run_dir = os.path.join(
            scratch_dir,
            "Sleigh_RL",
            "checkpoints",
            run_name
        )

        os.makedirs(run_dir, exist_ok=True)

        save_path = os.path.join(
            run_dir,
            "sleigh_alpha_throttle_PID_PIDs_old3_100.pt"
        )

        # log_path = os.path.join(run_dir, "training_log_theta_noise1.csv")

    else:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

        # if log_path is None:
        #     log_path = os.path.join(
        #         os.path.dirname(save_path),
        #         "training_log.csv"
        #     )

    print(f"Saving checkpoints to: {save_path}")
 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = Actor().to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=5e-5)

    episodes = 100
    progress = tqdm(range(episodes), desc="Training")

    episode_loss_history = []
    episode_cte_history = []

    ic = torch.zeros(9, device=device)
    lookahead = 0.3
    alpha_max = 1.45

    A = torch.tensor(0.0, device=device)
    w = torch.tensor(2 * math.pi * 2.0, device=device)

    tf = 6.0
    N = 601
    H = 15
    B = 8
    Kp_a = -3
    Ki_a =-0.05
    Kd_a =-0.4

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    # ---------------------------
    # Curriculum
    # ---------------------------
    difficulty = 0
    window = 10
    cte_thresholds = [0.05, 0.03]
    # torch.autograd.set_detect_anomaly(True)
    chunk_loss_history = []
    for ep in progress:

        # ---------------------------
        # Base path (N,2)
        is_circle = False
        if difficulty == 0:
            base_path = make_sine_path(
                N=N, amp=0.5, cycles=0.5, device=device
            )
            is_difficult = False
        elif difficulty == 1:
            if ep % 2 == 0:
                base_path = make_sine_path(
                    N=N, amp=1.0, cycles=1.0, device=device
                )
                is_difficult = False
            else:
                base_path = make_three_quarter_circle_path(
                    N=N, radius=1.0, device=device
                )
                is_circle = True
                is_difficult = False
        else:
            if ep % 2 == 0:
                base_path = make_sine_path(
                    N=N, amp=1.5, cycles=2.0, device=device
                )
                is_difficult = False
            else:
                base_path = make_three_quarter_circle_path(
                    N=N, radius=0.5, device=device
                )
                is_difficult = True
                is_circle = True

        # ---------------------------
        # Create batched perturbed paths
        # ---------------------------
        N_path = base_path.shape[0]

        ref_path_batch = base_path.unsqueeze(0).repeat(B, 1, 1)  # (B,N,2)

        if not is_circle:
        # # Smooth lateral deviation per env
            phase_shift = 2 * torch.pi * torch.rand(B, device=device)
            x_vals = torch.linspace(0, 2*torch.pi, N_path, device=device)

            perturb = 0.02 * torch.sin(
                x_vals.unsqueeze(0) + phase_shift.unsqueeze(1)
            )

            ref_path_batch[:, :, 1] += perturb

        # ---------------------------
        # Structured desired speeds
        # ---------------------------


        # ---------------------------
        # Rollout + TBPTT
        # ---------------------------
        (_,_,_,loss_term,total_loss,state_aug_history,_,_,_,_
        ) = environment_map_alpha_pid(ic,ref_path_batch,tf,N,A,w,Kp_a, Ki_a, Kd_a,
                alpha_max,lookahead,policy,optimizer,H,B=B, chunk_loss_history=chunk_loss_history, is_difficult= is_difficult)
  

        # ---------------------------
        # Logging
        # ---------------------------
        ep_loss = total_loss.mean().item()
        episode_loss_history.append(ep_loss)

        mean_abs_cte = loss_term[:, :, 2].abs().mean().item()
        episode_cte_history.append(mean_abs_cte)

        progress.set_postfix({
            "loss": f"{ep_loss:.4f}",
            "cte": f"{mean_abs_cte:.3f}",
            "diff": difficulty
        })

        # ---------------------------
        # Curriculum update
        # ---------------------------
        if len(episode_cte_history) >= window:

            recent_cte = np.mean(episode_cte_history[-window:])

            if difficulty == 0 and recent_cte < cte_thresholds[0]:
                difficulty = 1
                print("\nUpgrading to difficulty 1")

            elif difficulty == 1 and recent_cte < cte_thresholds[1]:
                difficulty = 2
                print("\nUpgrading to difficulty 2")

        # ---------------------------
        # Save checkpoint
        # ---------------------------
        if ep % 20 == 0:
            torch.save({
                "model_state_dict": policy.state_dict(),
                "episode_loss_history": episode_loss_history,
                "chunk_loss_history": chunk_loss_history,
                "episode_cte_history": episode_cte_history,
                    }, save_path)
    # torch.save(policy.state_dict(), save_path)
    torch.save({
        "model_state_dict": policy.state_dict(),
        "episode_loss_history": episode_loss_history,
        "chunk_loss_history": chunk_loss_history,
        "episode_cte_history": episode_cte_history,
            }, save_path)

    return policy, episode_loss_history, chunk_loss_history, episode_cte_history

if __name__ == "__main__":
    print("Starting training...")
    policy, loss_history, chunk_loss_history, cte_history = train_alpha_throttle()
    print("Training complete.")


