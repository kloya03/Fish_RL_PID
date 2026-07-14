from actor_numpy import ActorNumpy
from obs_norm_numpy import ObsNormalizer
import numpy as np

actor = ActorNumpy("actor_weights.npz")
norm  = ObsNormalizer("obs_norm.npz")
#obs order: [ux, uy, qh, qh-0.75, ux-0.2, delta_prev, alpha_prev]
# obs = jnp.asarray(

#             [
#                 ux,
#                 uy,
#                 qh,
#                 e_thetah  qh - 0.75,
#                 vel_error  ux-0.2,
#                 deltas[k-1] if k > 0 else 0.0,
#                 alpha_list[k-1] if k > 0 else 0.0



#             ],
#             dtype=jnp.float32)

while True:
    #obs = get_robot_obs()   # your sensors
    obs = norm.normalize(obs)
    action = actor.act(obs)
    action = np.tanh(action)  # if your action space is bounded, e.g. [-1,1]
    alpha_change = action[0]  # e.g. steering
    delta_change = action[1]  # e.g. fin angle

    delta_change = 3*dt * delta_change
    prev_delta = deltas[k-1] if k > 0 else 0.0
    delta = prev_delta + delta_change  # Assuming actions are already rate-limited by the policy
    delta = np.clip(delta, -1,1)
    alpha = alpha_change * 87000 * dt
    prev_alpha = alpha_list[k-1] if k > 0 else 0.0
    alpha = prev_alpha + alpha
    alpha = np.clip(alpha, -4619, 4619)
    obs = norm.normalize(obs)
    action = actor.act(obs)
    action = np.tanh(action)
    alpha = action[0]
    servo_angle = action[1]  


    send_to_robot(action)

# import numpy as np
# import time
# from actor_numpy import ActorNumpy
# from checkpoint_loader import load_actor

# obs_dim = 7
# act_dim = 2

# # load once
# jax_actor,_ = load_actor("final", obs_dim, act_dim)
# np_actor = ActorNumpy("actor_weights.npz")

# obs = np.random.randn(obs_dim)

# # ---------- correctness check ----------
# mean_jax,_ = jax_actor(obs)
# mean_np,_  = np_actor.forward(obs)
# print("diff:", np.max(np.abs(mean_jax-mean_np)))

# # ---------- inference timing ----------
# N = 20000   # number of forward passes

# t0 = time.time()
# for _ in range(N):
#     np_actor.forward(obs)
# t1 = time.time()

# avg = (t1 - t0)/N * 1000  # ms
# print(f"Numpy actor avg inference: {avg:.6f} ms")
# print(f"Frequency: {1/(avg/1000):.1f} Hz")

