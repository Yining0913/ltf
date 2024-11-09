import torch
import numpy as np
import os
from fairgym.envs.default_reward import *
import gymnasium
from jax import numpy as jnp
import gym
from fairgym.envs.replicator.qualification_replicator_env import (
    GroupQualificationReplicatorEnv,
)
import matplotlib.pyplot as plt
from EnvWrapper import CustomEnv


def plott(obs):
    s0 = torch.Tensor(obs)
    plt.plot(obs[0, :, 0], "red")
    plt.plot(obs[1, :, 0], "blue")
    plt.plot(obs[0, :, 1], "green")
    plt.plot(obs[1, :, 1], "purple")
    plt.show()


os.environ["MUJOCO_GL"] = "egl"
env = CustomEnv(
    done=True,
    episode_len=50,
    lbd=0.0,
    loss=zero_one_loss,  # useless
    disparity=qualification_rate_disparity,  # useless
)

accuracy_term = zero_one_accuracy
disparity_term = qualification_rate_disparity

num_samples = 100000
state, info = env.reset(return_info=True)
num_groups = env.num_groups
s, a, r, g = [], [], [], []
for i in range(num_samples):
    print(i)
    s.append(state)
    action = np.random.rand(num_groups)
    a.append(action.tolist())
    state, reward, done, info = env.step(action)
    r.append(
        float(
            accuracy_term(
                info["prev_state"],
                info["prev_action"],
                info["prev_results"],
                info["current_state"],
            )
        )
    )
    g.append(
        float(
            1
            - disparity_term(
                info["prev_state"],
                info["prev_action"],
                info["prev_results"],
                info["current_state"],
            )
        )
    )
    if done:
        print(i, done)
        state, info = env.reset(return_info=True)

s = torch.Tensor(s)
a = torch.Tensor(a)
r = torch.Tensor(r)
g = torch.Tensor(g)

indices = torch.randperm(s.shape[0])
s = s[indices]
a = a[indices]
r = r[indices]
g = g[indices]

train_data = {"s": s, "a": a, "r": r, "g": g}


torch.save(train_data, "./data/train_phi.pt")
