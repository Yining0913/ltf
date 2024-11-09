import os

import numpy as np
import sys
import gymnasium
from utilities import *
# redirect all gym imports to gymnasium
sys.modules["gym"] = gymnasium
from stable_baselines3 import TD3
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)

from functools import partial
from fairgym.envs.replicator.reward import (
    qualification_rate_disparity,
    euclidean_demographic_partity,
    zero_one_loss,
)
from jax import numpy as jnp
# from fairgym.envs.replicator.qualification_replicator_env import (
#     GroupQualificationReplicatorEnv,
# )
from fairgym.envs.replicator.qualification_replicator_env import (
    QualificationReplicatorEnv
)
from EnvWrapper_adult import CustomEnv

os.environ["MUJOCO_GL"] = "egl"
env = CustomEnv(
    done=True,
    episode_len=100,
    lbd=0.0,
    loss=zero_one_loss,
    disparity=euclidean_demographic_partity,
)

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
# action_noise = NormalActionNoise(
#     mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
# )
# model = DDPG.load("GroupQualificationReplicator_part1")
# env = model.get_env()
model = TD3(
    "MlpPolicy",
    env,
    action_noise=None,
    verbose=1,
    learning_rate=0.0005,
    batch_size=128,
    learning_starts=100,
)
model.learn(total_timesteps=500000, log_interval=1)
model.save("adult_01")
# model.load("GroupQualificationReplicator.zip")
