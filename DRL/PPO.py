import os

import numpy as np
import sys
import gymnasium

# redirect all gym imports to gymnasium
sys.modules["gym"] = gymnasium
from stable_baselines3 import PPO
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)

from functools import partial
from fairgym.envs.default_reward import *
from jax import numpy as jnp
from fairgym.envs.replicator.qualification_replicator_env import (
    GroupQualificationReplicatorEnv,
)
from EnvWrapper import CustomEnv
from utilities import *

os.environ["MUJOCO_GL"] = "egl"
env = CustomEnv(
    done=True,
    episode_len=100,
    lbd=0.0,
    loss=loss_tp_tn,
    disparity=qualification_rate_disparity,
)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.001,
    batch_size=128,
)
model.learn(total_timesteps=200000, log_interval=1)
model.save("PPO_replicator_loss_tp_tn+qualification_rate_disparity")

