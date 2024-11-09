import os

import numpy as np
import sys
import gymnasium
from utilities import *
# redirect all gym imports to gymnasium
sys.modules["gym"] = gymnasium
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)

from functools import partial
# from fairgym.envs.default_reward import *
from fairgym.envs.replicator.reward import *
from jax import numpy as jnp
# from fairgym.envs.replicator.qualification_replicator_env import (
#     GroupQualificationReplicatorEnv,
# )
from fairgym.envs.replicator.qualification_replicator_env import (
    QualificationReplicatorEnv,
)
from EnvWrapper import CustomEnv

os.environ["MUJOCO_GL"] = "egl"
env = CustomEnv(
    done=True,
    episode_len=50,
    lbd=0.0,
    loss=zero_one_loss,
    disparity=qualification_rate_disparity,
)

# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions), sigma=0.2 * np.ones(n_actions)
)
# model = DDPG.load("GroupQualificationReplicator_part1")
# env = model.get_env()
model = DDPG(
    "MlpPolicy",
    env,
    action_noise=action_noise,
    verbose=1,
    learning_rate=0.001,
    batch_size=128,
    learning_starts=1000,
)
model.learn(total_timesteps=200000, log_interval=1)
model.save("GroupQualificationReplicator_loss_tp_tn+0.3qr")
# model.load("GroupQualificationReplicator.zip")
