import os
from stable_baselines3.common.monitor import Monitor
import numpy as np
import sys
import gymnasium
import gym
from utilities import *
# redirect all gym imports to gymnasium
sys.modules["gym"] = gymnasium
from stable_baselines3 import TD3
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)

from functools import partial
from fairgym.envs.default_reward import (
    qualification_rate_loss,
    qualification_rate_disparity,
    euclidean_demographic_partity,
    zero_one_loss,
)
from jax import numpy as jnp
from fairgym.envs.replicator.qualification_replicator_env import (
    GroupQualificationReplicatorEnv,
)
from EnvWrapper import CustomEnv, RelaxationEnv
from callback import Callback
import argparse


parser = argparse.ArgumentParser(description="")
   
parser.add_argument(
    "--loss",
    type=str,
    default='zero_one_loss',
)

parser.add_argument(
    "--disparity",
    type=str,
    default='equal_opportunity_disparity',
)

args = parser.parse_args()



loss_term = args.loss
disparity_term = args.disparity

print(loss_term)
print(disparity_term)

lbd = 1

str_to_loss = {
    "loss_tp_frac" : loss_tp_frac,
    "zero_one_loss" : zero_one_loss,
    "loss_tp_tn": loss_tp_tn,
    "equal_opportunity_disparity": equal_opportunity_disparity,
    "equalized_odds_disparity": equalized_odds_disparity,
    "euclidean_demographic_partity": euclidean_demographic_partity,
    "none": euclidean_demographic_partity
}


log_dir = "/home/tyin/dreamer_011/DRL/"
lbd = 1
timesteps = 200000
dataset = "adult"
name = dataset + "_" +loss_term + "_" + disparity_term
os.makedirs(log_dir, exist_ok=True)


os.environ["MUJOCO_GL"] = "egl"
env = RelaxationEnv(
    done=True,
    episode_len=150,
    lbd=lbd,
    loss=str_to_loss[loss_term],
    disparity=str_to_loss[disparity_term],
    save=True,
    dataset=dataset
)
env.observation_space = gym.spaces.Box(0, 1, (130,), dtype=np.float32)
env.action_space = gym.spaces.Box(0.0, 1.0, (2,), dtype=np.float32)
print(type(env.observation_space))
env = Monitor(env, log_dir)
# The noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(
    mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
)
model = TD3(
    "MlpPolicy",
    env,
    action_noise=action_noise,
    verbose=1,
    learning_rate=0.0005,
    batch_size=128,
    learning_starts=300,
)
callback = Callback(check_freq=3, log_dir=log_dir, name=name)
model.learn(total_timesteps=timesteps, log_interval=1,  callback=callback)
# model.save("replicator_loss_tp_frac_euclidean_demographic_partity")
# model.load("GroupQualificationReplicator.zip")
