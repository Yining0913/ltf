import numpy as np
import os
import sys
import gymnasium
import gym
sys.modules["gym"] = gymnasium
from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from functools import partial
from fairgym.envs.default_reward import *
from jax import numpy as jnp
from DDPGExperiment import DDPGExperiment
from DDPGphase import *
from utilities import *
import time
from fairgym.envs.replicator.qualification_replicator_env import (
    GroupQualificationReplicatorEnv,
)
from fairgym.plotting.phase import make_replicator_phase_plot
# import gym
from Agents import DDPGAgent
from EnvWrapper import CustomEnv, RelaxationEnv

import argparse
os.environ["MUJOCO_GL"] = "egl"

def to_plot(env, agent, info):

    state = info["current_state"]
    results = info["prev_results"]

    out = {
        "Qualification Rate (g0)": (state.pr_Y1[0], "#13a5cd"),
        "Qualification Rate (g1)": (state.pr_Y1[1], "#ff24da"),
    }

    # very first render does not have past action
    if results is not None:
        # out = out | {
        #     "Accept Rate (g0)": (results.accept_rate[0], "green"),
        #     "Accept Rate (g1)": (results.accept_rate[1], "red"),
        #     # 'Disparity': (group_disparity(info), ''),
        #     # 'Utility': (classifier_utility(info), ''),
        #     "Reward": (info["reward"], "black"),
        # }
        out["Accept Rate (g0)"] = (results.accept_rate[0], "green")
        out["Accept Rate (g1)"] = (results.accept_rate[1], "red")
#         out["Reward"] = (info["reward"], "black")
#         out["Accuracy (g0)"]: (1 - (state.tp_frac[0] + state.tn_frac[0]), "blue")
#         out["Accuracy (g1)"]: (1 - (state.tp_frac[1] + state.tn_frac[1]), "violet")

    return out

def main():
    
    parser = argparse.ArgumentParser(description="")
   
    parser.add_argument(
        "--loss",
        type=str,
        default='loss_tp_tn',
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
    
    str_to_loss = {
        "loss_tp_frac" : loss_tp_frac,
        "zero_one_loss" : zero_one_loss,
        "loss_tp_tn": loss_tp_tn,
        "equal_opportunity_disparity": equal_opportunity_disparity,
        "equalized_odds_disparity": equalized_odds_disparity,
        "euclidean_demographic_partity": euclidean_demographic_partity,
        "none": euclidean_demographic_partity
    }

    lbd = 0.5
    res = 100
    num_runs = 20
    seed = 0
    dataset = "adult"
    episode_len = 150
    
    str_to_loss = {
    "loss_tp_frac" : str_to_loss[loss_term],
    "zero_one_loss" : str_to_loss[disparity_term],
    "loss_tp_tn": loss_tp_tn,
    "equal_opportunity_disparity": equal_opportunity_disparity,
    "equalized_odds_disparity": equalized_odds_disparity,
    "euclidean_demographic_partity": euclidean_demographic_partity,
    "none": euclidean_demographic_partity
    }
    name = dataset + "_" + loss_term + "_" + disparity_term
    
    os.environ["MUJOCO_GL"] = "egl"
    env = RelaxationEnv(
        done=True,
        episode_len=episode_len,
        lbd=lbd,
        loss=str_to_loss[loss_term],
        disparity=str_to_loss[disparity_term],
        save=False
    )
    env.observation_space = gym.spaces.Box(0, 1, (130,), dtype=np.float32)
    env.action_space = gym.spaces.Box(0.0, 1.0, (2,), dtype=np.float32)
    print(type(env.observation_space))
    

    model = TD3.load(name)

    agent = DDPGAgent(model)

    experiment = DDPGExperiment(agent, env)
    experiment.reset(
        return_info=True
    )
    
    observation, reward, terminated, info = experiment.record(
        name, n_steps=episode_len, to_plot=to_plot, render_flag=True
    )
#     print(reward)

#     make_replicator_phase_plot(agent, env, 0, name+".pdf", res=16)
    make_replicator_phase_plot(
            agent, env, seed=seed, res=res, filename=dataset+ "_" +"drl_"+loss_term+"+"+disparity_term+str(seed)+".eps", 
            options={"pr_G": [0.5, 0.5]}, num_runs=num_runs, color=disparity_term, return_truncated=False
        )


# from phase import make_replicator_phase_plot
# make_replicator_phase_plot(agent, envs, seed=0)


if __name__ == "__main__":
    main()
