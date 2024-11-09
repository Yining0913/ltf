#!/usr/bin/env python

from fairgym.plotting.phase import make_replicator_phase_plot
from fairgym.envs.default_reward import *
from fairgym.envs.base_env import BaseEnv
from functools import partial

from utilities import *
import gymnasium as gym
import argparse
from fairgym.agent import GreedyAgent
from fairgym.experiment import Experiment

# Use to view when/how many times Jax traces and compiles functions
# import jax
# jax.config.update("jax_log_compiles", True)


def to_plot(env, agent, info):

    state = info["current_state"]
    results = info["prev_results"]

    out = {
        "Qualification Rate (g0)": (state.pr_Y1[0], "#13a5cd"),
        "Qualification Rate (g1)": (state.pr_Y1[1], "#ff24da"),
    }

    # very first render does not have past action
    if results is not None:
        out = out | {
            "Accept Rate (g0)": (results.accept_rate[0], "green"),
            "Accept Rate (g1)": (results.accept_rate[1], "red"),
            # 'Disparity': (group_disparity(info), ''),
            # 'Utility': (classifier_utility(info), ''),
            "Reward": (info["reward"], "black"),
        }

    return out


################################################################################
# //  ____                            _
# // |  _ \ _____      ____ _ _ __ __| |
# // | |_) / _ \ \ /\ / / _` | '__/ _` |
# // |  _ <  __/\ V  V / (_| | | | (_| |
# // |_| \_\___| \_/\_/ \__,_|_|  \__,_|
# //


def _reward_fn(
    lambda_,
    loss_term,
    disparity_term,
    prev_state,
    prev_action,
    prev_results,
    current_state,
):
    """
    What the environment's reward signal is, as a function
    of the action we just took

    TODO Even though this gets jitted in env.step, we could jit it here to hold the
     (changing on reset) args as static.
    """
    loss = loss_term(prev_state, prev_action, prev_results, current_state)
    disparity = disparity_term(prev_state, prev_action, prev_results, current_state)

    return 1 - (loss * (1 - lambda_) + lambda_ * disparity)



################################################################################
# //  ____
# // |  _ \ _   _ _ __
# // | |_) | | | | '_ \
# // |  _ <| |_| | | | |
# // |_| \_\\__,_|_| |_|
# //
# Run simulation


def main():

    # WARN: NOT THE SAME AS DP-FAIR CLASSIFIER EXPLORED IN LAST PAPER
    # (this one is regularized fairness; last one was constrained fairness)
    # WARN: IF UTILITY IS NONCONVEX, WE DON'T EXPECT PHASE PORTRAIT AND INDIVIDUAL
    # EXPERIMENTS TO AGREE!

    # with env and agent jit
    # env: BaseEnv = gym.make("GroupQualificationReplicator-v0", reward_fn=reward_fn)
    # agent = GreedyAgent(env, use_jit=True)
    # experiment = Experiment(agent, env)
    # experiment.reset(seed=1, options={"pr_q": [0.2, 0.8], "pr_G": [0.5, 0.5]})

    # observation, reward, terminated, truncated, info = experiment.record(
    #     "baseline", n_steps=100, to_plot=to_plot, render_flag=True
    # )

    # Without env jit
    
    
    
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
    res = 32
    num_runs = 5
    
    str_to_loss = {
    "loss_tp_frac" : loss_tp_frac,
    "zero_one_loss" : zero_one_loss,
    "loss_tp_tn": loss_tp_tn,
    "equal_opportunity_disparity": equal_opportunity_disparity,
    "equalized_odds_disparity": equalized_odds_disparity,
    "euclidean_demographic_partity": euclidean_demographic_partity,
    "none": euclidean_demographic_partity
    }
    
    if disparity_term == "none":
        lbd = 0

    reward_fn = partial(
        _reward_fn,
        lbd,
        str_to_loss[loss_term],
        str_to_loss[disparity_term],
    )
    
    env: BaseEnv = gym.make(
        "AdultQualificationReplicator-v0", reward_fn=reward_fn, use_jit=True
    )
    agent = GreedyAgent(env, use_jit=True)
    experiment = Experiment(agent, env)
    experiment.reset(seed=1, options={
        "pr_G": [0.5, 0.5]}
    )

#     observation, reward, terminated, truncated, info = experiment.record(
#         "baseline", n_steps=100, to_plot=to_plot, render_flag=True
#     )
    for seed in [0]:
        make_replicator_phase_plot(
            agent, env, seed=seed, res=res, filename="adult_myopic_"+loss_term+"+"+disparity_term+str(seed)+".eps", 
            options={"pr_G": [0.5, 0.5]}, num_runs=num_runs, color=disparity_term, 
        )


if __name__ == "__main__":
    main()
