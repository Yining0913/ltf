#!/usr/bin/env python3

import types
import torch

import numpy as np

import gym

from util.agent import GreedyAgent
from experiment_UCBFair import Experiment

def to_plot(info):
    info_ns = types.SimpleNamespace(**info)
    assert info['type'] == 'post_update'

    sv, sav = info['sv'], info['sav']

    out = {
        'Qualification Rate (g0)': (sv.q_g[0], '#13a5cd'),
        'Qualification Rate (g1)': (sv.q_g[1], '#ff24da')
    }

    # very first render does not have past action
    if sav is not None:
        out = out | {
            'Accept Rate (g0)': (sav.ar_g[0], 'green'),
            'Accept Rate (g1)': (sav.ar_g[1], 'red'),
            # 'Disparity': (group_disparity(info), ''),
            # 'Utility': (classifier_utility(info), ''),
            'Reward': (info['r'], 'black'),
            'Disparity': (info['g'], 'grey'),
        }

    return out


env = gym.make('gym_fair_thresholds/FairThresholds-Replicator-v0')
# redefine_env_reward_function_from_post_update_info(env, lamdba=0.8)
#
# agent = GreedyAgent(env)
#
#
# folder_name = "expk500h15t1668396543"
# folder_name = "expk500h15t1668397511"
# folder_name = "expk500h15beta0.01t1668398989"
# folder_name = "expk500h15beta0.01t1668402591"
# folder_name = "expk500h15beta0.05t1668450597"
# folder_name = "expk500h15beta0.05t1668450773"
folder_name = "expk500h15beta0.05t1668451072"
k_list = [0, 200, 496, 497, 498, 499]

for k in k_list:
    path = './result/' + folder_name

    infos = torch.load(path+"/k"+str(k)+'.pt')


    experiment = Experiment(None, env)
    experiment.reset(seed=0, options={
        'q_g': np.array([0.2, 0.8])
    })
    experiment.record(infos, to_plot=to_plot, name=folder_name+'_'+str(k))

    # from phase import make_replicator_phase_plot
    # make_replicator_phase_plot(agent, env, seed=0)
