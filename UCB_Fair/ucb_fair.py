import gym
import gym_fair_thresholds
import math
from time import time
from itertools import product
from state_action_mapping.state_action_mapping import *
import matplotlib.pyplot as plt
import gc
import numpy as np
import torch
import torch.nn.functional as F
import os

from jax import numpy as jnp
from util.util import jax_to_torch as jtt

def classifier_utility(prev_state, prev_action, prev_results, current_state):
    """
    Measurement of inherent classifier utility
    """

    # true positive fraction of each group after action taken
    # Pr(Y_hat = 1, Y = 1 | G)
    tp_a = prev_results.tp_a

    # true negative fraction of each group after action taken
    # Pr(Y_hat = 0, Y = 0 | G)
    tn_a = prev_results.tn_a

    # accuracy for each group after action taken
    # Pr(Y_hat = Y | G)
    acc_a = prev_results.tp_a + prev_results.tn_a

    # total accuracy after action taken
    # Pr(Y_hat = Y)
    return jnp.sum(prev_state.pr_g * acc_a)


def group_disparity(prev_state, prev_action, prev_results, current_state):
    """
    Measurement of fairness violation
    """

    # # qualification rate of induced state
    # v = current_state.pr_q.reshape(info['num_groups'], 1)

    # acceptance rates after action taken
    v = prev_results.ar_a.reshape(prev_state.pr_x.shape[0], 1)

    # violation of demographic parity after action taken
    return jnp.sum((v - jnp.transpose(v)) ** 2) / 2

env = gym.make('gym_fair_thresholds/FairThresholds-Replicator-v0')

class UCB_Fair:
    def __init__(self, parameters):
        self.device = parameters['device']

        self.eps = parameters['eps']
        self.alpha = parameters['alp']
        self.beta = parameters['beta']
        self.eta = parameters['eta']

        self.xi = parameters['xi']



        self.range_low = parameters['range_low']
        self.range_high = parameters['range_high']
        self.num_groups = parameters['num_groups']
        self.H = parameters['H']
        self.K = parameters['K']
        self.d = parameters['d']
        self.b = self.H - parameters['b']
        self.Y = [0]
        self.s_encoder = StateEmbedding()
        self.phi = StateActionMapping(10, 2, 12)
        self.phi.load_state_dict(torch.load('./state_action_mapping/data/phi_model.pt', map_location=self.device))

        self.eta = self.xi / ((self.K * self.H * self.H) ** 0.5)

        # K*H*d
        self.wr_graph = torch.zeros(self.K, self.H, self.d, device=self.device)
        self.wg_graph = torch.zeros_like(self.wr_graph)

        # #K*H*d*d
        # self.lbd_graph = torch.zeros(self.K, self.H, self.d, self.d, device=self.device)
        # self.lbd_graph[0, :, :, :] = torch.eye(self.d, device=self.device).unsqueeze(0).repeat(self.H, 1, 1) #init K = 0

        # H*d*d
        self.phi_phi_T = torch.zeros(self.H, self.d, self.d, device=self.device)

        # H*d
        self.phi_r_v = torch.zeros(self.H, self.d, device=self.device)
        self.phi_g_v = torch.zeros_like(self.phi_r_v)

        # K*H
        # self.s_graph = torch.zeros_like(self.K, self.H, device=self.device)
        # self.a_graph = torch.zeros_like(self.s_graph)

        # K*H
        self.r_graph = torch.zeros(self.K, self.H, device=self.device)
        self.g_graph = torch.zeros_like(self.r_graph)

        # K*(H+1)
        self.vr_graph = torch.zeros(self.K, self.H + 1, device=self.device)
        self.vg_graph = torch.zeros_like(self.vr_graph)

        t = str(int(time()))
        self.out_path = './result/exp'+'k'+str(self.K)+'h'+str(self.H)+'beta'+str(self.beta)+'t'+t
        os.mkdir(self.out_path)


    def split_region(self):
        self.num_per_dimension = math.ceil((self.range_high - self.range_low) / self.eps / 2)
        self.actual_eps = (self.range_high - self.range_low) / self.num_per_dimension / 2
        self.centers_per_dimension = list(np.linspace(self.range_low - self.actual_eps, self.range_high + self.actual_eps, self.num_per_dimension + 2)[1:-1])
        self.m = self.num_per_dimension ** self.num_groups
        self.volume = (self.actual_eps * 2) ** self.num_groups
        loopall = [self.centers_per_dimension] * self.num_groups
        self.centers = torch.Tensor(tuple(product(*loopall)), device=self.device)
        # self.alpha = math.log(self.m) * self.K / 2 / (1 + self.xi + self.H)

    def train(self):
        self.split_region()
        infos = [0] * self.H
        s_a_list = [0] * self.H
        for k in range(self.K):
            state, info = env.reset(seed=k, return_info=True)
            for h in range(self.H):
                action = self.draw_action(state, h, k)
                s_a_list[h] = (state, action)
                state, reward, terminated, truncated, info = env.step(np.array(action))

                prev_state = info['prev_state']
                prev_action = info['prev_action']
                prev_results = info['prev_results']
                current_state = info['current_state']
                r = classifier_utility(prev_state, prev_action, prev_results, current_state)
                g = 1 - group_disparity(prev_state, prev_action, prev_results, current_state)
                info['r'] = r
                info['g'] = 1 - g
                infos[h] = (state, info)
                self.r_graph[k, h] = jtt(r)
                self.g_graph[k, h] = jtt(g)
            torch.save(infos, self.out_path+"/"+"k"+str(k)+".pt")
            for h in range(self.H-1, -1, -1):
                # s, a = self.s_a_list[k][h]
                s, a = s_a_list[h]
                phi = self.phi(jtt(s).unsqueeze(0).to(self.device, dtype=torch.float), a.unsqueeze(0), self.s_encoder)[0].permute(1, 0)
                self.phi_phi_T[h] += phi @ phi.T
                self.phi_r_v[h] += (phi * (self.r_graph[k, h] + self.vr_graph[k, h + 1])).squeeze()
                self.phi_g_v[h] += (phi * (self.g_graph[k, h] + self.vg_graph[k, h + 1])).squeeze()
            #
            # del infos
            # del s_a_list
            # gc.collect()

            # self.s_a_list = []
            #update Y



            self.Y.append(max(min((self.Y[-1] + self.eta * (self.b - self.vg_graph[k, 0])).item(), self.xi), 0))
            if k % 10 == 0:
                print(self.b - self.vg_graph[k, 0])
                print(k, self.Y[-1], self.r_graph[k, -3:], self.g_graph[k, -3:])


    def draw_action(self, state, h, k):
        Lambda = self.phi_phi_T[h] + torch.eye(self.d, device=self.device)
        Lambda_inverse = torch.inverse(Lambda)
        self.wr_graph[k, h] = torch.mv(Lambda_inverse, self.phi_r_v[h])
        self.wg_graph[k, h] = torch.mv(Lambda_inverse, self.phi_g_v[h])
        # self.compute_Q(state, k,h)
        num = self.centers.shape[0]
        phis = self.phi(jtt(state).unsqueeze(0).to(self.device, dtype=torch.float).repeat(num, 1,1,1), self.centers, self.s_encoder)[0] #self.m * d
        temp = phis @ Lambda_inverse
        temp = torch.bmm(temp.view(self.m, 1, self.d), phis.view(self.m, self.d, 1)).flatten() ** 0.5
        Qr = torch.clamp(phis @ self.wr_graph[k, h] + self.beta * temp, min=0, max=self.H)
        Qg = torch.clamp(phis @ self.wg_graph[k, h] + self.beta * temp, min=0, max=self.H)
        Qall_alpha = self.alpha * (Qr + self.Y[-1] * Qg)

        probs = F.softmax(Qall_alpha)
        # print(probs[:5])
        torch.manual_seed(seed=k)
        center = int(torch.distributions.categorical.Categorical(probs=probs).sample())
        action = torch.rand(self.num_groups) * self.actual_eps * 2 - self.actual_eps
        action += self.centers[center]

        #update V
        self.vr_graph[k, h] = torch.dot(Qr, probs)
        self.vg_graph[k, h] = torch.dot(Qg, probs)


        return action




parameters = {'device': 'cpu',
              'eps': 0.1,
              'alp': 1,
              'beta': 0.05,
              'eta': 0.01,
              'xi': 0,
              'b': 1.5,
              'range_low': 0,
              'range_high': 1,
              'num_groups': 2,
              'H': 15,
              'K': 500,
              'd': 10
              }

ucbfair = UCB_Fair(parameters)
ucbfair.train()

result = {
    # "s_a": ucbfair.s_a_list,
    # "vr": ucbfair.vr_graph,
    # "vg": ucbfair.vg_graph,
    # "sav": ucbfair.sav_list,
    "parameters": parameters,
    "r": ucbfair.r_graph,
    "g": ucbfair.g_graph,
}

torch.save(result, ucbfair.out_path+"/result.pt")
# torch.save(ucbfair.vr_graph, 'vr.pt')
# torch.save(ucbfair.vg_graph, 'vg.pt')
# torch.save(ucbfair.r_graph, 'r.pt')
# torch.save(ucbfair.g_graph, 'g.pt')
# pass
# ucbfair.split_region()
