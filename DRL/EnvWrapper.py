import os
import math
import time
import numpy as np
import matplotlib.pyplot as plt
from functools import partial

import gymnasium
import gym
import pickle
from fairgym.envs.replicator.qualification_replicator_env import (
    GroupQualificationReplicatorEnv,
)
from fairgym.envs.default_reward import (
    qualification_rate_loss,
    qualification_rate_disparity,
    euclidean_demographic_partity,
    zero_one_loss,
)

os.environ["MUJOCO_GL"] = "egl"


def reward_fn_(
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

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        done=False,
        episode_len=100,
        lbd=0.9,
        loss=zero_one_loss,
        disparity=euclidean_demographic_partity,
    ):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:

        reward_fn = partial(
            reward_fn_,
            lbd,
            loss,
            disparity,
        )

        self._env: QualificationReplicatorEnv = gymnasium.make(
            "GroupQualificationReplicator-v0",
            reward_fn=reward_fn,
            return_truncated=True,
        )
        self.action_space = gym.spaces.Box(0.0, 1.0, (2,), dtype=np.float32)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = gym.spaces.Box(0, 1, (130,), dtype=np.float32)
        self.num_groups = 2
        self.feature_bins = 32
        self.done = done
        self.episode_len = episode_len
        self.seed = 0

    def step(self, action):
        observation, reward, done, info = self._env.step(action)
        self.seed += 1
        self.step_count += 1
        observation = self.preprocess_obs(observation, info)
#         print(info["current_state"].pr_G)
        if self.done:
            if self.step_count >= self.episode_len:
                done = True
        #             if reward <= 0.6:
        #                 done = True

        return observation, reward, done, info

    def reset(self, seed=None, return_info=False, options=None):
        # t = int(str(time.time()).split(".")[1])
        # np.random.seed(seed=self.seed)
        # pr_q0 = np.random.rand()
        # print(pr_q0)
        # option = {"pr_q": np.array([0.5, 0.5])}
        if options is None:
            options = {}
        options["pr_G"] = np.array([0.5, 0.5])
        if seed is not None:
            self.seed = seed
        else:
            self.seed += 1
        observation, info = self._env.reset(
            return_info=True,
            seed=self.seed,
            options=options
        )
#         print(self.seed)
        observation = self.preprocess_obs(observation, info)
        # plt.plot(observation)
        self.step_count = 0
        if return_info:
            return observation, info
        else:
            return observation

    def render(self, mode="human"):
        self._env.render()

    def close(self):
        self._env.close()

    def preprocess_obs(self, obs, info):
        #         obs = torch.from_numpy(obs).to(dtype=torch.float32) / 255.0 - 0.5
        pr_G = info["current_state"].pr_G
        # pr_G = pr_G[:, np.newaxis]
        #         print(pr_G)
        pr_G = pr_G.reshape(2,1)
        pr_X = info["current_state"].pr_X * 10
        pr_Y1gX = info["current_state"].pr_Y1gX
        observation = np.concatenate((pr_G, pr_X, pr_Y1gX), axis=1)
        observation = observation.flatten()

        return observation
    
class RelaxationEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        done=False,
        episode_len=100,
        lbd=0.9,
        loss=zero_one_loss,
        disparity=euclidean_demographic_partity,
        save=True,
        dataset="group"
    ):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:

        reward_fn = partial(
            reward_fn_,
            lbd,
            loss,
            disparity,
        )
        self.loss_term = loss
        self.disparity_term = disparity
        if dataset == "group":
            self._env: QualificationReplicatorEnv = gymnasium.make(
                "GroupQualificationReplicator-v0",
                reward_fn=reward_fn,
                return_truncated=True,
            )
        elif dataset == "adult":
            self._env: QualificationReplicatorEnv = gymnasium.make(
                "AdultQualificationReplicator-v0",
                reward_fn=reward_fn,
                return_truncated=True,
            )
        else:
            raise Exception("error")
        self.action_space = gym.spaces.box.Box(0.0, 1.0, (2,), dtype=np.float32)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = gym.spaces.box.Box(0, 1, (130,), dtype=np.float32)
        self.num_groups = 2
        self.feature_bins = 32
        self.done = done
        self.episode_len = episode_len
        self.seed = 0
        self.lambda_final = lbd
        self.lambda_step = lbd / episode_len
        self.lambda_ = 0
        self.save = save
        self.accumulated_r = 0
        self.accumulated_g = 0
        self.accumulated_combo = 0
        self.filename = str(int(time.time())) + ".pickle"
        
    def compute_reward(self, info):
        prev_state = info["prev_state"]
        prev_action = info["prev_action"]
        prev_results = info["prev_results"]
        current_state = info["current_state"]
        loss = self.loss_term(prev_state, prev_action, prev_results, current_state)
        disparity = self.disparity_term(prev_state, prev_action, prev_results, current_state)
        disparity_processed = 1 - (1 - disparity) ** 20
        self.lambda_ += self.lambda_step
#         print(self.lambda_)
        return 1 - (loss * (1 - self.lambda_) + self.lambda_ * disparity_processed), 1 - loss, 1 - disparity
        
    def step(self, action):
        observation, reward, done, info = self._env.step(action)
        reward, r, g = self.compute_reward(info)
        self.accumulated_r += r
        self.accumulated_g += g
        self.accumulated_combo += reward
        self.seed += 1
        self.step_count += 1
        observation = self.preprocess_obs(observation, info)
        if self.done:
            if self.step_count >= self.episode_len:
                done = True
#         print(self.step_count, reward)
        return observation, reward, done, info

    def reset(self, seed=None, return_info=False, options=None):
        # t = int(str(time.time()).split(".")[1])
        # np.random.seed(seed=self.seed)
        # pr_q0 = np.random.rand()
        # print(pr_q0)
        # option = {"pr_q": np.array([0.5, 0.5])}
        if options is None:
            options = {}
        options["pr_G"] = np.array([0.5, 0.5])
        if seed is not None:
            self.seed = seed
        else:
            self.seed += 1
        observation, info = self._env.reset(
            return_info=True,
            seed=self.seed,
            options=options
        )
#         print(self.seed)
        observation = self.preprocess_obs(observation, info)
        # plt.plot(observation)
        self.step_count = 0
        self.lambda_ = 0
#         print(self.accumulated_r, self.accumulated_g, self.accumulated_combo)
        if self.save:
            with open(self.filename, 'ab') as f:
                pickle.dump([self.accumulated_r, self.accumulated_g, self.accumulated_combo], f)
                print('saved')
        self.accumulated_r = 0
        self.accumulated_g = 0
        self.accumulated_combo = 0
        
        if return_info:
            return observation, info
        else:
            return observation

    def render(self, mode="human"):
        self._env.render()

    def close(self):
        self._env.close()

    def preprocess_obs(self, obs, info):
        #         obs = torch.from_numpy(obs).to(dtype=torch.float32) / 255.0 - 0.5
        pr_G = info["current_state"].pr_G
        # pr_G = pr_G[:, np.newaxis]
        #         print(pr_G)
        pr_G = pr_G.reshape(2,1)
        pr_X = info["current_state"].pr_X * 10
        pr_Y1gX = info["current_state"].pr_Y1gX
        observation = np.concatenate((pr_G, pr_X, pr_Y1gX), axis=1)
        observation = observation.flatten()

        return observation