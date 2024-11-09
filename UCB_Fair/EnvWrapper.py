import os
import numpy as np
from functools import partial
import gymnasium
from jax import numpy as jnp

from fairgym.envs.default_reward import (
    zero_one_loss,
    euclidean_demographic_partity,
    qualification_rate_disparity,
    zero_one_accuracy,
)
from fairgym.envs.replicator.qualification_replicator_env import (
    GroupQualificationReplicatorEnv,
)
import math
import time
import matplotlib.pyplot as plt


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


# def reward_fn_(
#     lambda_,
#     accuracy_term,
#     disparity_term,
#     prev_state,
#     prev_action,
#     prev_results,
#     current_state,
# ):
#
#     """
#     What the environment's reward signal is, as a function
#     of the action we just took
#
#     TODO Even though this gets jitted in env.step, we could jit it here to hold the
#      (changing on reset) args as static.
#     """
#     accuracy = accuracy_term(prev_state, prev_action, prev_results, current_state)
#     disparity = disparity_term(prev_state, prev_action, prev_results, current_state)
#
#     return accuracy * (1 - lambda_) + lambda_ * (disparity)


class CustomEnv(gymnasium.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        done=False,
        episode_len=100,
        lbd=0.9,
        loss=zero_one_loss,
        disparity=euclidean_demographic_partity,
        seed=0,
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
        )
        self.action_space = gymnasium.spaces.Box(0.0, 1.0, (2,), dtype=np.float32)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = gymnasium.spaces.Box(
            0, 1, (2, 32, 2), dtype=np.float32
        )
        self.num_groups = 2
        self.feature_bins = 32
        self.done = done
        self.episode_len = episode_len
        self.seed = seed

    def step(self, action):
        observation, reward, done, illegal_state, info = self._env.step(action)
        self.step_count += 1
        # print("seed", self.seed)
        observation = self.preprocess_obs(observation, info)

        if self.done:
            if self.step_count >= self.episode_len:
                done = True
        #             if reward <= 0.6:
        #                 done = True

        return observation, reward, done, info

    def reset(self, return_info=False):

        # t = int(str(time.time()).split(".")[1])
        # np.random.seed(seed=self.seed)
        # pr_q0 = np.random.rand()
        # print(pr_q0)
        # option = {"pr_q": np.array([0.5, 0.5])}
        option = {
            # "pr_q": np.array([pr_q0, 1 - pr_q0]),
            "pr_G": np.array([0.5, 0.5]),
        }
        observation, info = self._env.reset(
            return_info=True, seed=self.seed, options=option
        )
        self.seed += 1
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
        pr_G = pr_G[:, np.newaxis]
        #         print(pr_G)
        pr_G = pr_G
        pr_X = info["current_state"].pr_X * 10
        pr_Y1gX = info["current_state"].pr_Y1gX
        observation = np.concatenate(
            (pr_X[:, :, np.newaxis], pr_Y1gX[:, :, np.newaxis]), axis=2
        )
        # observation = observation.flatten()
        # pr_Y1gX.at[:, 1:].set(np.diff(pr_Y1gX, axis=1))
        # np.diff(pr_Y1gX, axis=1)

        # observation = np.concatenate((pr_G, pr_X, np.diff(pr_Y1gX, axis=1)), axis=1)
        # observation[:, 33:] = pr_Y1gX[:, :-1]
        # observation = observation[:, :, np.newaxis]
        # observation = np.repeat(np.expand_dims(observation, axis=2), 64, axis=2)
        #
        # observation[:, :, 50:] = np.expand_dims(observation[:, 0, :], axis=2).repeat(14, axis=2)
        #
        #
        # observation = np.repeat(observation, 2, axis=0)
        # observation[1] = observation[2]
        # observation[2] = observation[1] - observation[0]
        # observation = observation[:3]

        # plott(observation)
        # plt.plot(observation1[0])
        # plt.plot(observation1[1])
        # obs = torch.diff(obs)
        # self.plott(observation)
        return observation

    def plott(self, observation):
        plt.figure()
        plt.plot(observation[0, :, 0], "red")
        plt.plot(observation[1, :, 0], "blue")
        plt.plot(observation[0, :, 1], "green")
        plt.plot(observation[1, :, 1], "purple")
