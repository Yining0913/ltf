from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from fairgym.plotting.video_ctx_mgr import Video


class DDPGExperiment:
    def __init__(self, agent, env):

        self.agent = agent
        self.env = env

        # for plotting
        self.fig = plt.figure()
        n_axs = 2
        scale = 6

        self.fig.set_size_inches(n_axs * scale, scale)

        self.axs = []
        for i in range(n_axs):
            self.axs.append(self.fig.add_subplot(1, n_axs, i + 1))

        self.history = defaultdict(list)
        self.init_observation, self.info = None, None

    def reset(self, **kwargs):
        self.init_observation, self.info = self.env.reset(**kwargs)

    def render(self, observation, info, val_dict=None):
        """
        Plotting
        """

        state = info["current_state"]
        results = info["prev_results"]

        ########################################################################
        ax = self.axs[0]

        # P(X | G) and P(Y=1 | X, G)
        for g in range(self.env.num_groups):
            if g == 0:
                color1 = "#00556d"  # g0 X
                color2 = "#13a5cd"  # g0 Y | X
            else:
                color1 = "#8f003a"  # g1 X
                color2 = "#ff24da"  # g1 Y | X
            ax.plot(state.pr_X[g], label=f"Pr(X=x|G={g})", color=color1)
            ax.plot(state.pr_Y1gX[g], label=f"Pr(Y=1|X=x,G={g})", color=color2)

        # Thresholds
        if results is not None:
            action = results.action
            thresholds = (
                action.reshape(self.env.num_groups, 1)
                * (self.env._env.num_feature_bins - 1)
                * 0.999
            )

            for i, t in enumerate(thresholds):
                if i == 0:
                    color = "green"
                else:
                    color = "red"

                ax.plot(np.ones(2) * t, [0, 1], label=f"Threshold (G={i})", color=color)

        ax.legend(loc="upper left")

        ########################################################################

        # History

        if val_dict is None:
            val_dict = {}

        for k, v in val_dict.items():
            self.history[k].append(v[0])

        ax = self.axs[1]
        for k, v in val_dict.items():
            ax.plot(self.history[k], label=k, color=v[1])

        ax.legend(loc="upper right")

    def record(self, name, n_steps=25, fps=10, render_flag=True, to_plot=None):
        """
        Run loop and render frames to video
        """

        observation, info = self.init_observation, self.info

        with Video(name, self.fig, render_flag=render_flag, fps=fps) as video:

            if to_plot is None:
                val_dict = {}
            else:
                val_dict = to_plot(self.env, self.agent, info)

            if render_flag:
                self.render(observation, info, val_dict)
                video.draw()

            for _ in tqdm(range(n_steps)):

                action = self.agent.policy(observation, info)
                observation, reward, terminated, info = self.env.step(action)

                if to_plot is None:
                    val_dict = {}
                else:
                    val_dict = to_plot(self.env, self.agent, info)

                if render_flag:
                    self.render(observation, info, val_dict)
                    video.draw()

                if terminated:
                    observation, info = self.env.reset(return_info=True)
                    self.env.close()
                    break

            self.env.close()

        return observation, reward, terminated, info
