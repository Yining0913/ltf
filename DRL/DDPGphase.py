import matplotlib as mpl
import matplotlib.ticker as ticker
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from jax import device_get

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": "Nimbus Roman",
        "mathtext.fontset": "cm",
        "mathtext.rm": "serif",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def make_replicator_phase_plot(agent, env, seed, filename, res=4):

    x = np.linspace(0.01, 0.99, res)  # q_1
    y = np.linspace(0.01, 0.99, res)  # q_2

    xx, yy = np.meshgrid(x, y)

    # Velocity vectors
    Vx = np.zeros((res, res))  # x-component (group 1)
    Vy = np.zeros((res, res))  # y-component (group 2)

    # Reward AT this point
    R = np.zeros((res, res))

    # Acceptance rate of group 1 AT this point
    AR = np.zeros((res, res))

    # https://eli.thegreenplace.net/2014/meshgrids-and-disambiguating-rows-and-columns-from-cartesian-coordinates/
    # Arrays indexed by row = y = group 2, column = x = group 1.

    for ix in tqdm(range(res)):  # group 1
        for iy in range(res):  # group 2

            # calculate _x_g, _y1_g given d1_g and d0_g
            pr_q = np.array([x[ix], y[iy]])

            observation, init_info = env.reset(
                seed=seed, return_info=True, options={"pr_q": pr_q}
            )
            observation = device_get(observation)

            # let agent decide what policy to take in this fake scenario
            action = agent.policy(observation, init_info)
            observation, reward, terminated, next_info = env.step(action)

            # what is reward for this action?
            R[iy, ix] = reward

            # what is acceptance rate for group 1?
            AR[iy, ix] = next_info["prev_results"].accept_rate[0]

            # what is velocity?
            next_pr_q = next_info["current_state"].pr_Y1
            Vx[iy, ix], Vy[iy, ix] = next_pr_q - pr_q

    # TODO save data intermediate

    fig = plt.figure()
    scale = 6

    fig.set_size_inches(scale, scale)

    ax = fig.add_subplot(1, 1, 1)

    ax.set_xlabel("Group 1 qualification rate $s_1$")
    ax.set_ylabel("Group 2 qualification rate $s_2$")
    ax.xaxis.set_major_locator(ticker.FixedLocator([0.1, 0.3, 0.5, 0.7, 0.9]))
    ax.yaxis.set_major_locator(ticker.FixedLocator([0.1, 0.3, 0.5, 0.7, 0.9]))

    colormin = 1
    colormax = 0

    color_array = AR

    colormin = np.round(min(colormin, np.min(color_array)), 2)
    colormax = np.round(max(colormax, np.max(color_array)), 2)

    thickness = np.sqrt(Vx * Vx + Vy * Vy) * 2.5 + 0.2

    ax.streamplot(xx, yy, Vx, Vy, color="black", linewidth=0.6, arrowsize=0.8)

    # cb_label = 'Classifier Reward (-violation of DP)'
    cb_label = "Group 1 acceptance rate"

    cs = ax.contourf(
        x,
        y,
        color_array,
        cmap=plt.get_cmap("Blues"),
        levels=np.array(np.linspace(colormin, colormax, 9)),
        alpha=0.8,
    )

    cb = fig.colorbar(
        cs,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        ticks=[colormin, colormax],
        ticklocation="left",
    )
    ax.text(
        1.15,
        0.5,
        cb_label,
        rotation=270,
        rotation_mode="anchor",
        horizontalalignment="center",
        verticalalignment="baseline",
        multialignment="center",
    )

    # plot equal qualification line
    ax.plot(
        [0.02, 0.98],
        [0.02, 0.98],
        color="black",
        linewidth=3.5,
        label="Equal qualification rates",
    )

    # Set legend location for equal qualification line
    # https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot/43439132#43439132

    ax.legend(bbox_to_anchor=(0.6, 1.06), loc="lower center", frameon=False)

    # plt.show()
    print("saving", filename)
    plt.savefig(filename, bbox_inches="tight")
