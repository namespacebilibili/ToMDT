import os

import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch


# importing from utils causes werid dependency conflicts. Copying here
def set_style(font_scale=1.6):
    import matplotlib
    import seaborn

    seaborn.set(font="serif", font_scale=font_scale)
    # Make the background white, and specify the specific font family
    seaborn.set_style(
        "white",
        {
            "font.family": "serif",
            "font.weight": "normal",
            "font.serif": ["Times", "Palatino", "serif"],
            "axes.facecolor": "white",
            "lines.markeredgewidth": 1,
        },
    )
    matplotlib.rcParams["text.usetex"] = True
    matplotlib.rc("font", family="serif", serif=["Palatino"])

hp_lstm = [{'cramped_room': ((4.0, 3.5777087639996634), 0.6865), 'asymmetric_advantages': ((20.0, 9.797958971132712), 0.2965), 'coordination_ring': ((4.0, 3.5777087639996634), 0.501), 'forced_coordination': ((0.0, 0.0), 0.6375), 'counter_circuit_o_1order': ((8.0, 7.155417527999327), 0.667)}, {'cramped_room': ((8.0, 4.381780460041329), 0.7305), 'asymmetric_advantages': ((4.0, 3.5777087639996634), 0.761), 'coordination_ring': ((4.0, 3.5777087639996634), 0.5905), 'forced_coordination': ((0.0, 0.0), 0.274), 'counter_circuit_o_1order': ((8.0, 7.155417527999327), 0.583)}]
hp_dt = [{'cramped_room': ((54.0, 6.356099432828281), 0.586), 'asymmetric_advantages': ((72.0, 13.023056476879765), 0.48275), 'coordination_ring': ((28.0, 8.579044235810887), 0.49375), 'forced_coordination': ((18.0, 5.253570214625478), 0.524), 'counter_circuit_o_1order': ((14.0, 2.8982753492378874), 0.44625)}, {'cramped_room': ((64.0, 8.390470785361211), 0.6175), 'asymmetric_advantages': ((36.0, 4.732863826479693), 0.55575), 'coordination_ring': ((36.0, 6.81175454637056), 0.47275), 'forced_coordination': ((12.0, 4.195235392680606), 0.5595), 'counter_circuit_o_1order': ((4.0, 2.5298221281347035), 0.47575)}]
hp_ppo = [{'cramped_room': ((120.0, 13.856406460551018), 0), 'asymmetric_advantages': ((56.0, 24.26520142096496), 0), 'coordination_ring': ((56.0, 15.388307249337075), 0), 'forced_coordination': ((32.0, 7.155417527999327), 0), 'counter_circuit_o_1order': ((0.0, 0.0), 0)}, {'cramped_room': ((116.0, 10.430723848324238), 0), 'asymmetric_advantages': ((64.0, 29.06544339933592), 0), 'coordination_ring': ((68.0, 15.594870951694343), 0), 'forced_coordination': ((8.0, 7.155417527999327), 0), 'counter_circuit_o_1order': ((36.0, 3.5777087639996634), 0)}]
hp_tom = [{'cramped_room': ((128.0, 7.155417527999327), 0), 'asymmetric_advantages': ((112.0, 24.396721091163048), 0), 'coordination_ring': ((92.0, 12.13260071048248), 0), 'forced_coordination': ((12.0, 4.381780460041329), 0), 'counter_circuit_o_1order': ((0.0, 0.0), 0)}, {'cramped_room': ((104.0, 10.430723848324238), 0), 'asymmetric_advantages': ((92.0, 23.73183515870612), 0), 'coordination_ring': ((76.0, 6.693280212272604), 0), 'forced_coordination': ((24.0, 8.763560920082657), 0), 'counter_circuit_o_1order': ((24.0, 6.693280212272604), 0)}]

def get_value(dic, pos):
    """
    The dictionary consists of layout:(mean, std), and we extract either the mean or the std based on its position
    """
    assert pos == 0 or pos == 1
    ls = []
    for key, values in dic.items():
        ls.append(values[0][pos])
    return ls


results_0 = [
    get_value(hp_lstm[0], 0),
    get_value(hp_dt[0], 0),
    get_value(hp_ppo[0], 0),
    get_value(hp_tom[0], 0),
    get_value(hp_lstm[1], 0),
    get_value(hp_dt[1], 0),
    get_value(hp_ppo[1], 0),
    get_value(hp_tom[1], 0),
]
stds = [
    get_value(hp_lstm[0], 1),
    get_value(hp_dt[0], 1),
    get_value(hp_ppo[0], 1),
    get_value(hp_tom[0], 1),
    get_value(hp_lstm[1], 1),
    get_value(hp_dt[1], 1),
    get_value(hp_ppo[1], 1),
    get_value(hp_tom[1], 1),
]


hist_algos = [
    "LSTM-HP",
    "DT-HP",
    "PPO-HP",
    "ToM-HP"
]
set_style()

fig, ax0 = plt.subplots(1, figsize=(18, 6))  # figsize=(20,6))

plt.rc("legend", fontsize=21)
plt.rc("axes", titlesize=25)
ax0.tick_params(axis="x", labelsize=18.5)
ax0.tick_params(axis="y", labelsize=18.5)

# there are 5 layouts
ind = np.arange(5)
width = 0.1
deltas = [-2.9, -1.5, -0.5, 0.5, 1.9, 2.9, 3.9, 4.9]
# everforest color: #264653
colors = ["#aeaeae", "#2d6777", "#F79646", "#264653"]

# in each loop, we plot the result for all 5 layouts for each algo
for i in range(len(results_0)):
    delta, algo = deltas[i], hist_algos[i % 4]
    offset = ind + delta * width
    color = colors[i % 4]
    if 0 <= i <= 3:
        ax0.bar(
            offset,
            results_0[i],
            width,
            color=color,
            lw=1.0,
            zorder=0,
            label=algo,
            yerr=stds[i],
        )
    else:
        ax0.bar(
            offset,
            results_0[i],
            width,
            color=color,
            edgecolor="white",
            lw=1.0,
            zorder=0,
            hatch="/",
            yerr=stds[i],
        )
fst = True

ax0.set_ylabel("Average reward per episode")
ax0.set_title("Performance with Human Proxy Models")

ax0.set_xticks(ind + width / 3)
ax0.set_xticklabels(
    (
        "Cramped Rm.",
        "Asymm. Adv.",
        "Coord. Ring",
        "Forced Coord.",
        "Counter Circ.",
    )
)

ax0.tick_params(axis="x", labelsize=18)
handles, labels = ax0.get_legend_handles_labels()
patch = Patch(
    facecolor="white",
    edgecolor="black",
    hatch="/",
    alpha=0.5,
    label="Switched start indices",
)
handles.append(patch)

# plot the legend
ax0.legend(handles=handles, loc="best")

ax0.set_ylim(0, 150)

plt.savefig("reward.pdf", format="pdf", bbox_inches="tight")
plt.show()
