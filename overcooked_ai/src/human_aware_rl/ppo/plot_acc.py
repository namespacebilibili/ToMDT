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


lstm_hp = [{'cramped_room': ((4.0, 3.5777087639996634), 0.6865), 'asymmetric_advantages': ((20.0, 9.797958971132712), 0.2965), 'coordination_ring': ((4.0, 3.5777087639996634), 0.501), 'forced_coordination': ((0.0, 0.0), 0.6375), 'counter_circuit_o_1order': ((8.0, 7.155417527999327), 0.667)}, {'cramped_room': ((8.0, 4.381780460041329), 0.7305), 'asymmetric_advantages': ((4.0, 3.5777087639996634), 0.761), 'coordination_ring': ((4.0, 3.5777087639996634), 0.5905), 'forced_coordination': ((0.0, 0.0), 0.274), 'counter_circuit_o_1order': ((8.0, 7.155417527999327), 0.583)}]
dt_hp = [{'cramped_room': ((92.0, 7.155417527999327), 0.5235), 'asymmetric_advantages': ((52.0, 12.13260071048248), 0.449), 'coordination_ring': ((32.0, 10.73312629199899), 0.4205), 'forced_coordination': ((24.0, 13.145341380123986), 0.456), 'counter_circuit_o_1order': ((4.0, 3.5777087639996634), 0.306)}, {'cramped_room': ((52.0, 18.41738309315414), 0.5605), 'asymmetric_advantages': ((72.0, 16.589153082662175), 0.4365), 'coordination_ring': ((40.0, 8.0), 0.3975), 'forced_coordination': ((12.0, 7.155417527999327), 0.4715), 'counter_circuit_o_1order': ((32.0, 4.381780460041329), 0.3795)}]
lstm_hp_masked = [{'cramped_room': ((0.0, 0.0), 0.02710843373493976), 'asymmetric_advantages': ((12.0, 4.381780460041329), 0.04932301740812379), 'coordination_ring': ((4.0, 3.5777087639996634), 0.15698924731182795), 'forced_coordination': ((0.0, 0.0), 0.008912655971479501), 'counter_circuit_o_1order': ((24.0, 6.693280212272604), 0.19523809523809524)}, {'cramped_room': ((4.0, 3.5777087639996634), 0.046189376443418015), 'asymmetric_advantages': ((16.0, 3.5777087639996634), 0.07305194805194805), 'coordination_ring': ((4.0, 3.5777087639996634), 0.11), 'forced_coordination': ((0.0, 0.0), 0.055299539170506916), 'counter_circuit_o_1order': ((20.0, 5.656854249492381), 0.22425952045133993)}]
dt_masked = [{'cramped_room': ((48.0, 12.13260071048248), 0.23768736616702354), 'asymmetric_advantages': ((64.0, 8.763560920082657), 0.301119023397762), 'coordination_ring': ((24.0, 6.693280212272604), 0.30223390275952694), 'forced_coordination': ((12.0, 7.155417527999327), 0.3726027397260274), 'counter_circuit_o_1order': ((8.0, 4.381780460041329), 0.2131979695431472)}, {'cramped_room': ((64.0, 13.145341380123986), 0.25355450236966826), 'asymmetric_advantages': ((88.0, 15.594870951694343), 0.25894134477825465), 'coordination_ring': ((32.0, 9.121403400793103), 0.27800269905533065), 'forced_coordination': ((16.0, 6.693280212272604), 0.3207236842105263), 'counter_circuit_o_1order': ((24.0, 3.5777087639996634), 0.3878437047756874)}]


def get_value(dic, pos):
    """
    The dictionary consists of layout:((mean, std), acc), and we extract either the mean or the std based on its position
    """
    assert pos == 0 or pos == 1
    ls = []
    for key, values in dic.items():
        ls.append(values[pos])
    return ls


results_0 = [
    get_value(lstm_hp[0], 1),
    get_value(dt_hp[0], 1),
    get_value(lstm_hp_masked[0], 1),
    get_value(dt_masked[0], 1),
    get_value(lstm_hp[1], 1),
    get_value(dt_hp[1], 1),
    get_value(lstm_hp_masked[1], 1),
    get_value(dt_masked[1], 1),
]

hist_algos = [
    "LSTM-HP",
    "DT-HP",
    "LSTM-HP-Masked",
    "DT-HP-Masked",
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
# for each algo, total of 7
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
        )
fst = True
# for h_line in dotted_line:
#     if fst:
#         ax0.hlines(
#             h_line[0],
#             xmin=-0.4,
#             xmax=0.4,
#             colors="red",
#             label="PPO_BC+BC",
#             linestyle=":",
#         )
#         fst = False
#     else:
#         ax0.hlines(h_line[0], xmin=-0.4, xmax=0.4, colors="red", linestyle=":")
#     ax0.hlines(h_line[1], xmin=0.6, xmax=1.4, colors="red", linestyle=":")
#     ax0.hlines(h_line[2], xmin=1.6, xmax=2.4, colors="red", linestyle=":")
#     ax0.hlines(h_line[3], xmin=2.6, xmax=3.45, colors="red", linestyle=":")
#     ax0.hlines(h_line[4], xmin=3.6, xmax=4.4, colors="red", linestyle=":")
ax0.set_ylabel("Prediction Accuracy")
ax0.set_title("Prediction Accuracy with Human Proxy Models")

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

ax0.set_ylim(0, 1)

plt.savefig("acc.pdf", format="pdf", bbox_inches="tight")
plt.show()
