# PKU 2023 CoRe Project 🧑‍🍳🤖

> Our code is based on the [Overcooked-AI](https://humancompatibleai.github.io) environment.

## Introduction 🥘

Overcooked-AI is a benchmark environment for fully cooperative human-AI task performance, based on the wildly popular video game [Overcooked](http://www.ghosttowngames.com/overcooked/).

The goal of the game is to deliver soups as fast as possible. Each soup requires placing up to 3 ingredients in a pot, waiting for the soup to cook, and then having an agent pick up the soup and delivering it. The agents should split up tasks on the fly and coordinate effectively in order to achieve high reward.

You can **try out the game [here](https://humancompatibleai.github.io/overcooked-demo/)** (playing with some previously trained DRL agents). To play with your own trained agents using this interface, or to collect more human-AI or human-human data, you can use the code [here](https://github.com/HumanCompatibleAI/overcooked_ai/tree/master/src/overcooked_demo). You can find some human-human and human-AI gameplay data already collected [here](https://github.com/HumanCompatibleAI/overcooked_ai/tree/master/src/human_aware_rl/static/human_data).

DRL implementations compatible with the environment are included in the repo as a submodule under src/human_aware_rl. 

The old [human_aware_rl](https://github.com/HumanCompatibleAI/human_aware_rl) is being deprecated and should only used to reproduce the results in the 2019 paper: *[On the Utility of Learning about Humans for Human-AI Coordination](https://arxiv.org/abs/1910.05789)* (also see our [blog post](https://bair.berkeley.edu/blog/2019/10/21/coordination/)).

For simple usage of the environment, it's worthwhile considering using [this environment wrapper](https://github.com/Stanford-ILIAD/PantheonRL).

## Installation ☑️

### Building from source 🔧

It is useful to setup a conda environment with Python 3.7 (virtualenv works too):

```
conda create -n overcooked_ai python=3.7
conda activate overcooked_ai
```

Clone the repo 
```
git clone https://github.com/namespacebilibili/overcooked_ai.git
```
Install requirements:
```
conda install --file requirements.txt
```
And
```
pip install -e overcooked_ai
```

### Reproduce Result

To generate dataset for DT, run
```
cd overcooked_ai/src/human_aware_rl/imitation && python dt_dataset.py
```

Train DT: run
```
cd overcooked_ai/src/human_aware_rl/imitation && python train_dt.py
```

To reproduce the BC result, run
```
cd overcooked_ai/src/human_aware_rl/imitation && python reproduce_bc.py
```

To train self-play PPO, run
```
cd overcooked_ai/src/human_aware_rl/ppo && ./run_experiment.sh
```

To evaluate our agent, run the corresponding function at `overcooked_ai/src/human_aware_rl/ppo/evaluate.py`.

## Python Visualizations 🌠

See [this Google Colab](https://colab.research.google.com/drive/1AAVP2P-QQhbx6WTOnIG54NXLXFbO7y6n#scrollTo=Z1RBlqADnTDw) for some sample code for visualizing trajectories in python.

We have incorporated a [notebook](Overcooked%20Tutorial.ipynb) that guides users on the process of training, loading, and evaluating agents. Ideally, we would like to enable users to execute the notebook in Google Colab; however, due to Colab's default kernel being Python 3.10 and our repository being optimized for Python 3.7, some functions are presently incompatible with Colab. To provide a seamless experience, we have pre-executed all the cells in the notebook, allowing you to view the expected output when running it locally following the appropriate setup.

Overcooked_demo can also start an interactive game in the browser for visualizations. Details can be found in its [README](src/overcooked_demo/README.md)

## Raw Data :ledger:

The raw data used in training is >100 MB, which makes it inconvenient to distribute via git. The code uses pickled dataframes for training and testing, but in case one needs to original data it can be found [here](https://drive.google.com/drive/folders/1aGV8eqWeOG5BMFdUcVoP2NHU_GFPqi57?usp=share_link) 

