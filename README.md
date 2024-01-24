# PKU 2023 CoRe Project 🧑‍🍳🤖

This is the repo for the PKU 2023 CoRe Submission *Decision Transformer for Modeling Theory of Mind in Multi-Agent Systems*.

> Our code is based on the [Overcooked-AI](https://humancompatibleai.github.io) environment.

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

