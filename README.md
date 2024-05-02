<div align=center>
  
# PrivTranR: Differentially Private Synthetic Transition Generation for Offline Reinforcement Learning
</div>

<!-- [![Twitter](https://badgen.net/badge/icon/twitter?icon=twitter&label)](https://twitter.com/cong_ml/status/1635642214586937346)
[![arXiv](https://img.shields.io/badge/arXiv-2210.07105-b31b1b.svg)](https://arxiv.org/abs/2303.06614) -->

This is the official implementaion of paper ***PrivTranR: Differentially Private Synthetic Transition Generation for Offline Reinforcement Learning***. This repository contains Pytorch training code and evaluation code. PRIVIMAGE is a Differetial Privacy (DP) offline RL transitions synthesis tool, which leverages the DP technique to generate synthetic transitions to replace the sensitive data, allowing organizations to share and utilize synthetic images without privacy concerns.

## 1. Contents
- PrivTranR: Differentially Private Synthetic Transition Generation for Offline Reinforcement Learning
  - [1. Contents](#1-contents)
  - [2. Project structure](#2-project-structure)
  - [3. Get Start](#3-get-start)
    - [3.1 Installation](#31-installation)
    - [3.2 Dataset and Files Preparation](#32-dataset-and-files-preparation)
    - [3.3 Training](#33-training)
    - [3.4 Inference](#34-inference)
  - [5. Acknowledgment](#5-acknowledgment)


## 2. Project structure

The structure of this project is as follows:
```
MuJoCo
    -- bash-agent-training.py ------------ the script for downstream tasks (training the agents using the real or synthetic transitions)
    -- bash-baselines-agent-training.py -- the script for downstream task for the baselines (training the agents using the real or synthetic transitions)
    -- bash-evaluate-marginal.py --------- the script for computing the marginal
    -- bash-train-diffusion.py ----------- the script for pre-training and fine-tuning of diffusion models
    -- awac.py --------------------------- training agent using awac algorithm
    -- cql.py ---------------------------- training agent using cql algorithm
    -- iql.py ---------------------------- training agent using iql algorithm
    -- td3_bc.py ------------------------- training agent using td3_bc algorithm
    -- marginal.py ------------------------- computing marginal between synthetic and real transitions
    -- mia.py ---------------------------- mia for the diffusion models

Baselines
    -- 
            
```

## Get Start

To install, clone the repository and run the following:

```bash 
git submodule update --init --recursive
pip install -r requirements.txt
```

For baseline, we refer to the public repository in [SynMeter](https://github.com/zealscott/SynMeter)

The code was tested on Python 3.8 and 3.9.
If you don't have MuJoCo installed, follow the instructions here: https://github.com/openai/mujoco-py#install-mujoco.

## Running Instructions

### marginal

python 

### Offline RL

Diffusion model training (this automatically generates samples and saves them):

```bash
python3 synther/diffusion/train_diffuser.py --dataset halfcheetah-medium-replay-v2
```

Baseline without SynthER (e.g. on TD3+BC):

```bash
python synther/corl/algorithms/td3_bc.py --config synther/corl/yaml/td3_bc/walker2d/medium_replay_v2.yaml --checkpoints_path corl_logs/
```

Offline RL training with SynthER:

```bash
# Generating diffusion samples on the fly.
python3 synther/corl/algorithms/td3_bc.py --config synther/corl/yaml/td3_bc/halfcheetah/medium_replay_v2.yaml --checkpoints_path corl_logs/ --name SynthER --diffusion.path path/to/model-100000.pt


# Using saved dp_diffusion samples.
python td3_bc.py --config synther/corl/yaml/td3_bc/halfcheetah/medium_replay_v2.yaml --checkpoints_path corl_logs/ --name DPsynthER --dp_epsilon 5 --diffusion.path path/to/samples.npz

```

### Baselines


## Thinking of adding SynthER to your own algorithm?

Our codebase has everything you need for diffusion with low-dimensional data along with example integrations with RL algorithms.
For a custom use-case, we recommend starting from the training script and `SimpleDiffusionGenerator` class
in `synther/diffusion/train_diffuser.py`. You can modify the hyperparameters specified in `config/resmlp_denoiser.gin`
to suit your own needs.

## Additional Notes

- Our codebase uses `wandb` for logging, you will need to set `--wandb-entity` across the repository.
- Our pixel-based experiments are based on a modified version of the [V-D4RL](https://github.com/conglu1997/v-d4rl) repository. The latent representations are derived from the trunks of the [actor](https://github.com/conglu1997/v-d4rl/blob/55fde823f3ddb001dd439a701c74390eb3ac34fb/drqbc/drqv2.py#L82) and [critic](https://github.com/conglu1997/v-d4rl/blob/55fde823f3ddb001dd439a701c74390eb3ac34fb/drqbc/drqv2.py#L108C15-L108C15).

## Acknowledgements

SynthER builds upon many works and open-source codebases in both diffusion modelling and reinforcement learning. We
would like to particularly thank the authors of:

- [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch/tree/main/denoising_diffusion_pytorch)
- [CORL](https://github.com/tinkoff-ai/CORL)
- [SynMeter](https://github.com/zealscott/SynMeter)
- [MIA](https://github.com/fseclab-osaka/mia-diffusion)

#### dpsynrl

## pretrain(completed)

## fine-tuning

```bash
# run lines 45 to 63, while commenting on lines 65 to 81
# dataset = halfcheetah, epsilon = 5 (uncompleted)
python bash-train-diffusion.py
# saved in f"./SynthER/results_{dataset}-medium-replay-v2"
```

## offlineRL

```bash
# run lines 65 to 81, while commenting on lines 45 to 63
python bash-train-diffusion.py
# results will be saved in "./corl_logs/f"DPsynthER-{self.env}-epsilon_{self.dp_epsilon}-seed_{self.seed}-{str(uuid.uuid4())[:8]}"
```
