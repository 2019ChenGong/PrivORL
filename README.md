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
    -- train_diffuser.py ----------------- pre-training and fine-tuning of diffusion models
    -- marginal.py ----------------------- computing marginal between synthetic and real transitions
    -- mia.py ---------------------------- mia for the diffusion models
    -- synther
      -- corl ---------------------------- configuration files of downstream tasks
      -- diffuser ------------------------ utilities and scripts of diffusion model
    -- t-SNE ----------------------------- t-SNE visualization
    -- config ---------------------------- gin configuration files of PrivTranR training and synthetizing
Baselines
    -- datasets
      -- maze2d-umaze-dense-v1
      -- ...
      -- walker2d-medium-repaly-v2 ------- csv and json files of your own datasets
      -- bash-data-generate.py ----------- the script for loading d4rl datasets and saving csv and json
      -- data-generate.py ---------------- loading d4rl datasets and saving csv and json
    -- exp ------------------------------- model weights and synthetic transitions
    -- samples --------------------------- npz files of synthetic transitions for downstream tasks      
    -- scripts
      -- bash-data-process.py ------------ the script for splitting datasets
      -- bash-train-baselines.py --------- the script for training and synthetizing of baselines
      -- bash-completion.py -------------- the script for completing the missing dimensions of synthetic transitions
      -- data_process.py ----------------- splitting datasets
      -- syn_synthesizer.py -------------- training and synthetizing of baselines
      -- completion.py ------------------- completing the missing dimensions of synthetic transitions
    -- synthesizer
      -- pgm.py -------------------------- training pgm and synthetizing transitions
      -- privsyn.py ---------------------- training privsyn and synthetizing transitions
      -- pategan.py ---------------------- training pategan and synthetizing transitions
      -- pretraining_pategan.py ---------- training pretraining_pategan and synthetizing transitions
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

### Curiosity-driven Pre-training

Diffusion model pre-training (this automatically generates samples and saves them):

```
python --dataset <the-name-of-dataset> --datasets_name <the-pretraining-dataset> ----curiosity_driven --curiosity_driven_rate 0.3 --dp_epsilon 10.0 --results_folder <the-target-folder>  --save_file_name <store_path> 
```

### Fine-tuning

Fine-tuning the pre-trained diffusion models:

```
python 
```

### Agent Training

Training agents using the synthetic transitions of PrivTranR

```
python 
```

### Abalation

```
python
```

### Marginal and Correlation Computing

```
python 
```

### Baselines

```
python
```

### MIA


## Acknowledgements

SynthER builds upon many works and open-source codebases in both diffusion modelling and reinforcement learning. We
would like to particularly thank the authors of:

- [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch/tree/main/denoising_diffusion_pytorch)
- [CORL](https://github.com/tinkoff-ai/CORL)
- [SynMeter](https://github.com/zealscott/SynMeter)
- [MIA](https://github.com/fseclab-osaka/mia-diffusion)
