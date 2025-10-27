<div align=center>
  
# PrivORL: Differentially Private Synthetic Dataset for Offline Reinforcement Learning
</div>

This is the official implementaion of paper ***PrivORL: Differentially Private Synthetic Dataset for Offline Reinforcement Learning***. This repository contains Pytorch training code and evaluation code. PrivORL leverages a diffusion model and diffusion transformer to synthesize *transitions and trajectories*, respectively, under DP. The synthetic dataset can then be securely released for downstream analysis and research. PrivORL adopts the popular approach of pre-training a synthesizer on public datasets, and then fine-tuning on sensitive datasets using DP Stochastic Gradient Descent (DP-SGD).
Additionally, PrivORL introduces curiosity-driven pre-training, which uses feedback from the curiosity module to diversify the synthetic dataset and thus can generate diverse synthetic transitions and trajectories that closely resemble the sensitive dataset.

<div align=center>
<img src="./plot/privorl_00.png" width = "500" alt="The workflow of PrivCode" align=center />
</div>

<p align="center">The workflow of PrivCode.</p>

## 1. Contents
PrivORL: Differentially Private Synthetic Dataset for Offline Reinforcement Learning

  - [1. Contents](#1-contents)
  - [2. Project structure](#2-project-structure)
  - [3. Get Start](#3-get-start)
    - [3.1 Installation](#31-installation)
    - [3.2 Dataset](#32-dataset)
  - [4. Running Instructions](#running-instructions)
  - [5. Acknowledgment](#5-acknowledgment)


## 2. Project structure

> **Note:** I should clean the structure after editing.


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

## 3. Get Start

### 3.1 Installation

To install, clone the repository and run the following:

```bash 
git submodule update --init --recursive
pip install -r requirements.txt
```

For baseline, we refer to the public repository in [SynMeter](https://github.com/zealscott/SynMeter)

The code was tested on Python 3.9.18.
If you don't have MuJoCo installed, follow the instructions here: https://github.com/openai/mujoco-py#install-mujoco.

### 3.2 Dataset

We use the dataset released from [D4RL](https://github.com/Farama-Foundation/D4RL). Our code will automatically download and preprocess the dataset. We list the supported sensitive dataset as follows,


| Domain         | Sensitive Dataset        |
|---------------|---------------------------|
| Maze2D        | maze2d-umaze            |
| Maze2D        | maze2d-medium           |
| Maze2D        | maze2d-large            |
| FrankaKitchen | kitchen-partial         |
| MuJoCo        | maze2d-medium-replay    |


## 4. Running Instructions

This paper includes PrivORL-n and PrivORL-j for DP offline RL transition and trajectory synthesis. We elaborate on them as follows.

### 4.1 PrviORL-n (Table I)

### 4.1 Curiosity-driven Pre-training

Diffusion model pre-training:

```
python train_diffuser.py --dataset <the-name-of-dataset> --datasets_name <the-pretraining-dataset> --curiosity_driven --curiosity_driven_rate 0.3 --results_folder <the-target-folder>  --save_file_name <store_path> 
```

For example,



### 4.2 Fine-tuning

Fine-tuning the pre-trained diffusion models (this automatically generates samples and saves them):

```
python train_diffuser.py --dataset <the-name-of-dataset> --dp_epsilon 10.0 --results_folder <the-target-folder>  --load_path <the-path-of-saved-pretraining-model> --save_file_name <store_path>
```

### 4.3 Agent Training

Training agents using the synthetic transitions of PrivTranR:

```
python cql/edac/iql/td3_bc.py --env <the-name-of-synthetic-transitions> --checkpoints_path <store_path> --config <the-path-of-configuration-file> --dp_epsilon <the-privacy-budget-of-synthetic-transitions> --diffusion.path <the-path-of-saved-transitions> --name <the-name-of-logging> --prefix <the-prefix-of-name> --save_checkpoints <whether-to-save-ckpt>
```

### 4.4 Abalation

NonCurPrivTranR:

Pre-train the diffusion model without curosity-driven method:

```
python train_diffuser.py --dataset <the-name-of-dataset> --datasets_name <the-pretraining-dataset> --results_folder <the-target-folder>  --save_file_name <store_path> 
```

Fine-tune the pre-trained diffusion model:

```
python train_diffuser.py --dataset <the-name-of-dataset> --dp_epsilon 10.0 --results_folder <the-target-folder>  --load_path <the-path-of-saved-pretraining-model> --save_file_name <store_path>
```

Train the agent using the synthetic transitisons:

```
python cql/edac/iql/td3_bc.py --env <the-name-of-synthetic-transitions> --checkpoints_path <store_path> --config <the-path-of-configuration-file> --dp_epsilon <the-privacy-budget-of-synthetic-transitions> --diffusion.path <the-path-of-saved-transitions> --name <the-name-of-logging> --prefix <the-prefix-of-name> --save_checkpoints <whether-to-save-ckpt>
```

NonPrePrivTranR:

Train the diffusion model without DP protection:
```
python train_NonPrePrivTranR.py --dataset <the-name-of-dataset> --dp_epsilon 10.0 --results_folder <the-target-folder>  --load_path <the-path-of-saved-pretraining-model> --save_file_name <store_path>
```

Train the agent using the synthetic transitions:

```
python cql/edac/iql/td3_bc.py --env <the-name-of-synthetic-transitions> --checkpoints_path <store_path> --config <the-path-of-configuration-file> --dp_epsilon <the-privacy-budget-of-synthetic-transitions> --diffusion.path <the-path-of-saved-transitions> --name <the-name-of-logging> --prefix <the-prefix-of-name> --save_checkpoints <whether-to-save-ckpt>
```

### 4.5 Marginal and Correlation Computing

Compute the marginal and correlation values between the synthetic and real transitions:

```
python marginal.py --dataset <the-name-of-synthetic-transitions> --dp_epsilon <the-privacy-budget-of-synthetic-transitions> --cur_rate <the-curiosity-rate-of-synthetic-transitions> --load_path <the-path-of-saved-transitions>
```

### 4.6 Baselines

Load d4rl datasets and save as csv and json:

```
python baselines/datasets/data_generate.py --dataset <the-name-of-dataset>
```

Split datasets into train, valid and test dataset:

```
python baselines/scripts/data_process.py --dataset <the-name-of-dataset>
```

Train the baseline models, automatically generate samples and save them (pretrain pre-pategan with infinite privacy budget, finetune with privacy budget of 10.0):

```
python baselines/scripts/syn_synthesizer.py --model <baseline-model> --dataset <the-name-of-dataset> --epsilon 10.0 --finetuning <for pre-pategan>
```

Complete the missing dimensions of synthetic transitions:

```
python baselines/scripts/completion.py --model <baseline-model> --dataset <the-name-of-transitions>
```

Save the csv transitions as npz to adapt to downstream tasks:

```
python baselines/scripts/csv_to_npz.py --model <baseline-model> --dataset <the-name-of-transitions> --epsilon 10.0 
```

Train the agent using the synthetic transitions of baselines:

```
python cql/edac/iql/td3_bc.py --env <the-name-of-synthetic-transitions> --checkpoints_path <store_path> --config <the-path-of-configuration-file> --dp_epsilon <the-privacy-budget-of-synthetic-transitions> --diffusion.path <the-path-of-saved-transitions> --name <the-name-of-logging> --prefix <the-prefix-of-name> --save_checkpoints <whether-to-save-ckpt>
```

### 4.7 MIA

Change the args nondp_weight, dp1_weight and dp10_weight to the corresponding checkpoints and run:

```
python mia.py
```


## 5. Acknowledgements

> **Note:** please update the acknowledgements.

PrivORL builds upon many works and open-source codebases in both diffusion modelling and reinforcement learning. We
would like to particularly thank the authors of: [denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch/tree/main/denoising_diffusion_pytorch), [CORL](https://github.com/tinkoff-ai/CORL), [D4RL](https://github.com/Farama-Foundation/D4RL), [SynMeter](https://github.com/zealscott/SynMeter), [MIA](https://github.com/fseclab-osaka/mia-diffusion).
