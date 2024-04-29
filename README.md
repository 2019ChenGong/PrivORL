# DPTrajectorySyn

<!-- [![Twitter](https://badgen.net/badge/icon/twitter?icon=twitter&label)](https://twitter.com/cong_ml/status/1635642214586937346)
[![arXiv](https://img.shields.io/badge/arXiv-2210.07105-b31b1b.svg)](https://arxiv.org/abs/2303.06614) -->



## Setup

To install, clone the repository and run the following:

```bash 
git submodule update --init --recursive
pip install -r requirements.txt
```

For baseline

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

### Online RL

Baselines (SAC, REDQ):

```bash
# SAC.
python3 synther/online/online_exp.py --env quadruped-walk-v0 --results_folder online_logs/ --exp_name SAC --gin_config_files 'config/online/sac.gin'

# REDQ.
python3 synther/online/online_exp.py --env quadruped-walk-v0 --results_folder online_logs/ --exp_name REDQ --gin_config_files 'config/online/redq.gin'
```

SynthER (SAC):

```bash
# DMC environments.
python3 synther/online/online_exp.py --env quadruped-walk-v0 --results_folder online_logs/ --exp_name SynthER --gin_config_files 'config/online/sac_synther_dmc.gin' --gin_params 'redq_sac.utd_ratio = 20' 'redq_sac.num_samples = 1000000'

# OpenAI environments (different gin config).
python3 synther/online/online_exp.py --env HalfCheetah-v2 --results_folder online_logs/ --exp_name SynthER --gin_config_files 'config/online/sac_synther_openai.gin' --gin_params 'redq_sac.utd_ratio = 20' 'redq_sac.num_samples = 1000000'
```

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
- [REDQ](https://github.com/watchernyu/REDQ)
- [CORL](https://github.com/tinkoff-ai/CORL)




#### dpsynrl

## pretrain(completed)

## fine-tuning

```bash
# run lines 45 to 63, while commenting on lines 65 to 81
# dataset = halfcheetah, epsilon = 5 (uncompleted)
python bash-train-diffusion.py
# saved in f"/p/fzv6enresearch/SynthER/results_{dataset}-medium-replay-v2"
```

## offlineRL

```bash
# run lines 65 to 81, while commenting on lines 45 to 63
python bash-train-diffusion.py
# results will be saved in "./corl_logs/f"DPsynthER-{self.env}-epsilon_{self.dp_epsilon}-seed_{self.seed}-{str(uuid.uuid4())[:8]}"
```
