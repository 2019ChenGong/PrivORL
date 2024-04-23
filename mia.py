import torch
import numpy as np
import argparse
import gym
import d4rl
import os
from accelerate import Accelerator

from synther.diffusion.utils import construct_diffusion_model



def make_inputs(dataset):
    obs = dataset['observations']
    actions = dataset['actions']
    next_obs = dataset['next_observations']
    rewards = dataset['rewards']
    terminals = dataset['terminals'].astype(np.float32)
    inputs = np.concatenate([obs, actions, rewards[:, None], next_obs, terminals[:, None]], axis=1)
    return inputs


def get_data_and_model(config):
    results_folder = f"./alter_curiosity_driven_results_{config.dataset}_{config.pretraining_rate}"
    ckpt_path = os.path.join(results_folder, "pretraining-model-9.pt")
    dp_ckpt_path = os.path.join(results_folder, "finetuning-model-4.pt")
    syndata_path = os.path.join(results_folder, f"cleaned_{config.dataset}_samples_{1e6}_10dp_{config.finetuning_rate}.npz")

    # load orginal and syn data
    env = gym.make(config.dataset)
    original_data = d4rl.qlearning_dataset(env)
    original_data = make_inputs(original_data)

    syn_data = np.load(syndata_path)
    syn_data = make_inputs(syn_data)

    # load finetuning model
    diffusion = torch.load(ckpt_path)
    dpdiffusion = torch.load(dp_ckpt_path)

    return original_data, syn_data, diffusion, dpdiffusion

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, default="kitchen-complete-v0")
    parser.add_argument("--pretraining_rate", type=float, default=1.0)
    parser.add_argument("--finetuning_rate", type=float, default=0.8)
    args = parser.parse_args()

    original_data, syn_data, diffusion, dpdiffusion = get_data_and_model(config=args)