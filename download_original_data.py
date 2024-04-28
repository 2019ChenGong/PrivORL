# Train diffusion model on D4RL transitions.
import argparse
import pathlib
import os
import d4rl
import gin
import gym
import numpy as np
import torch
import wandb
import ast

from synther.diffusion.utils import make_inputs, make_part_inputs, split_diffusion_samples, construct_diffusion_model


def load_data(dataset_name):
    env = gym.make(dataset_name)
    input = make_inputs(env, modelled_terminals = True)
 
    with open(f'saved_original_data/{dataset_name}_original_data.npy', 'wb') as f:
        np.save(f, input)

load_data('walker2d-medium-replay-v2')