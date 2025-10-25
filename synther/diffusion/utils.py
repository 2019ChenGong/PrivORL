# Utilities for diffusion.
from typing import Optional, List, Union

import pdb
import d4rl
import gin
import gym
import numpy as np
import torch
from torch import nn
import h5py
from typing import Dict

# GIN-required Imports.
from synther.diffusion.denoiser_network import ResidualMLPDenoiser
from synther.diffusion.elucidated_diffusion import ElucidatedDiffusion
from synther.diffusion.norm import normalizer_factory

from sklearn.model_selection import train_test_split

# Make transition dataset from data.
@gin.configurable
def make_inputs(
        env: gym.Env,
        modelled_terminals: bool = False,
) -> np.ndarray:
    dataset = d4rl.qlearning_dataset(env)
    obs = dataset['observations']
    actions = dataset['actions']
    next_obs = dataset['next_observations']
    rewards = dataset['rewards']
    inputs = np.concatenate([obs, actions, rewards[:, None], next_obs], axis=1)
    if modelled_terminals:
        terminals = dataset['terminals'].astype(np.float32)
        inputs = np.concatenate([inputs, terminals[:, None]], axis=1)
    return inputs


@gin.configurable
def make_part_inputs(
        env: gym.Env,
        sample_ratio: float,
        modelled_terminals: bool = False,
) -> np.ndarray:
    dataset = d4rl.qlearning_dataset(env)
    obs = dataset['observations']
    actions = dataset['actions']
    next_obs = dataset['next_observations']
    rewards = dataset['rewards']
    inputs = np.concatenate([obs, actions, rewards[:, None], next_obs], axis=1)
    if modelled_terminals:
        terminals = dataset['terminals'].astype(np.float32)
        inputs = np.concatenate([inputs, terminals[:, None]], axis=1)
    inputs_pretrain, inputs_finetune = train_test_split(inputs, test_size=1-sample_ratio, random_state=10, shuffle=True)
    return inputs_pretrain, inputs_finetune

# Convert diffusion samples back to (s, a, r, s') format.
@gin.configurable
def split_diffusion_samples(
        samples: Union[np.ndarray, torch.Tensor],
        env: gym.Env,
        modelled_terminals: bool = False,
        terminal_threshold: Optional[float] = None,
):
    # Compute dimensions from env
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    # Split samples into (s, a, r, s') format
    obs = samples[:, :obs_dim]
    actions = samples[:, obs_dim:obs_dim + action_dim]
    rewards = samples[:, obs_dim + action_dim]
    next_obs = samples[:, obs_dim + action_dim + 1: obs_dim + action_dim + 1 + obs_dim]
    if modelled_terminals:
        terminals = samples[:, -1]
        if terminal_threshold is not None:
            if isinstance(terminals, torch.Tensor):
                terminals = (terminals > terminal_threshold).float()
            else:
                terminals = (terminals > terminal_threshold).astype(np.float32)
        return obs, actions, rewards, next_obs, terminals
    else:
        return obs, actions, rewards, next_obs

@gin.configurable
def split_diffusion_samples_epicare(
    samples: Union[np.ndarray, torch.Tensor],
    obs_dim: int,
    action_dim: int,
    modelled_terminals: bool = False,
    terminal_threshold: Optional[float] = None,
):
    obs = samples[:, :obs_dim]
    actions = samples[:, obs_dim:obs_dim + action_dim]
    rewards = samples[:, obs_dim + action_dim]
    next_obs = samples[:, obs_dim + action_dim + 1: obs_dim + action_dim + 1 + obs_dim]
    if modelled_terminals:
        terminals = samples[:, -1]
        if terminal_threshold is not None:
            if isinstance(terminals, torch.Tensor):
                terminals = (terminals > terminal_threshold).float()
            else:
                terminals = (terminals > terminal_threshold).astype(np.float32)
        return obs, actions, rewards, next_obs, terminals
    else:
        return obs, actions, rewards, next_obs


@gin.configurable
def construct_diffusion_model(
        inputs: torch.Tensor,
        normalizer_type: str,
        denoising_network: nn.Module,
        disable_terminal_norm: bool = False,
        skip_dims: List[int] = [],
        cond_dim: Optional[int] = None,
) -> ElucidatedDiffusion:
    event_dim = inputs.shape[1]
    model = denoising_network(d_in=event_dim, cond_dim=cond_dim)

    if disable_terminal_norm:
        terminal_dim = event_dim - 1
        if terminal_dim not in skip_dims:
            skip_dims.append(terminal_dim)

    if skip_dims:
        print(f"Skipping normalization for dimensions {skip_dims}.")

    normalizer = normalizer_factory(normalizer_type, inputs, skip_dims=skip_dims)

    return ElucidatedDiffusion(
        net=model,
        normalizer=normalizer,
        event_shape=[event_dim],
    )


# --------- EpiCare数据加载部分 BEGIN ---------

def get_cutoff(dataset, episodes_avail=None):
    if episodes_avail is None:
        return len(dataset["terminals"])
    terminals = dataset["terminals"][:]
    terminals_cumsum = np.cumsum(terminals)
    cutoff_indices = np.where(terminals_cumsum == episodes_avail)[0]
    if len(cutoff_indices) == 0:
        return len(terminals)
    return cutoff_indices[0] + 1

def load_custom_dataset(dataset_path, episodes_avail=None) -> Dict[str, np.ndarray]:
    with h5py.File(dataset_path, "r") as dataset_file:
        cutoff = get_cutoff(dataset_file, episodes_avail)
        observations = dataset_file["observations"][:cutoff]
        actions = dataset_file["actions"][:cutoff]
        rewards = dataset_file["rewards"][:cutoff]
        next_observations = dataset_file["next_observations"][:cutoff]
        terminals = dataset_file["terminals"][:cutoff]

    observations = observations.astype(np.float32)
    actions = actions.astype(np.float32)
    rewards = rewards.astype(np.float32)
    next_observations = next_observations.astype(np.float32)
    terminals = terminals.astype(np.bool_)

    custom_dataset = {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "next_observations": next_observations,
        "terminals": terminals,
    }
    return custom_dataset

def make_inputs_epicare(
    dataset_path,
    episodes_avail=None,
    modelled_terminals=True
) -> np.ndarray:
    dataset = load_custom_dataset(dataset_path, episodes_avail)
    obs = dataset["observations"]
    actions = dataset["actions"]
    rewards = dataset["rewards"]
    next_obs = dataset["next_observations"]
    if modelled_terminals:
        terminals = dataset["terminals"].astype(np.float32)
        inputs = np.concatenate([obs, actions[:, None], rewards[:, None], next_obs, terminals[:, None]], axis=1)
    else:
        inputs = np.concatenate([obs, actions[:, None], rewards[:, None], next_obs], axis=1)
    return inputs

def make_part_inputs_epicare(
    dataset_path,
    sample_ratio,
    episodes_avail=None,
    modelled_terminals=True
):
    dataset = load_custom_dataset(dataset_path, episodes_avail)
    obs = dataset["observations"]
    actions = dataset["actions"]
    rewards = dataset["rewards"]
    next_obs = dataset["next_observations"]
    if modelled_terminals:
        terminals = dataset["terminals"].astype(np.float32)
        # pdb.set_trace()
        inputs = np.concatenate([obs, actions[:, None], rewards[:, None], next_obs, terminals[:, None]], axis=1)
    else:
        inputs = np.concatenate([obs, actions[:, None], rewards[:, None], next_obs], axis=1)
    inputs_pretrain, inputs_finetune = train_test_split(inputs, test_size=1-sample_ratio, random_state=10, shuffle=True)
    return inputs_pretrain, inputs_finetune