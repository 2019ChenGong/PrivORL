import numpy as np
import pandas as pd
import gym
import d4rl

from synther.diffusion.utils import make_inputs

from sdv.evaluation.single_table import evaluate_quality

# process data

env = gym.make('hopper-medium-replay-v2')
# input = make_inputs(env)
# input = torch.from_numpy(input).float()
dataset = d4rl.qlearning_dataset(env)
obs = dataset['observations']
actions = dataset['actions']
next_obs = dataset['next_observations']
rewards = dataset['rewards']
inputs = np.concatenate([obs, actions, rewards[:, None], next_obs], axis=1)
    
terminals = dataset['terminals'].astype(np.float32)
inputs = np.concatenate([inputs, terminals[:, None]], axis=1)

dpsynther_trajectory = np.load("./results_full_new/hopper_5m_samples_5ep.npz")

observations = dpsynther_trajectory['observations']
actions = dpsynther_trajectory['actions']
rewards = dpsynther_trajectory['rewards']
next_observations = dpsynther_trajectory['next_observations']
terminals = dpsynther_trajectory['terminals']

dpsynther_trajectory = pd.DataFrame({
    'observations': observations.flatten(),
    'actions': actions.flatten(),
    'rewards': rewards.flatten(),
    'next_observations': next_observations.flatten(),
    'terminals': terminals.flatten()
})