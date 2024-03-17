import numpy as np
import pandas as pd
import gym

from synther.diffusion.utils import make_inputs

from sdv.evaluation.single_table import evaluate_quality

# process data

env = gym.make('hopper-medium-replay-v2')
input = make_inputs(env)
input = torch.from_numpy(input).float()


dpsynther_trajectory = np.load("your_saved_file.npz")

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