import gym
import d4rl # Import required to register environments, you may need to also import the submodule
import numpy as np
import pandas as pd

# Create the environment
env = gym.make('halfcheetah-expert-v2')

# d4rl abides by the OpenAI gym interface
env.reset()
env.step(env.action_space.sample())

# Each task is associated with a dataset
# dataset contains observations, actions, rewards, terminals, and infos
dataset = env.get_dataset()

print(dataset['observations'].shape[0]) # An N x dim_observation Numpy array of observations

# Alternatively, use d4rl.qlearning_dataset which
# also adds next_observations.
dataset = d4rl.qlearning_dataset(env)

# maze-open-v0 action [1000000, 2] [1000000, 4]
# maze-umaze-v1 action [1000000, 2] [1000000, 4]
# maze-medium-v1 action [2000000, 2] [2000000, 4]
# maze-large-v1 action [4000000, 2] [4000000, 4]
# antmaze-umaze-v1 action [1000000, 8] [1000000, 29]
# antmaze-medium-v1 action [1000000, 8] [1000000, 29]
# antmaze-large-play-v1 action [1000000, 8] [1000000, 29]
# kitchen-complete-v0 action [3680, 9] [3680, 60]
# kitchen-partial-v0 action [136950, 9] [136950, 60]
# kitchen-mixed-v0 action [136950, 9] [136950, 60]

# from synther.diffusion.delete_nan import remove_errors

# dataset = "maze2d-medium-dense-v1"
    

# # original_path = f'curiosity_driven_results_{dataset}_0.3'
# original_path = f'curiosity_driven_results_{dataset}_0.3'
# sample_name = f'{dataset}_samples_1000000.0_10dp_0.5.npz'
# remove_errors(original_path, sample_name)