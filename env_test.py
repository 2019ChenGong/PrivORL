import gym
import d4rl # Import required to register environments, you may need to also import the submodule

# Create the environment
env = gym.make('antmaze-large-play-v1')

# d4rl abides by the OpenAI gym interface
env.reset()
env.step(env.action_space.sample())

# Each task is associated with a dataset
# dataset contains observations, actions, rewards, terminals, and infos
dataset = env.get_dataset()
print(dataset['observations']) # An N x dim_observation Numpy array of observations

# Alternatively, use d4rl.qlearning_dataset which
# also adds next_observations.
dataset = d4rl.qlearning_dataset(env)

# maze-open-v0 action [1000000, 2] [1000000, 4]
# maze-umaze-v1 action [1000000, 2] [1000000, 4]
# maze-medium-v1 action [2000000, 2] [2000000, 4]
# maze-umaze-v1 action [4000000, 2] [4000000, 4]
# antmaze-umaze-v1 action [1000000, 8] [1000000, 29]
# antmaze-medium-v1 action [1000000, 8] [1000000, 29]
# antmaze-large-play-v1 action [1000000, 8] [1000000, 29]
