import numpy as np
import gym
import d4rl
import pandas as pd

def make_inputs(
        env: gym.Env
) -> np.ndarray:
    dataset = d4rl.qlearning_dataset(env)
    
    return dataset

if __name__ == '__main__':
    env = gym.make('antmaze-umaze-v1')
    dataset = make_inputs(env)
    obs = dataset['observations']
    actions = dataset['actions']
    next_obs = dataset['next_observations']
    rewards = dataset['rewards']
    terminals = dataset['terminals'].astype(np.float32)
    

    obs_df = pd.DataFrame(obs, columns=[f'state_{i}' for i in range(obs.shape[1])])
    actions_df = pd.DataFrame(actions, columns=[f'action_{i}' for i in range(actions.shape[1])])
    rewards_df = pd.DataFrame(rewards, columns=['reward'])
    next_states_df = pd.DataFrame(next_obs, columns=[f'next_state_{i}' for i in range(next_states.shape[1])])
    terminals_df = pd.DataFrame(terminals, columns=['terminal'])
    print(1)
    