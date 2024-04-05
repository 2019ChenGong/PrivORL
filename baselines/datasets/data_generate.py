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
    row_num = len(terminals)
    label = np.ones((row_num, 1))
    
    obs_df = pd.DataFrame(obs, columns=[f'state_{i}' for i in range(obs.shape[1])])
    actions_df = pd.DataFrame(actions, columns=[f'action_{i}' for i in range(actions.shape[1])])
    rewards_df = pd.DataFrame(rewards, columns=['reward'])
    next_obs_df = pd.DataFrame(next_obs, columns=[f'next_state_{i}' for i in range(next_obs.shape[1])])
    terminals_df = pd.DataFrame(terminals, columns=['terminal'])
    label_df = pd.DataFrame(label, columns=['label'])

    df = pd.concat([obs_df, actions_df, rewards_df, next_obs_df, terminals_df, label_df], axis=1)

    df.to_csv('baselines/datasets/antmaze-umaze-v1/antmaze-umaze-v1.csv', index=False)

    # generate json
    json_columns = []
    for col in df.columns:
        col_info = {
            "min": df[col].min(),
            "max": df[col].max(),
            "name": col,
            "type": "continuous"
        }
        json_columns.append(col_info)
    json_data = {
    "columns": json_columns,    
    "task": "regression",
    "train_size": 8710,
    "val_size": 2178,
    "test_size": 2723,
    "label": "Class"
    }


    print(1)
    