import argparse
import numpy as np
import pandas as pd
import gym
import d4rl

from synther.diffusion.utils import make_inputs

from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality


def flatten_multidimensional_columns(df):
    """
    Automatically flatten all multi-dimensional columns in a DataFrame.
    :param df: DataFrame to be processed
    :return: flattened DataFrame
    """
    for column in df.columns:
        # Check if the first element in each column is a list (or similarly iterable)
        if isinstance(df[column].iloc[0], (list, tuple)):
            # Determine the length of the list from the first row
            dimension = len(df[column].iloc[0])
            # Create new columns for each element in the list
            expanded_cols = pd.DataFrame(df[column].tolist(), index=df.index, columns=[f'{column}_{i+1}' for i in range(dimension)])
            # Concatenate the new columns and drop the original column
            df = pd.concat([df, expanded_cols], axis=1).drop(columns=[column])
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='halfcheetah')
    args = parser.parse_args()

    # process data

    original_env = gym.make(f'{args.dataset}-medium-replay-v2')
    
    if args.dataset == 'hopper':
        column_dimension = {
            'observations': 11,
            'actions': 3,
            'next_observations': 11
        }
    elif args.dataset == 'halfcheetah':
        column_dimension = {
            'observations': 17,
            'actions': 6,
            'next_observations': 17
        }
    # elif args.dataset == 'walker2d':
        # column_dimension = {
        #     'observations': 11,
        #     'actions': 3,
        #     'next_observations': 11
        # }

    original_dataset = d4rl.qlearning_dataset(original_env)
    original_obs = original_dataset['observations']
    original_actions = original_dataset['actions']
    original_next_obs = original_dataset['next_observations']
    original_rewards = original_dataset['rewards']
    original_terminals = original_dataset['terminals'].astype(np.float32)

    original_data = {
        'actions': original_actions.tolist(),
        'observations': original_obs.tolist(),
        'rewards': original_rewards.tolist(),
        'next_observations': original_next_obs.tolist(),
        'terminals': original_terminals.tolist()
    }
    original_experience = pd.DataFrame(original_data)

    original_experience = flatten_multidimensional_columns(original_experience)

    # trajectories = []
    # trajectory = []
    # for i, input in enumerate(inputs):
    #     trajectory.append(input)
    #     if terminals[i] == 1.0 or i == len(inputs) - 1:
    #         trajectory = np.array(trajectory)
    #         trajectories.append(trajectory)
            
    #         trajectory = []

    dpsynther_trajectory = np.load(f"results_{args.dataset}-medium-replay-v2/{args.dataset}-medium-replay-v2_samples_5000000.0_8dp.npz")

    dpsynther_obs = dpsynther_trajectory['observations']
    dpsynther_actions = dpsynther_trajectory['actions']
    dpsynther_rewards = dpsynther_trajectory['rewards']
    dpsynther_next_obs = dpsynther_trajectory['next_observations']
    dpsynther_terminals = dpsynther_trajectory['terminals']

    dpsynther_data = {
        'actions': dpsynther_actions.tolist(),
        'observations': dpsynther_obs.tolist(),
        'rewards': dpsynther_rewards.tolist(),
        'next_observations': dpsynther_next_obs.tolist(),
        'terminals': dpsynther_terminals.tolist()
    }
    dpsynther_experience = pd.DataFrame(dpsynther_data)

    dpsynther_experience = flatten_multidimensional_columns(dpsynther_experience, column_dimension)

    # generate metadata
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(dpsynther_experience)

    quality_report = evaluate_quality(
        real_data=original_experience,
        synthetic_data=dpsynther_experience,
        metadata=metadata)

