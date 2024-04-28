import argparse
import numpy as np
import pandas as pd
import torch
import gym
import d4rl
import os
import json
import pickle

from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality

def flatten_multidimensional_columns(df):
    """
    Manually flatten all multi-dimensional columns in a DataFrame.
    """
    for column in df.columns:
        if isinstance(df[column].iloc[0], (list, tuple)):
            dimension = len(df[column].iloc[0])
            expanded_data = np.vstack(df[column].values)
            for i in range(dimension):
                df[f'{column}_{i+1}'] = expanded_data[:, i]
            df.drop(columns=[column], inplace=True)
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='kitchen-partial-v0')
    # parser.add_argument('--dataset', type=str, default='maze2d-medium-dense-v1')
    # parser.add_argument('--dataset', type=str, default='maze2d-umaze-dense-v1')
    parser.add_argument('--pretraining_rate', type=float, default=1.0)
    parser.add_argument('--finetuning_rate', type=float, default=0.8)
    parser.add_argument('--cur_rate', type=float, default=0.1)
    parser.add_argument('--dp_epsilon', type=int, default=10)
    parser.add_argument('--samples', type=float, default=1e6)
    parser.add_argument('--load_path', type=str, default='')
    args = parser.parse_args()

    # Initialize gym environment and extract dataset
    original_env = gym.make(args.dataset)
    original_dataset = d4rl.qlearning_dataset(original_env)
    original_experience = pd.DataFrame({
        'actions': original_dataset['actions'].tolist(),
        'observations': original_dataset['observations'].tolist(),
        'rewards': original_dataset['rewards'].tolist(),
        'next_observations': original_dataset['next_observations'].tolist()
    })

    original_experience = flatten_multidimensional_columns(original_experience)
    tensor_data = {k: torch.tensor(v, dtype=torch.float32).to('cuda') for k, v in original_experience.items()}
    tensor_data_cpu = {k: v.cpu().numpy() for k, v in tensor_data.items()}
    original_experience_cpu = pd.DataFrame(tensor_data_cpu)

    # Load and process synthetic dataset similarly
    dpsynther_trajectory = np.load(args.load_path)
    # dpsynther_trajectory = np.load(f'baselines/samples/{args.dataset}/pategan_10.0/{args.dataset}.npz')
    dpsynther_experience = pd.DataFrame({
        'actions': dpsynther_trajectory['actions'].tolist(),
        'observations': dpsynther_trajectory['observations'].tolist(),
        'rewards': dpsynther_trajectory['rewards'].tolist(),
        'next_observations': dpsynther_trajectory['next_observations'].tolist()
    })

    dpsynther_experience = flatten_multidimensional_columns(dpsynther_experience)
    dpsynther_tensor_data = {k: torch.tensor(v, dtype=torch.float32).to('cuda') for k, v in dpsynther_experience.items()}
    dpsynther_tensor_data_cpu = {k: v.cpu().numpy() for k, v in dpsynther_tensor_data.items()}
    dpsynther_experience_cpu = pd.DataFrame(dpsynther_tensor_data_cpu)

    # Metadata and evaluation (now using the CPU data)

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(original_experience_cpu)

    quality_report = evaluate_quality(
        dataset=args.dataset,
        real_data=original_experience,
        synthetic_data=dpsynther_experience,
        metadata=metadata,
        cur_rate=args.cur_rate,
        epsilon=args.dp_epsilon)
    
    # save_path = f'marginal_results/{args.dataset}_quality_report.pkl'
    # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # quality_report.save(filepath=save_path)

    # save_path = f'marginal_results/{args.dataset}_quality_report.pkl'
    # from sdmetrics.reports.single_table import QualityReport
    # quality_report = QualityReport.load(save_path)
    # with open(save_path, 'rb') as file:
    #     data = pickle.load(file)
    # print(data)