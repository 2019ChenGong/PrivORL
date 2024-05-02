import pandas as pd
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, default="pretraining_pategan")
parser.add_argument("--dataset", "-d", type=str, default="kitchen-complete-v0")
args = parser.parse_args()

baseline_test = "default"
baseline_test = "10iter"

original_path = f'baselines/exp/{args.dataset}/pategan_eps_1_10.0/{args.dataset}.csv'
df_original = pd.read_csv(original_path)
original_columns_order = df_original.columns
state_columns_original = [col for col in df_original.columns if col.startswith('state_')]
next_state_columns_original = [col for col in df_original.columns if col.startswith('next_state_')]

# df_target = pd.read_csv(f'baselines/exp/{args.dataset}/{args.model}_10.0/{args.dataset}.csv')
df_target = pd.read_csv(f'baselines/exp/{args.dataset}/{args.model}/{baseline_test}_{args.dataset}.csv')
state_columns_target = [col for col in df_target.columns if col.startswith('state_')]
next_state_columns_target = [col for col in df_target.columns if col.startswith('next_state_')]

is_partial = False

for col in original_columns_order:
    if col.startswith('state_'):
        if col not in state_columns_target:
            is_partial = True
            print('there is missing')
            df_target[col] = df_original[col]
    elif col.startswith('next_state_'):
        if col not in next_state_columns_target:
            is_partial = True
            df_target[col] = df_original[col]

df_target = df_target[original_columns_order]

# df_target.to_csv(f'baselines/exp/{args.dataset}/{args.model}_10.0/completed_{args.dataset}.csv', index=False)
if is_partial:
    df_target.to_csv(f'baselines/exp/{args.dataset}/{args.model}/{baseline_test}_completed_{args.dataset}.csv', index=False)
