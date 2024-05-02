import pandas as pd
import numpy as np
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, default="pretraining_pategan")
parser.add_argument("--dataset", "-d", type=str, default="kitchen-complete-v0")
parser.add_argument("--epsilon", type=float, default=10.0)
args = parser.parse_args()

baseline_test = "default"
baseline_test = "10iter"

# read_root = f'baselines/exp/{args.dataset}/{args.model}_{args.epsilon}'
read_root = f'baselines/exp/{args.dataset}/{args.model}'

if args.dataset.split('-', 1)[0] == 'kitchen':
    df = pd.read_csv(os.path.join(read_root, f'{baseline_test}_completed_{args.dataset}.csv'))
else:
    df = pd.read_csv(os.path.join(read_root, f'{baseline_test}_{args.dataset}.csv'))

state_columns = [col for col in df.columns if col.startswith('state_')]
states = df[state_columns].to_numpy()

action_columns = [col for col in df.columns if col.startswith('action_')]
actions = df[action_columns].to_numpy()

reward = df['reward'].to_numpy()

next_state_columns = [col for col in df.columns if col.startswith('next_state_')]
next_states = df[next_state_columns].to_numpy()

if 'terminal' not in df.columns:
    terminal = pd.Series(0, index=df.index)
else:
    terminal = df['terminal'].apply(lambda x: 0.0 if x < 0.5 else 1.0).to_numpy()

save_root = f'baselines/samples/{args.dataset}/{args.model}_{args.epsilon}'

if not os.path.exists(save_root):
        os.makedirs(save_root)
np.savez(os.path.join(save_root, f'{baseline_test}_{args.dataset}.npz'), observations=states, actions=actions, next_observations=next_states, rewards=reward, terminals=terminal)
print(f'successfully saved {args.dataset}/{args.model}_{args.epsilon}')




