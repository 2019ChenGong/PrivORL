import argparse
import gym
import d4rl
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib as mpl

# Setting font family to 'Liberation Serif' or another available serif font
# mpl.rcParams['font.family'] = 'Liberation Serif'
# mpl.rcParams['font.size'] = 12  # Set default font size

def make_inputs(dataset):
    obs = dataset['observations']
    actions = dataset['actions']
    next_obs = dataset['next_observations']
    rewards = dataset['rewards']
    terminals = dataset['terminals'].astype(np.float32)
    inputs = np.concatenate([obs, actions, rewards[:, None], next_obs, terminals[:, None]], axis=1)
    return inputs

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, default="maze2d-umaze-dense-v1")
    parser.add_argument("--sample_points", type=int, default=1000)
    args = parser.parse_args()

    # Load original and synthetic data
    env = gym.make(args.dataset)
    original_data = d4rl.qlearning_dataset(env)
    original_data = make_inputs(original_data)

    # Dictionary of synthetic datasets with descriptions
    syn_datasets = {
        f'baselines/samples/{args.dataset}/pgm_10.0/{args.dataset}.npz': 'PGM',
        f'baselines/samples/{args.dataset}/privsyn_10.0/{args.dataset}.npz': 'Privsyn',
        f'baselines/samples/{args.dataset}/pategan_10.0/{args.dataset}.npz': 'PATE-GAN',
        f'baselines/samples/{args.dataset}/pretraining_pategan_10.0/10iter_{args.dataset}.npz': 'PrePATE-GAN',
        f'alter_curiosity_driven_results_{args.dataset}_1.0/cleaned_{args.dataset}_samples_1000000.0_10dp_0.8.npz': 'Ours',
    }

    plt.figure(figsize=(25, 5))  # Adjust the size as needed

    try:
        with open(f't-SNE/{args.dataset}_original_indices.npy', 'rb') as f:
            original_indices = np.load(f)
    except FileNotFoundError:
        original_indices = np.random.choice(original_data.shape[0], args.sample_points, replace=False)
        with open(f't-SNE/{args.dataset}_original_indices.npy', 'wb') as f:
            np.save(f, original_indices)

    for i, (syn_path, description) in enumerate(syn_datasets.items()):
        syn_data = np.load(syn_path)
        syn_data = make_inputs(syn_data)
        try:
            with open(f't-SNE/{args.dataset}_{description}.npy', 'rb') as f:
                syn_indices = np.load(f)
        except FileNotFoundError:
            syn_indices = np.random.choice(syn_data.shape[0], args.sample_points, replace=False)
            with open(f't-SNE/{args.dataset}_{description}.npy', 'wb') as f:
                np.save(f, syn_indices)

        selected_original_data = original_data[original_indices]
        selected_syn_data = syn_data[syn_indices]

        combined_experiences = np.concatenate([selected_original_data, selected_syn_data], axis=0)
        tsne = TSNE(n_components=2, random_state=42)
        experiences_2d = tsne.fit_transform(combined_experiences)

        original_points = experiences_2d[:args.sample_points]
        synthetic_points = experiences_2d[args.sample_points:]

        ax = plt.subplot(1, 5, i+1)
        ax.scatter(original_points[:, 0], original_points[:, 1], c='#88DDF1', label='Original Data', s=50, alpha=0.5)
        ax.scatter(synthetic_points[:, 0], synthetic_points[:, 1], c='#FD836E', label=f'Synthetic Data: {description}', s=50, alpha=0.5)
        ax.set_title(f'{description}', fontsize=18, fontweight='bold')  # Increased font size for title
        ax.set_xlabel('')  # Explicitly empty x-axis labels
        ax.set_ylabel('')  # Explicitly empty y-axis labels
        ax.set_xticks([])  # Remove x-axis tick numbers
        ax.set_yticks([])  # Remove y-axis tick numbers
        ax.legend(fontsize=14, title_fontsize=14)  # Increased font size for legend

    plt.tight_layout()
    plt.savefig('t-SNE/all_dpdiffusion.png')
    plt.close()
