import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import d4rl
from torch.nn.utils.rnn import pad_sequence
from transformers import GPT2Model


class TrajectoryEmbedding(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size=768):
        super().__init__()
        self.hidden_size = hidden_size
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.mlp_state = nn.Linear(state_dim, hidden_size)
        self.mlp_action = nn.Linear(action_dim, hidden_size)
        self.mlp_reward = nn.Linear(1, hidden_size)
        self.mlp_terminal = nn.Linear(1, hidden_size)
        self.mlp_next_state = nn.Linear(state_dim, hidden_size)

        self.gpt2 = GPT2Model.from_pretrained("gpt2")
        for param in self.gpt2.parameters():
            param.requires_grad = False

        self.initialize_mlps()
        for mlp in [self.mlp_state, self.mlp_action, self.mlp_reward, self.mlp_terminal, self.mlp_next_state]:
            for param in mlp.parameters():
                param.requires_grad = False

    def initialize_mlps(self):
        for mlp in [self.mlp_state, self.mlp_action, self.mlp_reward, self.mlp_terminal, self.mlp_next_state]:
            nn.init.xavier_uniform_(mlp.weight)
            mlp.bias.data.zero_()

    def forward(self, original_trajectory, synthetic_trajectory):

        max_batch_size = 100
        # pdb.set_trace()
        original_trajectory = original_trajectory[:max_batch_size]
        synthetic_trajectory = synthetic_trajectory[:max_batch_size]

        print(f"Reshaped original_trajectory: {original_trajectory.shape}")  
        print(f"Reshaped synthetic_trajectory: {synthetic_trajectory.shape}")  

        original_emb = self.compute_embedding(original_trajectory)  # (100, 1000, hidden_size)
        synthetic_emb = self.compute_embedding(synthetic_trajectory)  # (100, 992, hidden_size)

        similarity_score = self.cosine_similarity(original_emb, synthetic_emb)
        
        return similarity_score

    
    def compute_embedding(self, trajectory):
        """
        trajectory: (batch_size, seq_len, feature_dim)
        """
        max_batch_size = 100
        if trajectory.shape[0] > max_batch_size:
            trajectory = trajectory[:max_batch_size]

        state, action, reward, terminal, next_state = torch.split(
            trajectory, [self.state_dim, self.action_dim, 1, 1, self.state_dim], dim=-1
        )

        state_embed = self.mlp_state(state)
        action_embed = self.mlp_action(action)
        reward_embed = self.mlp_reward(reward)
        terminal_embed = self.mlp_terminal(terminal)
        next_state_embed = self.mlp_next_state(next_state)

        inputs_embeds = (state_embed + action_embed + reward_embed + terminal_embed + next_state_embed) / 5

        print(f"Final inputs_embeds shape: {inputs_embeds.shape}")  # (batch_size, seq_len, hidden_size=768)

        # pdb.set_trace()
        outputs = self.gpt2(inputs_embeds=inputs_embeds)

        return outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

    def cosine_similarity(self, emb1, emb2):
        emb1 = emb1.mean(dim=1)
        emb2 = emb2.mean(dim=1)

        similarity = F.cosine_similarity(emb1, emb2, dim=-1)
        return similarity.mean()



def load_d4rl_dataset(env_name):
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)

    states = torch.tensor(dataset['observations'], dtype=torch.float32)
    actions = torch.tensor(dataset['actions'], dtype=torch.float32)
    rewards = torch.tensor(dataset['rewards'], dtype=torch.float32).unsqueeze(-1)
    terminals = torch.tensor(dataset['terminals'], dtype=torch.float32).unsqueeze(-1)
    next_states = torch.tensor(dataset['next_observations'], dtype=torch.float32)

    state_dim = states.shape[-1]
    action_dim = actions.shape[-1]

    input_dim = state_dim * 2 + action_dim + 2  # (state, next_state, action, reward, terminal)

    trajectories = []
    for i in range(len(states)):
        trajectory = torch.cat([states[i], actions[i], rewards[i], terminals[i], next_states[i]], dim=-1)
        trajectories.append(trajectory)

    trajectories_padded = pad_sequence(trajectories, batch_first=True, padding_value=0.0)

    return trajectories_padded, state_dim, action_dim




def load_synthetic_dataset(path):
    dpsynther_trajectory = np.load(path)
    
    actions = torch.tensor(dpsynther_trajectory['actions'], dtype=torch.float32)  # (992000, 2)
    observations = torch.tensor(dpsynther_trajectory['observations'], dtype=torch.float32)  # (992000, 4)
    rewards = torch.tensor(dpsynther_trajectory['rewards'], dtype=torch.float32).unsqueeze(-1)  # (992000, 1)
    terminals = torch.tensor(dpsynther_trajectory['terminals'], dtype=torch.float32).unsqueeze(-1)  # (992000, 1)
    next_observations = torch.tensor(dpsynther_trajectory['next_observations'], dtype=torch.float32)  # (992000, 4)

    print(f"Original actions shape: {actions.shape}")
    print(f"Original observations shape: {observations.shape}")
    print(f"Original rewards shape: {rewards.shape}")
    print(f"Original terminals shape: {terminals.shape}")
    print(f"Original next_observations shape: {next_observations.shape}")

    rewards = rewards.view(rewards.shape[0], -1)
    terminals = terminals.view(terminals.shape[0], -1)
    # pdb.set_trace()
    trajectory = torch.cat([observations, actions, rewards, terminals, next_observations], dim=-1)

    print(f"Final trajectory shape: {trajectory.shape}")
    
    return trajectory

def main(args):
    print("Loading datasets...")
    original_trajectory, state_dim, action_dim = load_d4rl_dataset(args.dataset)
    synthetic_trajectory = load_synthetic_dataset(args.load_path)

    print(f"State dimension: {state_dim}, Action dimension: {action_dim}")

    print("Initializing model...")
    model = TrajectoryEmbedding(state_dim, action_dim)

    print("Computing embeddings...")
    # pdb.set_trace()
    print("Computing similarity...")
    similarity_score = model(original_trajectory, synthetic_trajectory)
    print(f"Dataset Similarity Score: {similarity_score.item()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="maze2d-medium-dense-v1", help="D4RL dataset name")
    parser.add_argument("--load_path", type=str, default="/p/fzv6enresearch/liuzheng/MTDiff/results/maze2d-medium-dense-v1/-Mar28_11-43-47/state_200000/sampled_trajectories.npz", help="Path to synthetic dataset")
    # parser.add_argument("--load_path", type=str, default="baselines/samples/maze2d-medium-dense-v1/pgm_10.0/maze2d-medium-dense-v1.npz", help="Path to synthetic dataset")
    args = parser.parse_args()

    main(args)
