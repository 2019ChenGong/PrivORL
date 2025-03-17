import os
import torch
import numpy as np
import inspect
import pandas as pd
import pdb
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

import diffuser.utils as utils

# ----------------------------- #
#        Load Model Setup       #
# ----------------------------- #

class Parser(utils.Parser):
    dataset: str = 'maze2d-medium-dense-v1'
    config: str = 'config.locomotion'
    checkpoint_path: str = "logs/maze2d-medium-dense-v1/-Mar15_02-26-09/state_450000.pt"
    output_csv_path: str = "results/maze2d-medium-dense-v1/-Mar16_02-26-09/state_450000/sampled_trajectories.csv"
    # checkpoint_path: str = "logs_horizon16/maze2d-medium-dense-v1/-Mar15_02-28-29/state_450000.pt"
    # output_csv_path: str = "results_horizon16/maze2d-medium-dense-v1/-Mar15_02-28-29/state_450000/sampled_trajectories.csv"
    num_trajectories: int = 1000
    max_length: int = 1000


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()


def load_model(_, checkpoint_path, device):
    map_location = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
    
    # Use the utility function to load the diffusion experiment
    diffusion_experiment = utils.load_diffusion(
        # loadbase=os.path.dirname(checkpoint_path),   # Base directory of the checkpoint
        # dataset=None,                                # Dataset is optional, based on how `utils.load_diffusion` works
        os.path.dirname(checkpoint_path),                    # Full checkpoint path
        epoch=450000,                                  # Load the latest epoch or specify if needed
        device=device,
        seed=None                                    # Optional, if you need deterministic results
    )
    
    # Use the EMA model for higher-quality sampling
    diffusion = diffusion_experiment.ema.to(device)
    diffusion.clip_denoised = True  # Ensure denoising clipping is applied for stable sampling
    
    print(f'EMA Model loaded from {checkpoint_path} on {map_location}')
    
    return diffusion



def sample_complete_trajectory(diffusion, initial_condition, args, max_length, trajectory_index, dataset):
    trajectory = []
    current_state = initial_condition
    terminal = False
    step_count = 0

    state_dim = dataset.observation_dim
    action_dim = dataset.action_dim

    with tqdm(total=max_length, desc=f"Trajectory {trajectory_index} Progress", leave=False) as pbar:
        while not terminal and step_count < max_length:
            condition = torch.tensor(current_state, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(args.device)

            sample = diffusion.module.conditional_sample(
                horizon=args.transition_steps
            )

            sampled_transitions = sample.cpu().numpy().squeeze(0)

            for sampled_transition in sampled_transitions:
                states = sampled_transition[:state_dim]
                actions = sampled_transition[state_dim:state_dim + action_dim]
                reward = sampled_transition[state_dim + action_dim]
                terminal_flag = int(sampled_transition[state_dim + action_dim + 1] >= 0.5)
                next_states = sampled_transition[state_dim + action_dim + 2:]

                trajectory.append([
                    trajectory_index, step_count, *states, *actions, reward, terminal_flag, *next_states
                ])

                if terminal_flag == 1:
                    terminal = True
                    break

                current_state = next_states
                step_count += 1
                pbar.update(1)

    return trajectory



def save_trajectories_to_csv(trajectories, output_csv_path, state_dim, action_dim):
    state_cols = [f'state_{i}' for i in range(state_dim)]
    action_cols = [f'action_{i}' for i in range(action_dim)]
    next_state_cols = [f'next_state_{i}' for i in range(state_dim)]
    columns = ['trajectory_id', 'step_id'] + state_cols + action_cols + ['reward', 'terminal'] + next_state_cols

    df = pd.DataFrame(trajectories, columns=columns)
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    df.to_csv(output_csv_path, index=False)
    print(f'Trajectories saved to {output_csv_path}')


def sample_trajectory(rank, world_size, args):
    setup(rank, world_size)

    dataset_config = utils.Config(
        args.loader,
        savepath=(args.savepath, 'dataset_config.pkl'),
        env=args.dataset,
        replay_dir_list=[],
        task_list=[args.dataset],
        horizon=args.horizon,
        normalizer=args.normalizer,
        preprocess_fns=args.preprocess_fns,
        use_padding=args.use_padding,
        max_path_length=args.max_path_length,
        discount=args.discount,
        termination_penalty=args.termination_penalty,
        normed=args.normed,
        meta_world=False,
        seq_length=5,
        use_d4rl=True,
    )

    dataset = dataset_config()
    
    model_config = utils.Config(
        args.model,
        savepath=(args.savepath, 'model_config.pkl'),
        horizon=args.horizon,
        transition_dim=dataset.observation_dim + 1,
        cond_dim=dataset.observation_dim,
        num_tasks=1,
        device=rank,
        verbose=False,
        action_dim=dataset.action_dim,
    )

    diffusion_config = utils.Config(
        args.diffusion,
        savepath=(args.savepath, 'diffusion_config.pkl'),
        horizon=args.horizon,
        observation_dim=dataset.observation_dim,
        action_dim=dataset.action_dim,
        n_timesteps=args.n_diffusion_steps,
        loss_type=args.loss_type,
        clip_denoised=args.clip_denoised,
        predict_epsilon=args.predict_epsilon,
        action_weight=args.action_weight,
        loss_weights=args.loss_weights,
        loss_discount=args.loss_discount,
        device=rank,
    )

    model = model_config().to(rank)
    diffusion = diffusion_config(model).to(rank)
    diffusion = load_model(diffusion, args.checkpoint_path, rank)
    diffusion = DDP(diffusion, device_ids=[rank])

    trajectories = []
    per_gpu_samples = (args.num_trajectories + world_size - 1) // world_size
    start_idx = rank * per_gpu_samples
    end_idx = min(start_idx + per_gpu_samples, args.num_trajectories)

    for i in tqdm(range(start_idx, end_idx), desc=f"GPU {rank} Sampling"):
        initial_condition = np.random.randn(dataset.observation_dim)
        trajectory = sample_complete_trajectory(diffusion, initial_condition, args, args.max_length, i, dataset)
        trajectories.extend(trajectory)

    gathered_trajectories = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_trajectories, trajectories)

    if rank == 0:
        all_trajectories = [item for sublist in gathered_trajectories for item in sublist]
        save_trajectories_to_csv(all_trajectories, args.output_csv_path, dataset.observation_dim, dataset.action_dim)

    cleanup()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    
    args = Parser().parse_args('diffusion')
    args.transition_steps = 32
    args.horizon = 32
    
    world_size = torch.cuda.device_count()
    
    mp.spawn(
        sample_trajectory,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()