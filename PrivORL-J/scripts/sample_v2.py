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
    dataset: str = 'halfcheetah-medium-replay-v2'
    config: str = 'config.locomotion'
    # sample_checkpoint_path: str = "logs/halfcheetah-medium-replay-v2/pretrain/horizon32/state_500000.pt"
    # output_path: str = "logs/halfcheetah-medium-replay-v2/pretrain/horizon32/state_500000/sampled_trajectories.npz"
    sample_checkpoint_path: str = "logs/halfcheetah-medium-replay-v2/finetune/epsilon5_horizon32/state_200000.pt"
    output_path: str = "logs/halfcheetah-medium-replay-v2/finetune/epsilon5_horizon32/state_200000/sampled_trajectories.npz"
    num_trajectories: int = 1000
    max_length: int = 1000
    sample_batch_size: int = 8  # 批量采样的batch size


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()


def load_model(_, device, args):
    map_location = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
    
    # Use the utility function to load the diffusion experiment
    if not args.finetune:
        diffusion_experiment = utils.load_diffusion(
            # loadbase=os.path.dirname(checkpoint_path),   # Base directory of the checkpoint
            # dataset=None,                                # Dataset is optional, based on how `utils.load_diffusion` works
            os.path.dirname(args.sample_checkpoint_path),                    # Full checkpoint path
            epoch='final',                                  # Load the latest epoch or specify if needed
            # epoch=200000,                                  # Load the latest epoch or specify if needed
            device=device,
            seed=None,                                    # Optional, if you need deterministic results
            sample=True,
            args=args
        )
    
    else:
        diffusion_experiment = utils.load_diffusion(
            # loadbase=os.path.dirname(checkpoint_path),   # Base directory of the checkpoint
            # dataset=None,                                # Dataset is optional, based on how `utils.load_diffusion` works
            os.path.dirname(args.sample_checkpoint_path),                    # Full checkpoint path
            # epoch=500000,                                  # Load the latest epoch or specify if needed
            epoch='final',                                  # Load the latest epoch or specify if needed
            device=device,
            seed=1,                                    # Optional, if you need deterministic results
            sample=True,
            args=args
        )
    
    # Use the EMA model for higher-quality sampling
    diffusion = diffusion_experiment.ema.to(device)
    diffusion.clip_denoised = True  # Ensure denoising clipping is applied for stable sampling
    
    print(f'EMA Model loaded from {args.sample_checkpoint_path} on {map_location}')
    
    return diffusion



def sample_complete_trajectory(diffusion, initial_condition, args, max_length, trajectory_index, dataset):
    """单个轨迹采样 - 保留用于兼容性"""
    state_dim = dataset.observation_dim
    action_dim = dataset.action_dim
    trajectory = []

    prev_transition = torch.zeros((1, state_dim * 2 + action_dim + 2), dtype=torch.float32).to(args.device)

    current_state = torch.tensor(initial_condition, dtype=torch.float32).unsqueeze(0).to(args.device)

    for step_count in range(max_length // args.horizon):
        conditions = prev_transition  # 形状 [1, cond_dim]

        if args.privacy:
            samples = diffusion.module._module.conditional_sample(
                cond=conditions,
                task=torch.tensor([0], device=args.device),
                value=torch.tensor([0], device=args.device),
                horizon=args.horizon
            )
        else:
            samples = diffusion.module.conditional_sample(
                cond=conditions,
                task=torch.tensor([0], device=args.device),
                value=torch.tensor([0], device=args.device),
                horizon=args.horizon
            )

        # 处理返回值：可能是Sample对象或直接是tensor
        if isinstance(samples, torch.Tensor):
            sampled_transitions = samples.cpu().numpy().squeeze(0)
        else:
            sampled_transitions = samples.trajectories.cpu().numpy().squeeze(0)

        for i, sampled_transition in enumerate(sampled_transitions):
            states = sampled_transition[:state_dim]
            actions = sampled_transition[state_dim:state_dim + action_dim]
            reward = sampled_transition[state_dim + action_dim]
            terminal_flag = int(sampled_transition[state_dim + action_dim + 1] >= 0.5)
            next_states = sampled_transition[state_dim + action_dim + 2:]

            trajectory.append([
                trajectory_index,
                step_count * args.horizon + i,
                *states, *actions, reward, terminal_flag, *next_states
            ])

            prev_transition = torch.tensor(
                np.concatenate([
                    states, actions, [reward], [terminal_flag], next_states
                ]),
                dtype=torch.float32
            ).unsqueeze(0).to(args.device)

            if terminal_flag == 1:
                return trajectory

        current_state = torch.tensor(next_states, dtype=torch.float32).unsqueeze(0).to(args.device)

    return trajectory


def sample_trajectories_batch(diffusion, initial_conditions_batch, args, max_length, start_indices, dataset):
    """批量采样多个轨迹

    Args:
        diffusion: 扩散模型
        initial_conditions_batch: 初始条件列表 [batch_size, state_dim]
        args: 参数
        max_length: 最大轨迹长度
        start_indices: 轨迹起始索引列表
        dataset: 数据集

    Returns:
        trajectories: 列表的列表，每个元素是一个轨迹
    """
    batch_size = len(initial_conditions_batch)
    state_dim = dataset.observation_dim
    action_dim = dataset.action_dim

    # 为每个轨迹初始化
    trajectories = [[] for _ in range(batch_size)]
    active_mask = torch.ones(batch_size, dtype=torch.bool, device=args.device)  # 标记哪些轨迹还在采样
    step_counts = torch.zeros(batch_size, dtype=torch.long, device=args.device)  # 每个轨迹的步数

    # 初始化prev_transitions
    prev_transitions = torch.zeros((batch_size, state_dim * 2 + action_dim + 2), dtype=torch.float32, device=args.device)

    max_steps = max_length // args.horizon

    for step in range(max_steps):
        if not active_mask.any():
            break

        # 只对还未结束的轨迹采样
        active_indices = torch.where(active_mask)[0]
        active_batch_size = len(active_indices)

        if active_batch_size == 0:
            break

        # 获取active trajectories的条件
        conditions = prev_transitions[active_indices]  # [active_batch_size, cond_dim]

        # 批量采样
        if args.privacy:
            samples = diffusion.module._module.conditional_sample(
                cond=conditions,
                task=torch.zeros(active_batch_size, dtype=torch.long, device=args.device),
                value=torch.zeros(active_batch_size, dtype=torch.long, device=args.device),
                horizon=args.horizon
            )
        else:
            samples = diffusion.module.conditional_sample(
                cond=conditions,
                horizon=args.horizon
            )

        # 处理返回值：可能是Sample对象或直接是tensor
        # shape: [active_batch_size, horizon, transition_dim]
        if isinstance(samples, torch.Tensor):
            sampled_transitions = samples.cpu().numpy()
        else:
            sampled_transitions = samples.trajectories.cpu().numpy()

        # 处理每个active trajectory
        for batch_idx, global_idx in enumerate(active_indices):
            global_idx_item = global_idx.item()

            if not active_mask[global_idx_item]:
                continue

            current_step_count = step_counts[global_idx_item].item()

            # 处理这个trajectory的所有transitions
            for i in range(args.horizon):
                sampled_transition = sampled_transitions[batch_idx, i]

                states = sampled_transition[:state_dim]
                actions = sampled_transition[state_dim:state_dim + action_dim]
                reward = sampled_transition[state_dim + action_dim]
                terminal_flag = int(sampled_transition[state_dim + action_dim + 1] >= 0.5)
                next_states = sampled_transition[state_dim + action_dim + 2:]

                trajectories[global_idx_item].append([
                    start_indices[global_idx_item],
                    current_step_count * args.horizon + i,
                    *states, *actions, reward, terminal_flag, *next_states
                ])

                # 更新prev_transition
                prev_transitions[global_idx_item] = torch.tensor(
                    np.concatenate([states, actions, [reward], [terminal_flag], next_states]),
                    dtype=torch.float32,
                    device=args.device
                )

                # 如果遇到终止状态，标记为inactive
                if terminal_flag == 1:
                    active_mask[global_idx_item] = False
                    break

            # 更新步数
            step_counts[global_idx_item] += 1

    return trajectories



def save_trajectories_to_npz(trajectories, output_npz_path, state_dim, action_dim):
    """
    Save trajectories directly to NPZ format compatible with D4RL/buffer.py

    The buffer.py expects:
    - observations: [N, state_dim]
    - actions: [N, action_dim]
    - rewards: [N,] (1D array, buffer.py will add dimension with [..., None])
    - terminals: [N,] (1D array, buffer.py will add dimension with [..., None])
    - next_observations: [N, state_dim]
    """
    # Convert list of trajectories to numpy array
    trajectories = np.array(trajectories)

    # Extract columns: [trajectory_id, step_id, state_0...state_n, action_0...action_m, reward, terminal, next_state_0...next_state_n]
    # Skip trajectory_id (col 0) and step_id (col 1)
    col_idx = 2

    observations = trajectories[:, col_idx:col_idx + state_dim]
    col_idx += state_dim

    actions = trajectories[:, col_idx:col_idx + action_dim]
    col_idx += action_dim

    # IMPORTANT: Keep rewards and terminals as 1D arrays
    # buffer.py will add the extra dimension with [..., None]
    rewards = trajectories[:, col_idx]  # Shape: [N,]
    col_idx += 1

    terminals = trajectories[:, col_idx]  # Shape: [N,]
    col_idx += 1

    next_observations = trajectories[:, col_idx:col_idx + state_dim]

    # Create the data dictionary
    data = {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "next_observations": next_observations
    }

    # Save to NPZ file
    os.makedirs(os.path.dirname(output_npz_path), exist_ok=True)
    np.savez(output_npz_path, **data)
    print(f'Trajectories saved to {output_npz_path}')
    print(f'Data shapes:')
    print(f'  observations: {observations.shape}')
    print(f'  actions: {actions.shape}')
    print(f'  rewards: {rewards.shape}')
    print(f'  terminals: {terminals.shape}')
    print(f'  next_observations: {next_observations.shape}')


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
        state_dim=dataset.observation_dim,
        transition_dim=dataset.observation_dim * 2 + dataset.action_dim + 2,
        cond_dim=dataset.observation_dim * 2 + dataset.action_dim + 2,
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
    diffusion = load_model(diffusion, rank, args)
    diffusion = DDP(diffusion, device_ids=[rank])

    trajectories = []
    per_gpu_samples = (args.num_trajectories + world_size - 1) // world_size
    start_idx = rank * per_gpu_samples
    end_idx = min(start_idx + per_gpu_samples, args.num_trajectories)

    # 使用批量采样
    sample_batch_size = args.sample_batch_size
    num_batches = (end_idx - start_idx + sample_batch_size - 1) // sample_batch_size

    for batch_idx in tqdm(range(num_batches), desc=f"GPU {rank} Sampling"):
        batch_start = start_idx + batch_idx * sample_batch_size
        batch_end = min(batch_start + sample_batch_size, end_idx)
        actual_batch_size = batch_end - batch_start

        # 生成一批初始条件
        initial_conditions_batch = [
            np.random.randn(dataset.observation_dim)
            for _ in range(actual_batch_size)
        ]
        start_indices = list(range(batch_start, batch_end))

        # 批量采样
        batch_trajectories = sample_trajectories_batch(
            diffusion, initial_conditions_batch, args, args.max_length, start_indices, dataset
        )

        # 展开所有轨迹
        for traj in batch_trajectories:
            trajectories.extend(traj)

    gathered_trajectories = [None for _ in range(world_size)]
    dist.all_gather_object(gathered_trajectories, trajectories)

    if rank == 0:
        all_trajectories = [item for sublist in gathered_trajectories for item in sublist]
        save_trajectories_to_npz(all_trajectories, args.output_path, dataset.observation_dim, dataset.action_dim)

    cleanup()


def main():
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    
    args = Parser().parse_args('diffusion')
    args.horizon = 32

    if args.finetune:
        args.privacy = True
    else:
        args.privacy = False

    world_size = torch.cuda.device_count()
    
    mp.spawn(
        sample_trajectory,
        args=(world_size, args),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()