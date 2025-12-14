import math
import os
import copy
import json
import numpy as np
import torch
import einops
import pdb
import random
from .arrays import batch_to_device, to_np, to_device, apply_dict
from .timer import Timer
from tqdm import tqdm
from .cloud import sync_logs
# import metaworld
import time
import gym
import d4rl
import statistics
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from collections import defaultdict
from torch.nn.parallel import DistributedDataParallel as DDP
# from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP

# Create a dummy class for DPDDP to avoid import errors
class DPDDP:
    pass

# Opacus imports commented out - now using manual DP-SGD
# from opacus import PrivacyEngine
# from opacus.validators import ModuleValidator
# from opacus.utils.batch_memory_manager import BatchMemoryManager


DTYPE = torch.float
from collections import namedtuple
import diffuser.utils as utils
from diffuser.utils.rnd import Rnd
DTBatch = namedtuple('DTBatch', 'actions rtg observations timestep mask')
AugBatch = namedtuple('AugBatch', 'trajectories tasks trajectory_ids fragment_indices path_length cond')

DEVICE = 'cuda'
from torch.utils.tensorboard import SummaryWriter
def cycle(dl):
    while True:
        for data in dl:
            yield data

def to_torch(x, dtype=None, device=None):
    dtype = dtype or DTYPE
    device = device or DEVICE
    if type(x) is dict:
        return {k: to_torch(v, dtype, device) for k, v in x.items()}
    elif torch.is_tensor(x):
        return x.to(device).type(dtype)
        # import pdb; pdb.set_trace()
    return torch.tensor(x, dtype=dtype, device=device)

class EMA():
    '''
        empirical moving average
    '''
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new



class AugTrainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        sample=False,
        checkpoint_path="",
        curiosity_rate=0.3,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,  # 现在直接表示 global steps 间隔
        label_freq=100,   # 现在直接表示 global steps 间隔
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        bucket=None,
        envs=[],
        task_list=[],
        is_unet=False,
        trainer_device=None,
        horizon=32,
        privacy=False,
        noise_multiplier=1.0,
        max_grad_norm=1.0,
        max_fragments_per_trajectory=128,
    ):
        super().__init__()
        self.original_model = diffusion_model
        self.model = diffusion_model.to(trainer_device)
        self.ema = EMA(ema_decay)
        self.update_ema_every = update_ema_every
        self.envs = envs
        self.device = trainer_device
        self.horizon = int(horizon)
        self.task_list = task_list
        self.is_unet = is_unet

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq  # 现在直接表示 global steps 间隔
        self.label_freq = label_freq  # 现在直接表示 global steps 间隔
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.max_fragments_per_trajectory = max_fragments_per_trajectory

        self.dataset = dataset
        self.sample = sample
        
        # 计算每个trajectory的平均fragments数
        self.avg_fragments_per_trajectory = self.dataset.total_fragments / self.dataset.n_episodes
        print(f"[INFO] Average fragments per trajectory: {self.avg_fragments_per_trajectory:.2f}")
        print(f"[INFO] Max fragments per trajectory for training: {self.max_fragments_per_trajectory}")

        # 现在dataloader以trajectory为单位
        self.raw_dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
            collate_fn=dataset.collate_fn
        )
        print("len(self.raw_dataloader) is: ", len(self.raw_dataloader))
        self.dataloader = cycle(self.raw_dataloader)
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True,
            collate_fn=dataset.collate_fn
        ))

        self.renderer = renderer
        self.train_lr = train_lr
        self.privacy = privacy
        # 不再乘以sqrt(avg_fragments)，因为现在是trajectory-level DP
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm

        print("self.privacy is : ", self.privacy)
        print("self.sample is : ", self.sample)
        if not self.privacy:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.train_lr)

        self.ema_model = copy.deepcopy(self.model)

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.writer = SummaryWriter(self.logdir)

        self.json_log_path = os.path.join(self.logdir, "training_log.json")

        self.reset_parameters()
        self.step = 0  # 保留原有的step，用于记录总的fragment steps
        self.global_step = 0  # 新增：用于控制save频率的全局步数

        if not self.privacy:
            self.curiosity_rate = curiosity_rate
            self.sample_batch_size = train_batch_size
            for name, param in self.model.named_parameters():
                print(f"{name}: requires_grad={param.requires_grad}")

                # Infer condition dimension from a sample
                # Get a sample subtrajectory to determine condition dimension
                sample_subtraj, sample_cond, _ = self.dataset.get_subtrajectory(0, 0)
                if sample_cond is not None:
                    cond_dim = sample_cond.shape[0]
                else:
                    cond_dim = self.dataset.observation_dim  # fallback

                initial_cond = torch.zeros((1, cond_dim), device=self.device)
                diffusion_samples = self.model.conditional_sample(
                    cond=initial_cond,
                    task=torch.tensor([0], device=self.device),
                    value=torch.tensor([0], device=self.device),
                    horizon=self.horizon
                )
            self.rnd = Rnd(input_dim=diffusion_samples.shape[2], device=self.device)

    def sample_subtrajectories_from_trajectory(self, trajectory_id, num_possible_subtrajs, max_samples):
        """
        从单个trajectory中随机采样连续的subtrajectories（允许重叠）

        Args:
            trajectory_id: trajectory的ID
            num_possible_subtrajs: 该trajectory可以采样的subtrajectory总数
            max_samples: 最大采样数

        Returns:
            sampled_subtrajs: 采样的subtrajectory数据列表
            sampled_conditions: 对应的条件列表
            sampled_lengths: 对应的实际长度列表
            actual_samples: 实际采样数量
        """
        trajectory_info = self.dataset.trajectory_fragments[trajectory_id]
        path_length = trajectory_info['path_length']

        # 确定实际采样数量
        actual_samples = min(max_samples, num_possible_subtrajs)

        # 随机采样起始位置（允许重叠）
        max_start = max(0, path_length - self.horizon)
        if actual_samples == num_possible_subtrajs and path_length >= self.horizon:
            # 如果采样所有可能的subtrajectory，使用所有起始位置
            sampled_starts = list(range(num_possible_subtrajs))
        else:
            # 随机采样起始位置
            sampled_starts = [random.randint(0, max_start) for _ in range(actual_samples)]

        sampled_subtrajs = []
        sampled_conditions = []
        sampled_lengths = []

        for start_idx in sampled_starts:
            subtraj, condition, actual_length = self.dataset.get_subtrajectory(trajectory_id, start_idx)
            sampled_subtrajs.append(subtraj)
            sampled_conditions.append(condition)
            sampled_lengths.append(actual_length)

        return sampled_subtrajs, sampled_conditions, sampled_lengths, actual_samples

    def process_trajectory_batch(self, trajectory_batch):
        """
        处理trajectory batch，采样连续的subtrajectories并组织成训练数据

        Args:
            trajectory_batch: 包含trajectory信息的batch

        Returns:
            all_trajectory_subtrajs: 所有trajectory的subtrajectory数据，按trajectory组织
            total_subtrajs: 总的subtrajectory数量
        """
        trajectory_ids = trajectory_batch.trajectory_ids
        batch_size = len(trajectory_ids)

        # 为每个trajectory采样连续的subtrajectories
        all_trajectory_subtrajs = []

        for i, traj_id in enumerate(trajectory_ids):
            traj_data = trajectory_batch.trajectory_data[i]
            num_possible_subtrajs = traj_data['num_fragments']  # 重命名但含义变了，现在是可能的subtrajectory数

            subtrajs, conditions, lengths, actual_samples = self.sample_subtrajectories_from_trajectory(
                traj_id, num_possible_subtrajs, self.max_fragments_per_trajectory
            )

            all_trajectory_subtrajs.append({
                'trajectory_id': traj_id,
                'subtrajectories': subtrajs,
                'conditions': conditions,
                'lengths': lengths,
                'num_samples': actual_samples
            })

        return all_trajectory_subtrajs

    def reset_parameters(self):
        """重置EMA模型参数"""
        if isinstance(self.model, (DDP, DPDDP)):
            self.ema_model.load_state_dict(self.model.module.state_dict(), strict=False)
        else:
            self.ema_model.load_state_dict(self.model.state_dict(), strict=False)

    def step_ema(self):
        """更新EMA模型"""
        if self.global_step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model.module if isinstance(self.model, (DDP, DPDDP)) else self.model)

    def train(self, n_train_steps):
        self.model.train()
        timer = Timer()

        completed_fragment_steps = 0
        target_fragment_steps = n_train_steps

        if self.privacy:
            return self.finetune(target_fragment_steps, timer)
        else:
            return self.pretrain(target_fragment_steps, timer)

    def pretrain(self, target_fragment_steps, timer):
        """非隐私模式下的per-trajectory训练"""
        print(f"[INFO] Save frequency: every {self.save_freq} global steps (trajectory batches)")
        print(f"[INFO] Label frequency: every {self.label_freq} global steps (trajectory batches)")
        
        # 生成curiosity-driven samples
        diffusion_samples = []
        transition_dim = self.dataset.action_dim + 2 * self.dataset.observation_dim + 2
        initial_cond = torch.zeros((self.sample_batch_size, transition_dim), device=self.device)

        for i in tqdm(range(self.sample_batch_size), desc="Sampling Curiosity-Driven Samples", unit="sample"):
            sampled_data = self.model.conditional_sample(
                cond=initial_cond[i:i+1],
                task=torch.tensor([0], device=self.device),
                value=torch.tensor([0], device=self.device),
                horizon=self.horizon
            )
            
        print("[INFO] Generating curiosity-driven samples...")
        for i in tqdm(range(self.sample_batch_size), desc="Sampling Curiosity-Driven Samples", unit="sample"):
            sampled_data = self.model.conditional_sample(
                cond=initial_cond[i:i+1],
                task=torch.tensor([0], device=self.device),
                value=torch.tensor([0], device=self.device),
                horizon=self.horizon
            )
            sampled_data = sampled_data.squeeze(1)
            diffusion_samples.append(sampled_data)

        diffusion_samples = torch.cat(diffusion_samples, dim=0)

        # 计算RND loss并选择high-curiosity samples
        # 注意：每次调用rnd.forward()都会自动更新prediction network
        sample_loss_list = []
        for sample in diffusion_samples:
            sample_loss = self.rnd(sample)  # forward() now handles training automatically
            sample_loss_list.append(sample_loss.item())

        diffusion_samples_loss = torch.tensor(sample_loss_list, device=self.device)
        _, selected_idx = torch.topk(diffusion_samples_loss, k=int(self.sample_batch_size * self.curiosity_rate))
        selected_samples = diffusion_samples[selected_idx]
        self.idx = list(range(selected_samples.shape[0]))

        # Per-trajectory训练循环（非隐私模式，简化版）
        progress_bar = tqdm(total=target_fragment_steps, desc="Training (Non-Privacy)")
        completed_fragment_steps = 0

        for i, trajectory_batch in enumerate(self.dataloader):
            if completed_fragment_steps >= target_fragment_steps:
                break

            # 直接使用 save_freq 和 label_freq（现在表示 global steps）
            if self.global_step % self.save_freq == 0 and self.global_step > 0:
                label = self.global_step // self.label_freq * self.label_freq
                self.save(self.global_step)

            # 处理trajectory batch
            all_trajectory_subtrajs = self.process_trajectory_batch(trajectory_batch)

            # === 使用与finetune相同的梯度聚合逻辑，但不加DP噪声 ===
            trajectory_gradients = {}
            total_subtrajs = 0

            for traj_data in all_trajectory_subtrajs:
                traj_id = traj_data['trajectory_id']
                subtrajs = traj_data['subtrajectories']
                conditions = traj_data['conditions']
                num_samples = traj_data['num_samples']

                # 累积该trajectory的所有subtrajectory的梯度
                traj_grads = []

                for subtraj_idx in range(num_samples):
                    # 转换为tensor
                    subtraj_tensor = torch.tensor(subtrajs[subtraj_idx], dtype=torch.float32).unsqueeze(0).to(self.device)
                    # Handle None condition (for no-condition version)
                    if conditions[subtraj_idx] is not None:
                        cond_tensor = torch.tensor(conditions[subtraj_idx], dtype=torch.float32).unsqueeze(0).to(self.device)
                    else:
                        cond_tensor = None
                    task_tensor = torch.zeros(1, dtype=torch.long).to(self.device)

                    # 计算loss
                    loss, infos = self.model.compute_loss(subtraj_tensor, task_tensor, cond_tensor)

                    # 计算梯度（不累积，每次独立计算）
                    self.model.zero_grad()
                    loss.backward()

                    # 收集该subtrajectory的梯度
                    subtraj_grad = [p.grad.clone().detach() if p.grad is not None else torch.zeros_like(p)
                                   for p in self.model.parameters()]
                    traj_grads.append(subtraj_grad)

                # 聚合该trajectory的所有subtrajectory梯度（平均）
                aggregated_grad = []
                for param_idx in range(len(traj_grads[0])):
                    param_grads = [traj_grads[i][param_idx] for i in range(num_samples)]
                    avg_grad = torch.stack(param_grads).mean(dim=0)
                    aggregated_grad.append(avg_grad)

                trajectory_gradients[traj_id] = aggregated_grad
                total_subtrajs += num_samples

            # 平均所有trajectory的梯度（不加噪声，这是与finetune的唯一区别）
            num_trajectories = len(trajectory_gradients)
            averaged_gradients = []
            for param_idx in range(len(list(trajectory_gradients.values())[0])):
                param_grads = [list(trajectory_gradients.values())[i][param_idx] for i in range(num_trajectories)]
                avg_grad = torch.stack(param_grads).mean(dim=0)
                averaged_gradients.append(avg_grad)

            # 应用梯度到模型
            self.model.zero_grad()
            for param, grad in zip(self.model.parameters(), averaged_gradients):
                param.grad = grad

            self.optimizer.step()

            # 更新计数
            completed_fragment_steps += total_subtrajs
            self.step += total_subtrajs

            # 注意：现在使用梯度聚合，没有统一的loss值，暂时记录0
            self.writer.add_scalar('Loss', 0.0, global_step=self.global_step)

            if self.global_step % self.log_freq == 0:
                print(f'global_step: {self.global_step} | subtraj_step: {self.step} | trajectories: {num_trajectories} | subtrajs: {total_subtrajs} | completed: {completed_fragment_steps} | t: {timer():8.4f}', flush=True)
                self._log_to_json(self.global_step, 0.0)

            if self.global_step % self.update_ema_every == 0:
                self.step_ema()

            self.global_step += 1
            progress_bar.update(total_subtrajs)

            if completed_fragment_steps >= target_fragment_steps:
                break

        progress_bar.close()

        return completed_fragment_steps

    def finetune(self, target_fragment_steps, timer):
        print(f"[INFO] Save frequency: every {self.save_freq} global steps (trajectory batches)")
        print(f"[INFO] Label frequency: every {self.label_freq} global steps (trajectory batches)")
        print(f"[INFO] Using manual gradient computation with trajectory-level DP-SGD")

        progress_bar = tqdm(total=target_fragment_steps, desc="Training (Privacy)")
        completed_fragment_steps = 0

        # 不使用BatchMemoryManager，手动实现DPSGD
        for trajectory_batch in self.raw_dataloader:
            if completed_fragment_steps >= target_fragment_steps:
                break

            # 直接使用 save_freq 和 label_freq（现在表示 global steps）
            if self.global_step % self.save_freq == 0 and self.global_step > 0:
                label = self.global_step // self.label_freq * self.label_freq
                self.save(self.global_step)

            all_trajectory_subtrajs = self.process_trajectory_batch(trajectory_batch)

            # === 核心修改：per-subtrajectory梯度计算，然后trajectory-level聚合 ===
            trajectory_gradients = {}  # {traj_id: aggregated_gradient}
            total_subtrajs = 0

            for traj_data in all_trajectory_subtrajs:
                traj_id = traj_data['trajectory_id']
                subtrajs = traj_data['subtrajectories']
                conditions = traj_data['conditions']
                num_samples = traj_data['num_samples']

                # 累积该trajectory的所有subtrajectory的梯度
                traj_grads = []

                for subtraj_idx in range(num_samples):
                    # 转换为tensor
                    subtraj_tensor = torch.tensor(subtrajs[subtraj_idx], dtype=torch.float32).unsqueeze(0).to(self.device)
                    # Handle None condition (for no-condition version)
                    if conditions[subtraj_idx] is not None:
                        cond_tensor = torch.tensor(conditions[subtraj_idx], dtype=torch.float32).unsqueeze(0).to(self.device)
                    else:
                        cond_tensor = None
                    task_tensor = torch.zeros(1, dtype=torch.long).to(self.device)

                    # 计算loss
                    loss, infos = self.model._module.compute_loss(subtraj_tensor, task_tensor, cond_tensor)

                    # 计算梯度（不累积，每次独立计算）
                    self.model.zero_grad()
                    loss.backward()

                    # 收集该subtrajectory的梯度
                    subtraj_grad = [p.grad.clone().detach() if p.grad is not None else torch.zeros_like(p)
                                   for p in self.model._module.parameters()]
                    traj_grads.append(subtraj_grad)

                # 聚合该trajectory的所有subtrajectory梯度（平均）
                aggregated_grad = []
                for param_idx in range(len(traj_grads[0])):
                    param_grads = [traj_grads[i][param_idx] for i in range(num_samples)]
                    avg_grad = torch.stack(param_grads).mean(dim=0)
                    aggregated_grad.append(avg_grad)

                trajectory_gradients[traj_id] = aggregated_grad
                total_subtrajs += num_samples

            # === 应用trajectory-level DPSGD: Clipping + Noise ===
            num_trajectories = len(trajectory_gradients)

            # Step 1: Per-trajectory gradient clipping
            clipped_gradients = []
            for traj_id, traj_grad in trajectory_gradients.items():
                # 计算该trajectory梯度的L2 norm
                grad_norm = torch.sqrt(sum([g.norm(2) ** 2 for g in traj_grad]))

                # Clip梯度
                clip_factor = min(1.0, self.max_grad_norm / (grad_norm + 1e-6))
                clipped_grad = [g * clip_factor for g in traj_grad]
                clipped_gradients.append(clipped_grad)

            # Step 2: 平均所有trajectory的clipped gradients
            averaged_gradients = []
            for param_idx in range(len(clipped_gradients[0])):
                param_grads = [clipped_gradients[i][param_idx] for i in range(num_trajectories)]
                avg_grad = torch.stack(param_grads).mean(dim=0)
                averaged_gradients.append(avg_grad)

            # Step 3: 添加高斯噪声
            noisy_gradients = []
            noise_scale = self.noise_multiplier * self.max_grad_norm / num_trajectories
            for grad in averaged_gradients:
                noise = torch.randn_like(grad) * noise_scale
                noisy_grad = grad + noise
                noisy_gradients.append(noisy_grad)

            # Step 4: 应用梯度到模型
            self.model.zero_grad()
            for param, grad in zip(self.model._module.parameters(), noisy_gradients):
                param.grad = grad

            self.optimizer.step()

            # 更新完成的subtrajectory steps数
            completed_fragment_steps += total_subtrajs
            self.step += total_subtrajs

            # 注意：现在使用梯度聚合，没有统一的loss值，暂时记录0
            self.writer.add_scalar('Loss', 0.0, global_step=self.global_step)

            # 记录日志
            if self.global_step % self.log_freq == 0:
                print(f'global_step: {self.global_step} | subtraj_step: {self.step} | trajectories: {num_trajectories} | subtrajs: {total_subtrajs} | completed: {completed_fragment_steps} | t: {timer():8.4f}', flush=True)
                self._log_to_json(self.global_step, 0.0)

            if self.global_step % self.update_ema_every == 0:
                self.step_ema()

            self.global_step += 1
            progress_bar.update(total_subtrajs)

            if completed_fragment_steps >= target_fragment_steps:
                break

        progress_bar.close()

        return completed_fragment_steps

    def _log_to_json(self, step, loss):
        """记录训练日志到JSON文件"""
        if self.json_log_path is None:
            return
        log_entry = {
            "global_step": int(step), 
            "fragment_step": int(self.step), 
            "loss": float(loss)
        }
        if os.path.exists(self.json_log_path):
            with open(self.json_log_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data.append(log_entry)
        else:
            data = [log_entry]
        with open(self.json_log_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    def save(self, epoch):
        """保存模型"""
        if self.privacy:
            data = {
                'step': self.step,
                'global_step': self.global_step,
                'model': self.model.state_dict(),
                'ema': self.ema_model.state_dict(),
            }
        else:
            data = {
                'step': self.step,
                'global_step': self.global_step,
                'model': self.model.state_dict(),
                'ema': self.ema_model.state_dict(),
            }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}', flush=True)

    def load_for_sample(self, epoch):
        """加载模型用于采样"""
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath, map_location=self.device)
        self.step = data['step']
        if 'global_step' in data:
            self.global_step = data['global_step']
        if self.privacy:
            model_state_dict = data['model']
            new_model_state_dict = {}
            for key, value in model_state_dict.items():
                new_key = key
                if not new_key.startswith('_module.'):
                    new_key = '_module.' + new_key
                new_model_state_dict[new_key] = value
            self.model.load_state_dict(new_model_state_dict)
            ema_state_dict = data['ema']
            new_ema_state_dict = {}
            for key, value in ema_state_dict.items():
                new_key = key
                if not new_key.startswith('_module.'):
                    new_key = '_module.' + new_key
                new_ema_state_dict[new_key] = value
            self.ema_model.load_state_dict(new_ema_state_dict)
        else:
            self.model.load_state_dict(data['model'])
            self.ema_model.load_state_dict(data['ema'])
        print("successfully load!!")

    def load_for_finetune(self, checkpoint_path):
        """加载模型用于微调 (支持不同 condition encoding 架构)"""
        data = torch.load(checkpoint_path, map_location=self.device)
        if 'global_step' in data:
            self.global_step = data['global_step']

        model_state_dict = data['model']

        # Check if checkpoint has different condition encoding architecture
        # First, check for 5-token architecture (has cond_state_mlp)
        checkpoint_has_5token = any('cond_state_mlp' in key for key in model_state_dict.keys())

        # Also check if 1-token cond_mlp has different size (old vs new state-only)
        checkpoint_cond_mlp_key = None
        checkpoint_cond_mlp_shape = None
        for key in model_state_dict.keys():
            # Look for cond_mlp weight in checkpoint (with any prefix)
            if 'cond_mlp.0.weight' in key:
                checkpoint_cond_mlp_key = key
                checkpoint_cond_mlp_shape = model_state_dict[key].shape
                break

        # Get the actual model (handle privacy wrapper)
        if self.privacy:
            # In privacy mode, model is wrapped: AugDiffusion -> GradSampleModule -> model
            # Access the actual TasksAug model through: self.model._module (diffusion) -> .model (TasksAug)
            actual_model = self.model._module.model if hasattr(self.model, '_module') else self.model.model
        else:
            # In non-privacy mode, model is AugDiffusion, access .model to get TasksAug
            actual_model = self.model._module.model if hasattr(self.model, '_module') else self.model.model

        current_has_5token = hasattr(actual_model, 'cond_state_mlp')

        # Check if there's a size mismatch for 1-token cond_mlp
        cond_mlp_size_mismatch = False
        if not checkpoint_has_5token and not current_has_5token and checkpoint_cond_mlp_shape is not None:
            # Both use 1-token, check if sizes match
            if hasattr(actual_model, 'cond_mlp'):
                current_cond_mlp_shape = actual_model.cond_mlp[0].weight.shape
                if checkpoint_cond_mlp_shape != current_cond_mlp_shape:
                    cond_mlp_size_mismatch = True
                    print(f"[WARNING] 1-token cond_mlp size mismatch: checkpoint has {checkpoint_cond_mlp_shape}, "
                          f"current model has {current_cond_mlp_shape}")
                    print("[INFO] This is expected when switching from old (transition_dim) to new (state_dim) conditioning")

        if checkpoint_has_5token != current_has_5token or cond_mlp_size_mismatch:
            print(f"[WARNING] Checkpoint uses {'5-token' if checkpoint_has_5token else '1-token'} condition encoding, "
                  f"but current model uses {'5-token' if current_has_5token else '1-token'}.")
            print("[INFO] Loading with strict=False to skip condition MLP parameters")
            strict_loading = False
        else:
            strict_loading = True

        # Handle different wrapper prefixes:
        # Checkpoint might have: "_module.module.*" (DDP + wrapper), "_module.*" (single wrapper), or no prefix
        # Current model: just the base AugDiffusion model, no wrappers (self.model is AugDiffusion directly)

        # Remove all wrapper prefixes from checkpoint
        new_model_state_dict = {}
        for key, value in model_state_dict.items():
            new_key = key
            # Remove any wrapper prefixes
            if new_key.startswith('_module.module.'):
                new_key = new_key.replace('_module.module.', '')
            elif new_key.startswith('_module.'):
                new_key = new_key.replace('_module.', '')
            elif new_key.startswith('module.'):
                new_key = new_key.replace('module.', '')
            new_model_state_dict[new_key] = value

        self.model.load_state_dict(new_model_state_dict, strict=strict_loading)

        # Handle EMA model loading (also remove wrappers)
        if 'ema' in data:
            ema_state_dict = data['ema']
            new_ema_state_dict = {}
            for key, value in ema_state_dict.items():
                new_key = key
                # Remove any wrapper prefixes
                if new_key.startswith('_module.module.'):
                    new_key = new_key.replace('_module.module.', '')
                elif new_key.startswith('_module.'):
                    new_key = new_key.replace('_module.', '')
                elif new_key.startswith('module.'):
                    new_key = new_key.replace('module.', '')
                new_ema_state_dict[new_key] = value
            self.ema_model.load_state_dict(new_ema_state_dict, strict=False)

        if strict_loading:
            print("Successfully loaded checkpoint with matching architecture!")
        else:
            print("Successfully loaded checkpoint (with architecture mismatch - condition MLPs randomly initialized)!")


        

class MazeTrainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100000,
        save_parallel=False,
        results_folder='./results',
        n_reference=8,
        bucket=None,
        envs=[],
        task_list=[],
        is_unet=False,
        trainer_device=None,
        horizon=32,
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.envs = envs
        self.device = trainer_device
        self.horizon = horizon
        self.task_list = task_list
        self.is_unet=is_unet

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        self.dataloader = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=train_batch_size, num_workers=1, shuffle=True, pin_memory=True
        ))
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))
        self.renderer = renderer
        self.optimizer = torch.optim.Adam(diffusion_model.parameters(), lr=train_lr)

        self.logdir = results_folder
        self.bucket = bucket
        self.n_reference = n_reference
        self.writer = SummaryWriter(self.logdir)
        self.reset_parameters()
        self.step = 0
    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        
        self.ema.update_model_average(self.ema_model, self.model)

    #-----------------------------------------------------------------------------#
    #------------------------------------ api ------------------------------------#
    #-----------------------------------------------------------------------------#

    def train(self, n_train_steps):

        timer = Timer()
        best_score = 0
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch, self.device)
                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()
            self.writer.add_scalar('Loss', loss, global_step=self.step)
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()
            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                if self.step != 0:
                    score, success_rate = self.evaluate(self.device)
                    label = str(label) + '_' + str(score) + '_' + str(success_rate)
                    if score > best_score:
                        self.save(label)
                        best_score = score
                else:
                    score, success_rate = self.evaluate(self.device)
                    self.save(label)
                self.save(label)


            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.5f}' for key, val in infos.items()])
                print(f'{self.step}: {loss:8.5f} | {infos_str} | t: {timer():8.4f}', flush=True)
            self.step += 1
    def evaluate(self, device):
        num_eval = 10
        env_list = [gym.make(self.envs[i]) for j in range(num_eval) for i in range(len(self.envs))]
        score = 0
        dones = [False for j in range(num_eval) for i in range(len(self.envs))]
        episode_rewards = [0 for i in env_list]
        max_episode_length = 600
        obs_list = [env.reset()[None] for env in env_list]
        obs = np.concatenate(obs_list, axis=0)
        cond_task = torch.tensor([i for j in range(num_eval) for i in range(len(self.envs))],
                                 device=device).reshape(-1, )
        conditions = torch.zeros([obs.shape[0], 5, obs.shape[-1]], device=device)
        rtg = torch.ones((len(env_list),), device=device) * 0.95
        while False in dones:#for _ in range(max_episode_length):#
            obs = self.dataset.normalizer.normalize(obs, 'observations')
            conditions = torch.cat([conditions[:, 1:, :], to_torch(obs, device=device).unsqueeze(1)], dim=1)
            samples = self.ema_model.conditional_sample(conditions, task=cond_task,
                                                           value=rtg,
                                                           verbose=False, horizon=self.horizon, guidance=1.2)
            action = samples.trajectories[:, 0, :]
            action = action.reshape(-1, self.dataset.action_dim)
            action = to_np(action)
            action = self.dataset.normalizer.unnormalize(action, 'actions')
            obs_list = []
            for i in range(len(env_list)):
                if not dones[i]:
                    next_observation, reward, dones[i], info = env_list[i].step(action[i])
                    obs_list.append(next_observation[None])
                    episode_rewards[i] += reward
                else:
                    obs_list.append(torch.zeros(1, self.dataset.observation_dim))
            obs = np.concatenate(obs_list, axis=0)
        for i in range(len(self.envs)):
            tmp = []
            for j in range(num_eval):
                tmp.append(episode_rewards[i + j * len(self.envs)])
            this_score = statistics.mean(tmp)
            score += this_score
            print(f"task:{self.envs[i]},mean episodic return:{this_score}, "
                  f"std:{statistics.stdev(tmp)}")
        return score, 1.
    def save(self, epoch):
        '''
            saves model and ema to disk;
            syncs to storage bucket if a bucket is specified
        '''
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}', flush=True)
        if self.bucket is not None:
            sync_logs(self.logdir, bucket=self.bucket, background=self.save_parallel)

    def load(self, epoch):
        '''
            loads model and ema from disk
        '''
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath)

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])