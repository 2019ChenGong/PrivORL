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
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager


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
        max_fragments_per_trajectory=32,
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
        self.noise_multiplier = noise_multiplier * math.sqrt(self.avg_fragments_per_trajectory)
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

                transition_dim = self.dataset.action_dim + 2 * self.dataset.observation_dim + 2
                initial_cond = torch.zeros((1, transition_dim), device=self.device)
                diffusion_samples = self.model.conditional_sample(
                    cond=initial_cond,
                    task=torch.tensor([0], device=self.device),
                    value=torch.tensor([0], device=self.device),
                    horizon=self.horizon
                )
            self.rnd = Rnd(input_dim=diffusion_samples.shape[2], device=self.device)

    def sample_fragments_from_trajectory(self, trajectory_id, num_fragments, max_samples):
        """
        从单个trajectory中采样fragments
        
        Args:
            trajectory_id: trajectory的ID
            num_fragments: 该trajectory总的fragment数
            max_samples: 最大采样数
            
        Returns:
            sampled_fragments: 采样的fragment数据列表
            sampled_conditions: 对应的条件列表
            sampled_lengths: 对应的实际长度列表
            actual_samples: 实际采样数量
        """
        # 确定实际采样数量
        actual_samples = min(max_samples, num_fragments)
        
        # 随机采样fragment索引
        if actual_samples == num_fragments:
            # 如果采样数等于总数，直接使用所有fragments
            sampled_indices = list(range(num_fragments))
        else:
            # 随机采样
            sampled_indices = random.sample(range(num_fragments), actual_samples)
        
        sampled_fragments = []
        sampled_conditions = []
        sampled_lengths = []
        
        for local_fragment_idx in sampled_indices:
            fragment, condition, actual_length = self.dataset.get_fragment(trajectory_id, local_fragment_idx)
            sampled_fragments.append(fragment)
            sampled_conditions.append(condition)
            sampled_lengths.append(actual_length)
        
        return sampled_fragments, sampled_conditions, sampled_lengths, actual_samples

    def process_trajectory_batch(self, trajectory_batch):
        """
        处理trajectory batch，采样fragments并组织成训练数据
        
        Args:
            trajectory_batch: 包含trajectory信息的batch
            
        Returns:
            processed_steps: 每一步的训练数据列表
            max_steps: 需要的最大步数
        """
        trajectory_ids = trajectory_batch.trajectory_ids
        batch_size = len(trajectory_ids)
        
        # 为每个trajectory采样fragments
        all_trajectory_fragments = []
        max_sampled_fragments = 0
        
        for i, traj_id in enumerate(trajectory_ids):
            traj_data = trajectory_batch.trajectory_data[i]
            num_fragments = traj_data['num_fragments']
            
            fragments, conditions, lengths, actual_samples = self.sample_fragments_from_trajectory(
                traj_id, num_fragments, self.max_fragments_per_trajectory
            )
            
            all_trajectory_fragments.append({
                'trajectory_id': traj_id,
                'fragments': fragments,
                'conditions': conditions,
                'lengths': lengths,
                'num_samples': actual_samples
            })
            
            max_sampled_fragments = max(max_sampled_fragments, actual_samples)
        
        # 组织成步骤数据：每步包含batch_size个fragments，每个来自不同trajectory
        processed_steps = []
        
        for step_idx in range(max_sampled_fragments):
            step_fragments = []
            step_conditions = []
            step_lengths = []
            step_trajectory_ids = []
            
            for traj_data in all_trajectory_fragments:
                if step_idx < traj_data['num_samples']:
                    # 该trajectory在此步骤有fragment
                    step_fragments.append(traj_data['fragments'][step_idx])
                    step_conditions.append(traj_data['conditions'][step_idx])
                    step_lengths.append(traj_data['lengths'][step_idx])
                    step_trajectory_ids.append(traj_data['trajectory_id'])
                else:
                    # 该trajectory在此步骤没有fragment，随机选择一个已有的fragment
                    random_idx = random.randint(0, traj_data['num_samples'] - 1)
                    step_fragments.append(traj_data['fragments'][random_idx])
                    step_conditions.append(traj_data['conditions'][random_idx])
                    step_lengths.append(traj_data['lengths'][random_idx])
                    step_trajectory_ids.append(traj_data['trajectory_id'])
            
            # 转换为tensor
            step_fragments_tensor = torch.tensor(np.array(step_fragments), dtype=torch.float32)
            step_conditions_tensor = torch.tensor(np.array(step_conditions), dtype=torch.float32)
            step_lengths_tensor = torch.tensor(step_lengths, dtype=torch.long)
            step_trajectory_ids_tensor = torch.tensor(step_trajectory_ids, dtype=torch.long)
            
            processed_steps.append({
                'fragments': step_fragments_tensor,
                'conditions': step_conditions_tensor,
                'lengths': step_lengths_tensor,
                'trajectory_ids': step_trajectory_ids_tensor,
                'tasks': torch.zeros(batch_size, dtype=torch.long) 
            })
        
        return processed_steps, max_sampled_fragments

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
        sample_loss_list = []
        rnd_loss_total = 0.0
        for sample in diffusion_samples:
            sample_loss = self.rnd.forward(sample)
            sample_loss_list.append(sample_loss.item())
            rnd_loss_total += sample_loss

        rnd_loss_total = rnd_loss_total / len(diffusion_samples)
        self.rnd.train_step(rnd_loss_total)

        diffusion_samples_loss = torch.tensor(sample_loss_list, device=self.device)
        _, selected_idx = torch.topk(diffusion_samples_loss, k=int(self.sample_batch_size * self.curiosity_rate))
        selected_samples = diffusion_samples[selected_idx]
        self.idx = list(range(selected_samples.shape[0]))

        # Per-trajectory训练循环
        progress_bar = tqdm(total=target_fragment_steps, desc="Training (Per-Trajectory)")
        completed_fragment_steps = 0
        
        for i, trajectory_batch in enumerate(self.dataloader):
            if completed_fragment_steps >= target_fragment_steps:
                break
        
            # 直接使用 save_freq 和 label_freq（现在表示 global steps）
            if self.global_step % self.save_freq == 0 and self.global_step > 0:
                label = self.global_step // self.label_freq * self.label_freq
                # 用self.step作为保存的文件名（总的fragment步数）
                self.save(self.step)
            
            # 处理trajectory batch，得到多个训练步骤
            processed_steps, num_steps = self.process_trajectory_batch(trajectory_batch)
            
            # === 关键修改：使用trajectory-level loss聚合 ===
            trajectory_losses = defaultdict(list)
            
            # 对每个步骤计算loss
            for step_data in processed_steps:
                step_data = {k: v.to(self.device) for k, v in step_data.items()}
                
                trajectories = step_data['fragments']
                conditions = step_data['conditions']
                tasks = step_data['tasks']
                trajectory_ids = step_data['trajectory_ids']

                # 添加curiosity-driven augmentation
                if len(self.idx) > 0:
                    batch_size = len(trajectories)
                    num_augment = int(batch_size * self.curiosity_rate)
                    if num_augment > 0:
                        random_idx = random.sample(self.idx, min(len(self.idx), num_augment))
                        random_samples = selected_samples[random_idx]
                        
                        # 扩展到与batch_size匹配
                        if len(random_samples) < batch_size:
                            # 重复样本以匹配batch_size
                            repeat_times = (batch_size + len(random_samples) - 1) // len(random_samples)
                            random_samples = random_samples.repeat(repeat_times, 1, 1)[:batch_size]
                        
                        trajectories_aug = torch.cat((trajectories, random_samples[:batch_size]), dim=0)
                        cond_aug = torch.cat((conditions, conditions[:batch_size]), dim=0)
                        tasks_aug = torch.cat((tasks, tasks[:batch_size]), dim=0)
                        
                        trajectory_ids_aug = torch.cat((trajectory_ids, trajectory_ids), dim=0)
                    else:
                        trajectories_aug = trajectories
                        cond_aug = conditions
                        tasks_aug = tasks
                        trajectory_ids_aug = trajectory_ids
                else:
                    trajectories_aug = trajectories
                    cond_aug = conditions
                    tasks_aug = tasks
                    trajectory_ids_aug = trajectory_ids
                
                loss, infos = self.model.compute_loss(trajectories_aug, tasks_aug, cond_aug)
                
                for i, traj_id in enumerate(trajectory_ids_aug):
                    traj_id_item = traj_id.item()
                    trajectory_losses[traj_id_item].append(loss / len(trajectory_ids_aug))
            
            total_loss = None
            num_trajectories = len(trajectory_losses)
            
            for traj_id, losses in trajectory_losses.items():
                traj_loss = sum(losses) / len(losses) 
                
                if total_loss is None:
                    total_loss = traj_loss / num_trajectories
                else:
                    total_loss = total_loss + traj_loss / num_trajectories
            
            total_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            completed_fragment_steps += num_steps
            self.step += num_steps  # 更新总的fragment步数
            
            self.writer.add_scalar('Loss', total_loss.item(), global_step=self.global_step)
            
            if self.global_step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.5f}' for key, val in infos.items()])
                print(f'global_step: {self.global_step} | fragment_step: {self.step} | loss: {total_loss.item():8.5f} | {infos_str} | trajectories: {num_trajectories} | steps_per_traj: {num_steps} | completed_fragments: {completed_fragment_steps} | t: {timer():8.4f}', flush=True)
                self._log_to_json(self.global_step, total_loss.item())

            if self.global_step % self.update_ema_every == 0:
                self.step_ema()

            self.global_step += 1  # 每个trajectory batch处理后全局步数加1
            progress_bar.update(num_steps)

            if completed_fragment_steps >= target_fragment_steps:
                break
                
        progress_bar.close()
        
        return completed_fragment_steps

    def finetune(self, target_fragment_steps, timer):
        print(f"[INFO] Save frequency: every {self.save_freq} global steps (trajectory batches)")
        print(f"[INFO] Label frequency: every {self.label_freq} global steps (trajectory batches)")
        
        progress_bar = tqdm(total=target_fragment_steps, desc="Training (Privacy)")
        completed_fragment_steps = 0
        
        while completed_fragment_steps < target_fragment_steps:
            with BatchMemoryManager(
                data_loader=self.raw_dataloader,
                max_physical_batch_size=self.batch_size // self.gradient_accumulate_every,
                optimizer=self.optimizer
            ) as memory_safe_dataloader:
                for trajectory_batch in memory_safe_dataloader:
                    if completed_fragment_steps >= target_fragment_steps:
                        break

                    # 直接使用 save_freq 和 label_freq（现在表示 global steps）
                    if self.global_step % self.save_freq == 0 and self.global_step > 0:
                        label = self.global_step // self.label_freq * self.label_freq
                        # 用self.step作为保存的文件名（总的fragment步数）
                        self.save(self.step)

                    processed_steps, num_steps = self.process_trajectory_batch(trajectory_batch)
                    
                    trajectory_losses = defaultdict(list)
                    
                    for step_data in processed_steps:
                        step_data = {k: v.to(self.device) for k, v in step_data.items()}
                        
                        trajectories = step_data['fragments']
                        conditions = step_data['conditions']
                        tasks = step_data['tasks']
                        trajectory_ids = step_data['trajectory_ids']

                        loss, infos = self.model._module.compute_loss(trajectories, tasks, conditions)
                        
                        for i, traj_id in enumerate(trajectory_ids):
                            traj_id_item = traj_id.item()
                            trajectory_losses[traj_id_item].append(loss / len(trajectory_ids))
                    
                    # 计算每个trajectory的平均loss，然后计算总的平均loss
                    total_loss = None
                    num_trajectories = len(trajectory_losses)
                    
                    for traj_id, losses in trajectory_losses.items():
                        traj_loss = sum(losses) / len(losses)  # 该trajectory的平均loss
                        
                        if total_loss is None:
                            total_loss = traj_loss / num_trajectories
                        else:
                            total_loss = total_loss + traj_loss / num_trajectories
                    
                    # Backward and optimize (一次gradient update)
                    total_loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    # 更新完成的fragment steps数
                    completed_fragment_steps += num_steps
                    self.step += num_steps  # 更新总的fragment步数
                    
                    # 记录日志
                    self.writer.add_scalar('Loss', total_loss.item(), global_step=self.global_step)
                    
                    if self.global_step % self.log_freq == 0:
                        infos_str = ' | '.join([f'{key}: {val:8.5f}' for key, val in infos.items()])
                        print(f'global_step: {self.global_step} | fragment_step: {self.step} | loss: {total_loss.item():8.5f} | {infos_str} | trajectories: {num_trajectories} | steps_per_traj: {num_steps} | completed_fragments: {completed_fragment_steps} | t: {timer():8.4f}', flush=True)
                        self._log_to_json(self.global_step, total_loss.item())

                    if self.global_step % self.update_ema_every == 0:
                        self.step_ema()

                    self.global_step += 1  # 每个trajectory batch处理后全局步数加1
                    progress_bar.update(num_steps)

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
        """加载模型用于微调"""
        data = torch.load(checkpoint_path, map_location=self.device)
        if 'global_step' in data:
            self.global_step = data['global_step']
        if self.privacy:
            model_state_dict = data['model']
            new_model_state_dict = {}
            for key, value in model_state_dict.items():
                new_key = key
                if not new_key.startswith('_module.'):
                    new_key = '_module.module.' + new_key
                new_model_state_dict[new_key] = value
            self.model.load_state_dict(model_state_dict)
        else:
            model_state_dict = data['model']
            new_model_state_dict = {}
            for key, value in model_state_dict.items():
                new_key = key
                if new_key.startswith("_module.module."):
                    new_key = new_key.replace("_module.module.", "_module.")
                elif new_key.startswith("module."):
                    new_key = new_key.replace("module.", "_module.")
                new_model_state_dict[new_key] = value
            self.model.load_state_dict(new_model_state_dict)
            ema_state_dict = data['ema']
            new_ema_state_dict = {}
            for key, value in ema_state_dict.items():
                new_key = key.replace("_module.module.", "_module.")
                new_ema_state_dict[new_key] = value
            self.ema_model.load_state_dict(new_ema_state_dict, strict=False)
        print("successfully load!!")


        

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