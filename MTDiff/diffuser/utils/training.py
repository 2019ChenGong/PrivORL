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
import metaworld
import time
import gym
import d4rl
import statistics
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP


DTYPE = torch.float
from collections import namedtuple
import diffuser.utils as utils
from diffuser.utils.rnd import Rnd
DTBatch = namedtuple('DTBatch', 'actions rtg observations timestep mask')
AugBatch = namedtuple('AugBatch', 'trajectories task cond')
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


class MetaworldTrainer(object):
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
        label_freq=100,#100000,
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
        best_score, prev_label = 0, 0
        for step in range(n_train_steps):
            for i in range(self.gradient_accumulate_every):
                batch = next(self.dataloader)
                batch = batch_to_device(batch,self.device)
                loss, infos = self.model.loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()
            self.writer.add_scalar('Loss', infos['a0_loss'], global_step=self.step)
            self.writer.add_scalar('a0_Loss', loss, global_step=self.step)
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()
            if self.step % self.save_freq == 0:
                label = self.step // self.label_freq * self.label_freq
                if self.step != 0:
                    if self.is_unet:
                        score, success_rate = self.evaluate(self.device)
                    else:
                        score, success_rate = self.evaluate(self.device)
                    label = str(label) + '_' + str(score) + '_' + str(success_rate)
                    if score > best_score:
                        self.save(label)
                        best_score = score
                else:
                    self.save(label)
                self.save(label)


            if self.step % self.log_freq == 0:
                infos_str = ' | '.join([f'{key}: {val:8.5f}' for key, val in infos.items()])
                print(f'{self.step}: {loss:8.5f} | {infos_str} | t: {timer():8.4f}', flush=True)
            self.step += 1

    def evaluate(self, device):
        num_eval = 10
        task = [metaworld.MT1(env).train_tasks[i] for i in range(num_eval) for env in self.envs]
        mt1 = [metaworld.MT1(env) for i in range(num_eval) for env in self.envs]
        env_list = [mt1[i].train_classes[self.envs[i]]() for j in range(num_eval) for i in range(len(self.envs))]
        seed = 0
        for i in range(len(env_list)):
            env_list[i].set_task(task[i])
            env_list[i].seed(seed)
        score = 0
        total_success = 0
        env_success_rate = [0 for i in env_list]
        episode_rewards = [0 for i in env_list]
        max_episode_length = 500
        obs_list = [env.reset()[None] for env in env_list]
        obs = np.concatenate(obs_list, axis=0)
        cond_task = torch.tensor([i for j in range(num_eval) for i in range(len(self.envs))],
                                 device=device).reshape(-1, )
        conditions = torch.zeros([obs.shape[0], 2, obs.shape[-1]], device=device)
        rtg = torch.ones((len(env_list),), device=device) * 9
        for j in range(max_episode_length):
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
                next_observation, reward, terminal, info = env_list[i].step(action[i])
                obs_list.append(next_observation[None])
                episode_rewards[i] += reward
                if info['success'] > 1e-8:
                    env_success_rate[i] = 1
            obs = np.concatenate(obs_list, axis=0)
        for i in range(len(self.envs)):
            tmp = []
            tmp_suc = 0
            for j in range(num_eval):
                tmp.append(episode_rewards[i + j * len(self.envs)])
                tmp_suc += env_success_rate[i + j * len(self.envs)]
            this_score = statistics.mean(tmp)
            success = tmp_suc / num_eval
            total_success += success
            score += this_score
            print(f"task:{self.envs[i]},success rate:{success}, mean episodic return:{this_score}, "
                  f"std:{statistics.stdev(tmp)}")
        print('Total success rate:', total_success / len(self.envs))
        return score, total_success / len(self.envs)
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

    #-----------------------------------------------------------------------------#
    #--------------------------------- rendering ---------------------------------#
    #-----------------------------------------------------------------------------#

    def render_reference(self, batch_size=10):
        '''
            renders training points
        '''

        ## get a temporary dataloader to load a single batch
        dataloader_tmp = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=batch_size, num_workers=0, shuffle=True, pin_memory=True
        ))
        batch = dataloader_tmp.__next__()
        dataloader_tmp.close()

        ## get trajectories and condition at t=0 from batch
        trajectories = to_np(batch.trajectories)
        conditions = to_np(batch.conditions[0])[:,None]

        ## [ batch_size x horizon x observation_dim ]
        normed_observations = trajectories[:, :, self.dataset.action_dim + 1:]
        observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

        savepath = os.path.join(self.logdir, f'_sample-reference.png')
        self.renderer.composite(savepath, observations)

    def render_samples(self, batch_size=2, n_samples=2):
        '''
            renders samples from (ema) diffusion model
        '''
        for i in range(batch_size):

            ## get a single datapoint
            batch = self.dataloader_vis.__next__()
            conditions = to_device(batch.conditions, 'cuda:0')

            ## repeat each item in conditions `n_samples` times
            conditions = apply_dict(
                einops.repeat,
                conditions,
                'b d -> (repeat b) d', repeat=n_samples,
            )

            ## [ n_samples x horizon x (action_dim + observation_dim) ]
            samples = self.ema_model(conditions)
            trajectories = to_np(samples.trajectories)

            ## [ n_samples x horizon x observation_dim ]
            normed_observations = trajectories[:, :, self.dataset.action_dim + 1:]

            # [ 1 x 1 x observation_dim ]
            normed_conditions = to_np(batch.conditions[0])[:,None]

            ## [ n_samples x (horizon + 1) x observation_dim ]
            normed_observations = np.concatenate([
                np.repeat(normed_conditions, n_samples, axis=0),
                normed_observations
            ], axis=1)

            ## [ n_samples x (horizon + 1) x observation_dim ]
            observations = self.dataset.normalizer.unnormalize(normed_observations, 'observations')

            savepath = os.path.join(self.logdir, f'sample-{self.step}-{i}.png')
            self.renderer.composite(savepath, observations)

import json
import os
from tqdm import tqdm  # 导入 tqdm 用于进度条
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import copy

class AugTrainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        renderer,
        sample=False,
        checkpoint_path="",
        # rnd
        curiosity_driven_rate=0.3,

        ema_decay=0.995,
        train_batch_size=32,
        train_lr=2e-5,
        gradient_accumulate_every=2,
        step_start_ema=2000,
        update_ema_every=10,
        log_freq=100,
        sample_freq=1000,
        save_freq=1000,
        label_freq=100,#100000,
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
        max_grad_norm=1.0
    ):
        super().__init__()
        self.original_model = diffusion_model
        self.model = diffusion_model.to(trainer_device)
        self.ema = EMA(ema_decay)
        self.update_ema_every = update_ema_every
        self.envs = envs
        self.device = trainer_device
        self.horizon = horizon
        self.task_list = task_list
        self.is_unet = is_unet

        self.step_start_ema = step_start_ema
        self.log_freq = log_freq
        self.sample_freq = sample_freq
        self.save_freq = save_freq
        self.label_freq = label_freq
        self.save_parallel = save_parallel

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataset = dataset
        # sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=dist.get_rank())
        self.sample = sample
        # print("in training.py, sample is ", self.sample)

        self.raw_dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=1,
            pin_memory=True,
        )
        self.dataloader = cycle(self.raw_dataloader)
        self.dataloader_vis = cycle(torch.utils.data.DataLoader(
            self.dataset, batch_size=1, num_workers=0, shuffle=True, pin_memory=True
        ))

        self.renderer = renderer
        self.train_lr = train_lr
        self.privacy = privacy
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
        self.step = 0
        
        # self.save(self.step)

        # rnd
        if not self.privacy:
            self.curiosity_driven_rate = curiosity_driven_rate
            self.sample_batch_size = train_batch_size
            for name, param in self.model.named_parameters():
                print(f"{name}: requires_grad={param.requires_grad}")
            # self.sample_batch_size = 20

            initial_cond = torch.zeros((1, self.dataset.observation_dim), device=self.device)
            diffusion_samples = self.model.conditional_sample(
                cond=initial_cond,
                task=torch.tensor([0], device=self.device),  
                value=torch.tensor([0], device=self.device),  
                horizon=self.horizon
            )
            self.rnd = Rnd(input_dim=diffusion_samples.shape[2], device=self.device)



    def reset_parameters(self):
        if isinstance(self.model, (DDP, DPDDP)):
            self.ema_model.load_state_dict(self.model.module.state_dict(), strict=False)
        else:
            self.ema_model.load_state_dict(self.model.state_dict(), strict=False)

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model.module if isinstance(self.model, (DDP, DPDDP)) else self.model)

    def train(self, n_train_steps):
        self.model.train()
        timer = Timer()

        total_iterations = n_train_steps
        steps_taken = 0
        # print('n_train_steps is ', n_train_steps, 'raw_dataloader is ', len(self.raw_dataloader))

        if self.privacy:
                progress_bar = tqdm(total=total_iterations, desc="Training (Privacy)")
                
                while steps_taken < n_train_steps:
                    with BatchMemoryManager(
                        data_loader=self.raw_dataloader,
                        max_physical_batch_size=self.batch_size // self.gradient_accumulate_every,
                        optimizer=self.optimizer
                    ) as memory_safe_dataloader:
                        for batch in memory_safe_dataloader:
                            if steps_taken >= n_train_steps:
                                pdb.set_trace()
                                break

                            if self.step % self.save_freq == 0:
                                label = self.step // self.label_freq * self.label_freq
                                self.save(label)

                            batch = batch_to_device(batch, self.device)
                            loss, infos = self.model._module.compute_loss(*batch)

                            loss.backward()
                            self.optimizer.step()
                            self.optimizer.zero_grad()

                            self.writer.add_scalar('Loss', loss.item(), global_step=self.step)

                            if self.step % self.update_ema_every == 0:
                                self.step_ema()

                            if self.step % self.log_freq == 0:
                                infos_str = ' | '.join([f'{key}: {val:8.5f}' for key, val in infos.items()])
                                print(f'{self.step}: {loss:8.5f} | {infos_str} | t: {timer():8.4f}', flush=True)
                                self._log_to_json(self.step, loss.item())

                            self.step += 1
                            steps_taken += 1
                            progress_bar.update(1)

                            if steps_taken >= n_train_steps:
                                break
                progress_bar.close()
                    
        else:
            # rnd
            diffusion_samples = []
            initial_cond = torch.zeros((self.sample_batch_size, self.dataset.observation_dim), device=self.device)
            for i in tqdm(range(self.sample_batch_size), desc="Sampling Curiosity-Driven Samples", unit="sample"):
                sampled_data = self.model.conditional_sample(
                    cond=initial_cond[i:i+1],
                    task=torch.tensor([0], device=self.device),
                    value=torch.tensor([0], device=self.device),
                    horizon=self.horizon
                )  # Shape will be [1, self.horizon, input_dim]
                
                diffusion_samples.append(sampled_data.unsqueeze(0))  # Adding a new batch dimension, resulting in [1, 1, self.horizon, input_dim]

            # Concatenate all the sampled data into a single tensor with shape [self.sample_batch_size, 1, self.horizon, input_dim]
            diffusion_samples = torch.cat(diffusion_samples, dim=0)
            # print("diffusion_samples.shape is ", diffusion_samples.shape)

            sample_loss_list = []
            rnd_loss_total = 0.0
            for sample in diffusion_samples:
                sample_loss = self.rnd.forward(sample)  # Each sample is now of shape [1, self.horizon, input_dim]
                sample_loss_list.append(sample_loss.item())
                rnd_loss_total += sample_loss

            rnd_loss_total = rnd_loss_total / len(diffusion_samples)
            self.rnd.train_step(rnd_loss_total)

            # Calculate the RND loss for each of the samples
            diffusion_samples_loss = torch.tensor(sample_loss_list, device=self.device)

            # Select the top k curiosity-driven samples based on RND loss
            _, selected_idx = torch.topk(diffusion_samples_loss, k=int(self.sample_batch_size * self.curiosity_driven_rate))
            # print(selected_idx)
            selected_samples = diffusion_samples[selected_idx, :, :, :]
            # print("selected_samples.shape is ", selected_samples.shape)
            self.idx = [i for i in range(selected_samples.shape[0])]
            # print("self.idx is ", self.idx)

            progress_bar = tqdm(total=total_iterations, desc=f"Training")
            for i, batch in enumerate(self.dataloader):
                if steps_taken >= n_train_steps:
                    break
            
                if self.step % self.save_freq == 0:
                    label = self.step // self.label_freq * self.label_freq
                    self.save(label)
                    
                batch = batch_to_device(batch, self.device)
                trajectories, task, cond = batch.trajectories, batch.task, batch.cond

                # Concatenate the selected curiosity-driven samples with the current batch
                random_idx = random.sample(self.idx, int(len(self.idx) * self.curiosity_driven_rate))
                # print("random_idx is ", random_idx)
                # print(random_idx)
                random_samples = selected_samples[random_idx, :, :, :]
                # print("batch.shape is ", batch.shape)
                # print("random_samples.shape is ", random_samples.shape)
                # print("batch.trajectories.shape is ", batch.trajectories.shape)
                trajectories = torch.cat((batch.trajectories, random_samples), dim=0)
                task = batch.task
                cond = torch.cat((cond, cond[:len(random_idx)]), dim=0)
                # print("Creating AugBatch with:", trajectories.shape, task.shape, cond.shape)
                batch = AugBatch(trajectories, task, cond)
                
                loss, infos = self.model.compute_loss(*batch)
                loss = loss / self.gradient_accumulate_every
                loss.backward()

                if (i + 1) % self.gradient_accumulate_every == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                self.writer.add_scalar('Loss', loss.item(), global_step=self.step)

                if self.step % self.update_ema_every == 0:
                    self.step_ema()

                if self.step % self.log_freq == 0:
                    infos_str = ' | '.join([f'{key}: {val:8.5f}' for key, val in infos.items()])
                    print(f'{self.step}: {loss:8.5f} | {infos_str} | t: {timer():8.4f}', flush=True)
                    self._log_to_json(self.step, loss.item())

                self.step += 1
                steps_taken += 1
                progress_bar.update(1)
            progress_bar.close()

    def _log_to_json(self, step, loss):
        if self.json_log_path is None:
            return

        log_entry = {"step": step, "loss": float(loss)}
        if os.path.exists(self.json_log_path):
            with open(self.json_log_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            data.append(log_entry)
        else:
            data = [log_entry]

        with open(self.json_log_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)

    def save(self, epoch):
        if self.privacy:
            data = {
                'step': self.step,
                'model': self.model.state_dict(),
                'ema': self.ema_model.state_dict(),
            }
            # data = {
            #     'step': self.step,
            #     'model': self.model.state_dict(),
            #     'ema': self.ema_model.state_dict(),
            # }
        else:
            data = {
                'step': self.step,
                'model': self.model.state_dict(),
                'ema': self.ema_model.state_dict(),
            }

        savepath = os.path.join(self.logdir, f'state_{epoch}.pt')
        torch.save(data, savepath)
        print(f'[ utils/training ] Saved model to {savepath}', flush=True)

    def load_for_sample(self, epoch):
        loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(loadpath, map_location=self.device)
        self.step = data['step']

        if self.privacy:
            model_state_dict = data['model']
            
            new_model_state_dict = {}
            for key, value in model_state_dict.items():
                new_key = key
                # If the key doesn't already start with '_module.', add it
                if not new_key.startswith('_module.'):
                    new_key = '_module.' + new_key
                new_model_state_dict[new_key] = value

            # print("Initialized model state_dict keys:", self.model.state_dict().keys())
            
            # print("Adjusted new_model_state_dict keys:", new_model_state_dict.keys())
            self.model.load_state_dict(new_model_state_dict)

            ema_state_dict = data['ema']
            new_ema_state_dict = {}
            for key, value in ema_state_dict.items():
                new_key = key
                # If the key doesn't already start with '_module.', add it
                if not new_key.startswith('_module.'):
                    new_key = '_module.' + new_key
                new_ema_state_dict[new_key] = value
            
            self.ema_model.load_state_dict(new_ema_state_dict)

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
            
            self.model.load_state_dict(model_state_dict)

            ema_state_dict = data['ema']
            new_ema_state_dict = {}
            for key, value in ema_state_dict.items():
                new_key = key.replace("_module.module.", "_module.")
                new_ema_state_dict[new_key] = value
                
            self.ema_model.load_state_dict(ema_state_dict)
        print("successfully load!!")

    def load_for_finetune(self, checkpoint_path):
        # loadpath = os.path.join(self.logdir, f'state_{epoch}.pt')
        data = torch.load(checkpoint_path, map_location=self.device)
        # self.step = data['step']

        if self.privacy:
            model_state_dict = data['model']
            
            new_model_state_dict = {}
            for key, value in model_state_dict.items():
                new_key = key
                # If the key doesn't already start with '_module.', add it
                if not new_key.startswith('_module.'):
                    new_key = '_module.module.' + new_key
                new_model_state_dict[new_key] = value

            # print("Initialized model state_dict keys:", self.model.state_dict().keys())
            
            # print("Adjusted new_model_state_dict keys:", new_model_state_dict.keys())
            self.model.load_state_dict(model_state_dict)

        # for no dp finetune
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