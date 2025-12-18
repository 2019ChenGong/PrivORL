from collections import namedtuple
import numpy as np
import random
import torch
import pdb
import diffuser.utils as utils
from .preprocessing import get_preprocess_fn
from .d4rl import load_environment, sequence_dataset, load_dmc_dataset, load_metaworld_dataset, load_antmaze_dataset, load_maze2d_dataset
from .normalization import DatasetNormalizer
from .buffer import ReplayBuffer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from numpy.random import randn
import matplotlib as mpl
from scipy import stats
import os
import gym
import d4rl
import d4rl.gym_mujoco
from torch.utils.data import Sampler

Batch = namedtuple('Batch', 'trajectories conditions')
AugBatch = namedtuple('AugBatch', 'trajectories tasks trajectory_ids fragment_indices path_length cond')
TaskBatch = namedtuple('TaskBatch', 'trajectories conditions task value')
DTBatch = namedtuple('DTBatch', 'actions rtg observations timestep mask')
DT1Batch = namedtuple('DT1Batch', 'actions rtg observations timestep mask task')
PromptDTBatch = namedtuple('PromptDTBatch', 'DTBatch actions rtg observations timestep mask')
ValueBatch = namedtuple('ValueBatch', 'trajectories conditions values')
MTValueBatch = namedtuple('MTValueBatch', 'trajectories conditions task values')
TrajectoryBatch = namedtuple('TrajectoryBatch', 'trajectory_data trajectory_ids max_fragments_per_traj')

class SequenceDataset(torch.utils.data.Dataset):

    def __init__(self, env='hopper-medium-replay', horizon=64,
        normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
        max_n_episodes=10000, termination_penalty=0, use_padding=True, seed=None):
        self.preprocess_fn = get_preprocess_fn(preprocess_fns, env)
        self.env = env = load_environment(env)
        self.env.seed(seed)
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        itr = sequence_dataset(env, self.preprocess_fn)

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for i, episode in enumerate(itr):
            fields.add_path(episode)
            #print("episode:",episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.indices = self.make_indices(fields.path_lengths, horizon)

        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        print(fields)
        # shapes = {key: val.shape for key, val in self.fields.items()}
        # print(f'[ datasets/mujoco ] Dataset fields: {shapes}')

    def normalize(self, keys=['observations', 'actions', 'rewards']):
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def make_indices(self, path_lengths, horizon):
        '''
            makes indices for sampling from dataset;
            each index maps to a datapoint
        '''
        indices = []
        for i, path_length in enumerate(path_lengths):
            max_start = min(path_length - 1, self.max_path_length - horizon)
            if not self.use_padding:
                max_start = min(max_start, path_length - horizon)
            for start in range(max_start):
                end = start + horizon
                indices.append((i, start, end))
        indices = np.array(indices)
        return indices

    def get_conditions(self, observations):
        '''
            condition on current observation for planning
        '''
        return {0: observations[0]}

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]

        observations = self.fields.normed_observations[path_ind, start:end]
        actions = self.fields.normed_actions[path_ind, start:end]
        rewards = self.fields.normed_rewards[path_ind, start:end]
        conditions = self.get_conditions(observations)
        trajectories = np.concatenate([rewards, actions, observations], axis=-1)
        batch = Batch(trajectories, conditions)
        return batch
'''load offline DMC dataset'''
# class MetaSequenceDataset(torch.utils.data.Dataset):

#     def __init__(self, replay_dir_list=[], task_list=[], horizon=64,
#         normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
#         max_n_episodes=200000, termination_penalty=0, use_padding=True, seed=None, meta_world=False, maze2d=False, antmaze=False, optimal=True):
#         self.replay_dir_list = replay_dir_list
#         self.task_list = task_list
#         self.reward_scale = 400.0
#         self.horizon = horizon
#         self.max_path_length = max_path_length
#         self.use_padding = use_padding
#         self.record_values = []
#         if meta_world:
#             itr = load_metaworld_dataset(self.replay_dir_list, self.task_list, optimal=optimal)
#         elif maze2d:
#             itr = load_maze2d_dataset(self.task_list)
#         elif antmaze:
#             itr = load_antmaze_dataset(self.task_list)
#         else:
#             itr = load_dmc_dataset(self.replay_dir_list, self.task_list)

#         fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
#         for i, episode in enumerate(itr):
#             if len(episode['rewards']) > self.max_path_length:
#                 # episode = {k: episode[k][:max_path_length] for k in episode.keys()}
#                 continue
#             self.record_values.append(episode['rewards'].sum())
#             fields.add_path(episode)
#         fields.finalize()

#         self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
#         self.indices = self.make_indices(fields.path_lengths, horizon)

#         self.observation_dim = fields.observations.shape[-1]
#         self.action_dim = fields.actions.shape[-1]
#         self.fields = fields
#         self.n_episodes = fields.n_episodes
#         self.path_lengths = fields.path_lengths
#         self.normalize()

#         print(fields)
#     def normalize(self, keys=['observations', 'actions', 'rewards']):
#         '''
#             normalize fields that will be predicted by the diffusion model
#         '''
#         for key in keys:
#             array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
#             normed = self.normalizer(array, key)
#             self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

#     def make_indices(self, path_lengths, horizon):
#         '''
#             makes indices for sampling from dataset;
#             each index maps to a datapoint
#         '''
#         indices = []
#         for i, path_length in enumerate(path_lengths):
#             max_start = min(path_length - 1, self.max_path_length - horizon)
#             if not self.use_padding:
#                 max_start = min(max_start, path_length - horizon)
#             for start in range(max_start):
#                 end = start + horizon
#                 indices.append((i, start, end))
#         indices = np.array(indices)
#         return indices

#     def get_conditions(self, observations):
#         '''
#             condition on current observation for planning
#         '''
#         return {0: observations[0]}

#     def __len__(self):
#         return len(self.indices)

#     def __getitem__(self, idx, eps=1e-4):
#         path_ind, start, end = self.indices[idx]
#         observations = self.fields.normed_observations[path_ind, start:end]
#         actions = self.fields.normed_actions[path_ind, start:end]
#         rewards = self.fields.normed_rewards[path_ind, start:end]
#         #observations = self.fields.observations[path_ind, start:end]
#         #actions = self.fields.actions[path_ind, start:end]
#         #rewards = self.fields.rewards[path_ind, start:end]
#         task = np.array(self.task_list.index(self.fields.get_task(path_ind))).reshape(-1,1)#self.get_task_id(self.fields.get_task(path_ind))
#         conditions = self.get_conditions(observations)
#         trajectories = np.concatenate([rewards, actions, observations], axis=-1)
#         #if np.any(np.isnan(trajectories)):
#         #    print("True->>>")
#         batch = TaskBatch(trajectories, conditions, task, 1)
#         #print("Batch item load!")
#         return batch


class MetaSequenceDataset(torch.utils.data.Dataset):
    def __init__(self, replay_dir_list=[], task_list=[], horizon=64,
                 normalizer='LimitsNormalizer', preprocess_fns=[], max_path_length=1000,
                 max_n_episodes=200000, termination_penalty=0, use_padding=True,
                 seed=None, meta_world=False, maze2d=False, antmaze=False,
                 optimal=True, use_d4rl=False, pad_to_max_length=True,
                 use_epicare=False, num_actions=16):  # 添加EpiCare参数

        self.replay_dir_list = replay_dir_list
        self.task_list = task_list
        self.reward_scale = 400.0
        self.horizon = horizon
        self.max_path_length = max_path_length
        self.use_padding = use_padding
        self.pad_to_max_length = pad_to_max_length  # 设置属性
        self.record_values = []
        self.use_epicare = use_epicare
        self.num_actions = num_actions

        if use_epicare:
            print("[INFO] Loading datasets from EpiCare (HDF5 files)...")
            assert len(task_list) > 0, "EpiCare requires at least one dataset path."

            # Load EpiCare datasets using load_epicare_dataset
            itr = load_epicare_dataset(task_list, episodes_avail=None, num_actions=num_actions)
            all_trajectories = list(itr)

        elif use_d4rl:
            print("[INFO] Loading datasets from D4RL...")
            assert len(task_list) > 0, "D4RL requires at least one dataset name."

            all_trajectories = []
            for dataset_name in task_list:
                print(f"[INFO] Loading dataset: {dataset_name}")
                env = gym.make(dataset_name)
                dataset = d4rl.qlearning_dataset(env)
                trajectories = self.parse_d4rl_dataset(dataset)
                all_trajectories.extend(trajectories)
        else:
            raise NotImplementedError("Only D4RL and EpiCare are supported in this configuration.")

        fields = ReplayBuffer(max_n_episodes, max_path_length, termination_penalty)
        for episode in all_trajectories:
            if len(episode['rewards']) > self.max_path_length:
                continue
            
            # 修复：添加padding逻辑
            if self.pad_to_max_length:
                episode = self._pad_episode_to_max_length(episode)
            
            self.record_values.append(episode['rewards'].sum())
            fields.add_path(episode)
        fields.finalize()

        self.normalizer = DatasetNormalizer(fields, normalizer, path_lengths=fields['path_lengths'])
        self.observation_dim = fields.observations.shape[-1]
        self.action_dim = fields.actions.shape[-1]
        self.fields = fields
        self.n_episodes = fields.n_episodes
        self.path_lengths = fields.path_lengths
        self.normalize()

        print(fields)

    def _pad_episode_to_max_length(self, episode):
        """将episode padding到max_path_length"""
        current_length = len(episode['observations'])
        
        if current_length >= self.max_path_length:
            return episode
        
        pad_length = self.max_path_length - current_length
        
        # 使用最后一个状态和动作进行padding
        last_obs = episode['observations'][-1:].repeat(pad_length, axis=0)
        last_action = episode['actions'][-1:].repeat(pad_length, axis=0)
        
        # rewards和terminals在padding部分设为0
        pad_rewards = np.zeros((pad_length,))
        pad_terminals = np.zeros((pad_length, 1))
        
        padded_episode = {
            'observations': np.concatenate([episode['observations'], last_obs], axis=0),
            'actions': np.concatenate([episode['actions'], last_action], axis=0),
            'rewards': np.concatenate([episode['rewards'], pad_rewards], axis=0),
            'terminals': np.concatenate([episode['terminals'], pad_terminals], axis=0)
        }
        
        return padded_episode
    
    def parse_d4rl_dataset(self, dataset):
        print("[INFO] Parsing D4RL dataset...")
        trajectories = []

        if np.sum(dataset['terminals']) == 0:
            total_length = len(dataset['observations'])
            start_idx = 0
            while start_idx < total_length:
                end_idx = min(start_idx + self.max_path_length, total_length)
                traj = {
                    'observations': dataset['observations'][start_idx:end_idx],
                    'actions': dataset['actions'][start_idx:end_idx],
                    'rewards': dataset['rewards'][start_idx:end_idx],
                    'terminals': dataset['terminals'][start_idx:end_idx].reshape(-1, 1)
                }
                trajectories.append(traj)
                start_idx = end_idx
        else:
            episode_starts = np.where(dataset['terminals'])[0] + 1
            start_idx = 0
            for end_idx in episode_starts:
                while end_idx - start_idx > self.max_path_length:
                    split_end = start_idx + self.max_path_length
                    traj = {
                        'observations': dataset['observations'][start_idx:split_end],
                        'actions': dataset['actions'][start_idx:split_end],
                        'rewards': dataset['rewards'][start_idx:split_end],
                        'terminals': dataset['terminals'][start_idx:split_end].reshape(-1, 1)
                    }
                    trajectories.append(traj)
                    start_idx = split_end

                traj = {
                    'observations': dataset['observations'][start_idx:end_idx],
                    'actions': dataset['actions'][start_idx:end_idx],
                    'rewards': dataset['rewards'][start_idx:end_idx],
                    'terminals': dataset['terminals'][start_idx:end_idx].reshape(-1, 1)
                }
                trajectories.append(traj)
                start_idx = end_idx

        return trajectories

    def normalize(self, keys=['observations', 'actions', 'rewards', 'terminals']):
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes * self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)


class RTGActDataset(MetaSequenceDataset):
    '''[DEBUG] traj_ind
        adds a value field to the datapoints for training the value function
    '''

    def __init__(self, *args, discount=0.99, normed=True, seq_length=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.discounts = self.discount ** np.arange(self.max_path_length)[:, None]
        self.seq_length = seq_length
        self.draw()
    def normalize(self, keys=['observations', 'actions']):#, 'observations', 'rewards'
        '''
            normalize fields that will be predicted by the diffusion model
        '''
        for key in keys:
            if key == 'rewards':
                self.fields[f'normed_{key}'] = self.fields[key] #/ self.reward_scale
                continue
            array = self.fields[key].reshape(self.n_episodes*self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)
    def __len__(self):
        return len(self.indices)
    def draw(self):
        V = np.array(self.record_values)
        print(V.shape)
        normed_V = (V - V.min()) / (V.max() - V.min())
        #normed_V = normed_V * 2 - 1
        sns.set_palette("hls")
        mpl.rc("figure", figsize=(9, 5))
        fig = sns.distplot(normed_V,bins=20)
        fig.set_xlabel("Normalized Return", fontsize=16)
        fig.set_ylabel("Density", fontsize=16)
        displot_fig = fig.get_figure()
        displot_fig.savefig('./sub-optimal.pdf', dpi = 400)
    def discount_cumsum(self, x, gamma):
        x = x.squeeze(-1)
        discount_cumsum = np.zeros_like(x)
        discount_cumsum[:,-1] = x[:,-1]
        for t in reversed(range(x.shape[-1] - 1)):
            discount_cumsum[:,t] = x[:,t] + gamma * discount_cumsum[:,t + 1]
        return np.expand_dims(discount_cumsum, axis=-1)
    def __getitem__(self, idx, eps=1e-4):
        path_ind, start, end = self.indices[idx]
        path_inds = []
        interval = int(self.fields.n_episodes/len(self.task_list))
        for i in range(len(self.task_list)):
            path_inds.append((path_ind+interval*i)%self.fields.n_episodes)
        observations = np.zeros((len(path_inds), self.seq_length, self.observation_dim))
        count = start
        k = self.seq_length - 1
        while count >= 0 and k >= 0:
            observations[:, k, :] = self.fields.normed_observations[path_inds, count]
            k -= 1
            count -= 1
        actions = self.fields.normed_actions[path_inds, start:end]
        task = np.array([self.task_list.index(self.fields.get_task(path_ind)) for path_ind in path_inds])#self.get_task_id(self.fields.get_task(path_ind))
        rtg = self.discount_cumsum(self.fields['rewards'][path_inds, start:], gamma=self.discount)[:,:end-start] / (self.max_path_length-start)
        batch = TaskBatch(actions, observations, task, rtg[:, 0])
        return batch

class AugBatch:
    def __init__(self, trajectories, task, trajectory_ids=None, fragment_indices=None, **kwargs):
        self.trajectories = trajectories
        self.task = task
        # 新增：记录每个fragment属于哪个trajectory
        self.trajectory_ids = trajectory_ids
        # 新增：记录每个fragment在其trajectory中的位置索引
        self.fragment_indices = fragment_indices
        for key, value in kwargs.items():
            setattr(self, key, value)


class AugDataset(MetaSequenceDataset):
    def __init__(self, *args, discount=0.99, normed=True, seq_length=2, double_samples=False, **kwargs):
        kwargs.pop("env", None)
        super().__init__(*args, **kwargs)
        self.discount = discount
        self.seq_length = seq_length
        
        if normed:
            self.normalize()
        
        if double_samples:
            self.double_samples()
        
        # 预计算每个trajectory可以分成多少个fragments
        self.calculate_fragments_info()

    def calculate_fragments_info(self):
        """
        预计算每个trajectory的可采样subtrajectory信息
        现在计算的是可以采样的连续subtrajectory的数量（允许重叠）
        """
        self.trajectory_fragments = []
        self.total_fragments = 0

        for idx in range(self.n_episodes):
            path_length = self.path_lengths[idx]
            # 计算可以采样多少个长度为horizon的连续subtrajectory（允许重叠）
            # 如果path_length < horizon，至少可以采样1个（会padding）
            num_fragments = max(1, path_length - self.horizon + 1)
            self.trajectory_fragments.append({
                'trajectory_id': idx,
                'num_fragments': num_fragments,
                'path_length': path_length,
                'start_fragment_idx': self.total_fragments
            })
            self.total_fragments += num_fragments

        print(f"[INFO] Total trajectories: {self.n_episodes}")
        print(f"[INFO] Total possible subtrajectories (with overlap): {self.total_fragments}")
        print(f"[INFO] Average subtrajectories per trajectory: {self.total_fragments / self.n_episodes:.2f}")

    def double_samples(self, n=2):
        """
        Multiply the dataset by duplicating episodes n times.
        """
        if n < 1:
            raise ValueError("n must be at least 1")
        
        self.n_episodes *= n
        fields_to_double = ['observations', 'actions', 'rewards', 'terminals', 
                            'normed_observations', 'normed_actions', 'normed_rewards', 'normed_terminals']
        for key in fields_to_double:
            if key in self.fields._dict:
                self.fields[key] = np.concatenate([self.fields[key]] * n, axis=0)
        self.path_lengths = np.concatenate([self.path_lengths] * n, axis=0)

    def normalize(self, keys=['observations', 'actions', 'rewards', 'terminals']):
        for key in keys:
            array = self.fields[key].reshape(self.n_episodes * self.max_path_length, -1)
            normed = self.normalizer(array, key)
            self.fields[f'normed_{key}'] = normed.reshape(self.n_episodes, self.max_path_length, -1)

    def __len__(self):
        # 现在返回trajectory数量，而不是fragment数量
        return self.n_episodes

    def get_subtrajectory(self, trajectory_id, start_idx):
        """
        从trajectory中提取一个连续的subtrajectory (STATE_COND VERSION)

        Args:
            trajectory_id: trajectory的ID
            start_idx: subtrajectory的起始位置索引

        Returns:
            subtrajectory: 连续的subtrajectory数据 (shape: [horizon, transition_dim])
            condition: 前一个state，shape: (observation_dim,)
            actual_length: subtrajectory的实际长度
        """
        trajectory_info = self.trajectory_fragments[trajectory_id]
        path_length = trajectory_info['path_length']

        # 确保start_idx有效
        if start_idx >= path_length:
            start_idx = max(0, path_length - self.horizon)

        # 计算subtrajectory的起始和结束位置（连续的horizon个transitions）
        end = min(start_idx + self.horizon, path_length)
        start = start_idx

        # 获取连续的trajectory数据（已归一化）
        observations = self.fields.normed_observations[trajectory_id, start:end]
        actions = self.fields.normed_actions[trajectory_id, start:end]
        rewards = self.fields.normed_rewards[trajectory_id, start:end].reshape(-1, 1)
        terminals = self.fields.normed_terminals[trajectory_id, start:end].reshape(-1, 1)

        # 获取next_observations（连续的）
        next_observations = self.fields.normed_observations[trajectory_id, start+1:end+1]
        if len(next_observations) < len(observations):
            padding = np.repeat(next_observations[-1:], len(observations) - len(next_observations), axis=0)
            next_observations = np.concatenate([next_observations, padding], axis=0)

        # --- STATE_COND VERSION: 计算 condition：只使用前一个 state ---
        if start > 0:
            # 只取前一个state
            prev_idx = start - 1
            prev_state = self.fields.normed_observations[trajectory_id, prev_idx]
            condition = prev_state.astype(np.float32)
        else:
            # 这是trajectory的开始，使用全零state向量
            condition = np.zeros((self.observation_dim,), dtype=np.float32)

        # Pad to horizon if needed
        actual_length = end - start
        if actual_length < self.horizon:
            pad_length = self.horizon - actual_length
            observations = np.concatenate([observations, np.repeat(observations[-1:], pad_length, axis=0)], axis=0)
            actions = np.concatenate([actions, np.repeat(actions[-1:], pad_length, axis=0)], axis=0)
            rewards = np.concatenate([rewards, np.repeat(rewards[-1:], pad_length, axis=0)], axis=0)
            terminals = np.concatenate([terminals, np.repeat(terminals[-1:], pad_length, axis=0)], axis=0)
            next_observations = np.concatenate([next_observations, np.repeat(next_observations[-1:], pad_length, axis=0)], axis=0)

        # 拼接成完整的transition: [state, action, reward, terminal, next_state]
        subtrajectory = np.concatenate([observations, actions, rewards, terminals, next_observations], axis=-1)

        return subtrajectory, condition, actual_length


    def __getitem__(self, idx):
        """
        获取一个trajectory的完整信息
        """
        trajectory_id = idx
        trajectory_info = self.trajectory_fragments[trajectory_id]
        num_fragments = trajectory_info['num_fragments']
        
        # 返回trajectory的基本信息，让trainer来处理fragment采样
        return {
            'trajectory_id': trajectory_id,
            'num_fragments': num_fragments,
            'path_length': trajectory_info['path_length']
        }

    @staticmethod
    def collate_fn(batch):
        """
        Collate function for trajectory-level batches
        """
        trajectory_ids = [item['trajectory_id'] for item in batch]
        num_fragments_list = [item['num_fragments'] for item in batch]
        path_lengths = [item['path_length'] for item in batch]
        
        return TrajectoryBatch(
            trajectory_data=batch,
            trajectory_ids=trajectory_ids,
            max_fragments_per_traj=max(num_fragments_list)
        )
