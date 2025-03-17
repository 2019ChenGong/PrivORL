import numpy as np

def atleast_2d(x):
    while x.ndim < 2:
        x = np.expand_dims(x, axis=-1)
    return x

class ReplayBuffer:

    def __init__(self, max_n_episodes, max_path_length, termination_penalty):
        self._dict = {
            'path_lengths': np.zeros(max_n_episodes, dtype=int),
        }
        self.tasks = []
        self._count = 0
        self.max_n_episodes = max_n_episodes
        self.max_path_length = max_path_length
        self.termination_penalty = termination_penalty

    def __repr__(self):
        return '[ datasets/buffer ] Fields:\n' + '\n'.join(
            f'    {key}: {val.shape}'
            for key, val in self.items()
        )

    def __getitem__(self, key):
        return self._dict[key]
    def get_task(self, id):
        print(f"Requested task id: {id}, Total tasks: {len(self.tasks)}")  # 调试信息
        if id < 0 or id >= len(self.tasks):
            raise IndexError(f"Task id {id} is out of range. Total tasks: {len(self.tasks)}")
        return self.tasks[id]

    def __setitem__(self, key, val):
        self._dict[key] = val
        self._add_attributes()

    @property
    def n_episodes(self):
        return self._count

    @property
    def n_steps(self):
        return sum(self['path_lengths'])

    def _add_keys(self, path):
        if hasattr(self, 'keys'):
            return
        self.keys = list(path.keys())

    def _add_attributes(self):
        '''
            can access fields with `buffer.observations`
            instead of `buffer['observations']`
        '''
        for key, val in self._dict.items():
            setattr(self, key, val)

    def items(self):
        return {k: v for k, v in self._dict.items()
                if k != 'path_lengths'}.items()

    def _allocate(self, key, array):
        assert key not in self._dict
        dim = array.shape[-1]
        shape = (self.max_n_episodes, self.max_path_length, dim)
        self._dict[key] = np.zeros(shape, dtype=np.float32)
        # print(f'[ utils/mujoco ] Allocated {key} with size {shape}')

    def add_path(self, path):
        path_length = len(path['observations'])
        
        # 如果路径超过最大长度，跳过
        if path_length > self.max_path_length:
            print(f"[DEBUG] 跳过路径，长度 {path_length} 超过最大长度 {self.max_path_length}")
            return
        
        # 初始化 keys
        self._add_keys(path)
        
        # 添加路径数据
        for key in self.keys:
            if key == 'task':
                self.tasks.append(path[key])
                continue
            array = atleast_2d(path[key])
            if key not in self._dict:
                self._allocate(key, array)
            self._dict[key][self._count, :path_length] = array
        
        # 记录路径长度
        self._dict['path_lengths'][self._count] = path_length
        self._count += 1
        # print(f"[DEBUG] 已添加路径，长度为 {path_length}")


    def truncate_path(self, path_ind, step):
        old = self._dict['path_lengths'][path_ind]
        new = min(step, old)
        self._dict['path_lengths'][path_ind] = new

    def finalize(self):
        # 如果没有添加任何路径，则 self.keys 不会被初始化
        if not hasattr(self, 'keys'):
            print("[Warning] 没有路径被添加到 ReplayBuffer。正在 finalize 一个空的 buffer。")
            self.keys = []
            
        # 删除多余的预留空间
        for key in self.keys + ['path_lengths']:
            if key == 'task':
                continue
            self._dict[key] = self._dict[key][:self._count]
        self._add_attributes()
        print(f'[ datasets/buffer ] 已完成 ReplayBuffer | 总计 {self._count} 个 episodes')
