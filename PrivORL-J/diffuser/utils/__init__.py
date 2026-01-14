from .serialization import *
from .training_v1 import AugTrainer as AugTrainer_v1
from .training_v2 import AugTrainer as AugTrainer_v2
# from .training_v2_1mlp import AugTrainer as AugTrainer_v2_1mlp
# from .training_v2_casual import AugTrainer as AugTrainer_v2_casual
# from .training_v2_share import AugTrainer as AugTrainer_v2_share
# from .training_v2_fusion import AugTrainer as AugTrainer_v2_fusion
# from .training_v2_bottleneck import AugTrainer as AugTrainer_v2_bottleneck
# from .training_v2_dual_path import AugTrainer as AugTrainer_v2_dual_path
# from .training_v2_lightweight_share import AugTrainer as AugTrainer_v2_lightweight_share
# from .training_v2_enhanced_fusion import AugTrainer as AugTrainer_v2_enhanced_fusion
# from .training_v2_fusion_with_global import AugTrainer as AugTrainer_v2_fusion_with_global
from .progress import *
from .setup import *
from .config import *
from .rendering import *
from .arrays import *
from .colab import *
from .logger import *
from gym.envs.registration import register
register(
    id='maze2d-1',
    entry_point='d4rl.pointmaze:MazeEnv',
    max_episode_steps=600,
    kwargs={
        'maze_spec':MAZE_1,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 13.13,
        'ref_max_score': 277.39,
    }
)
register(
    id='maze2d-2',
    entry_point='d4rl.pointmaze:MazeEnv',
    max_episode_steps=600,
    kwargs={
        'maze_spec':MAZE_2,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 13.13,
        'ref_max_score': 277.39,
    }
)
register(
    id='maze2d-3',
    entry_point='d4rl.pointmaze:MazeEnv',
    max_episode_steps=600,
    kwargs={
        'maze_spec':MAZE_3,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 13.13,
        'ref_max_score': 277.39,
    }
)
register(
    id='maze2d-4',
    entry_point='d4rl.pointmaze:MazeEnv',
    max_episode_steps=600,
    kwargs={
        'maze_spec':MAZE_4,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 13.13,
        'ref_max_score': 277.39,
    }
)
register(
    id='maze2d-5',
    entry_point='d4rl.pointmaze:MazeEnv',
    max_episode_steps=600,
    kwargs={
        'maze_spec':MAZE_5,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 13.13,
        'ref_max_score': 277.39,
    }
)
register(
    id='maze2d-6',
    entry_point='d4rl.pointmaze:MazeEnv',
    max_episode_steps=600,
    kwargs={
        'maze_spec':MAZE_6,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 13.13,
        'ref_max_score': 277.39,
    }
)
register(
    id='maze2d-7',
    entry_point='d4rl.pointmaze:MazeEnv',
    max_episode_steps=600,
    kwargs={
        'maze_spec':MAZE_7,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 13.13,
        'ref_max_score': 277.39,
    }
)
register(
    id='maze2d-8',
    entry_point='d4rl.pointmaze:MazeEnv',
    max_episode_steps=600,
    kwargs={
        'maze_spec':MAZE_8,
        'reward_type':'sparse',
        'reset_target': False,
        'ref_min_score': 13.13,
        'ref_max_score': 277.39,
    }
)
