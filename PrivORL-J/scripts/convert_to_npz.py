import pandas as pd
import numpy as np

def convert_diffusion_to_cql(csv_path, npz_path):
    # 读取 CSV 文件
    df = pd.read_csv(csv_path)
    
    # 自动提取相关字段
    state_cols = [col for col in df.columns if col.startswith('state_')]
    action_cols = [col for col in df.columns if col.startswith('action_')]
    next_state_cols = [col for col in df.columns if col.startswith('next_state_')]

    # 确保关键字段存在
    for required in ['reward', 'terminal']:
        if required not in df.columns:
            raise ValueError(f"Missing column: {required}")

    # 提取数据
    observations = df[state_cols].values
    actions = df[action_cols].values
    next_observations = df[next_state_cols].values
    rewards = df['reward'].values.reshape(-1, 1)
    terminals = df['terminal'].values.reshape(-1, 1)

    # 构建保存格式
    data = {
        "observations": observations,
        "actions": actions,
        "next_observations": next_observations,
        "rewards": rewards,
        "terminals": terminals
    }

    # 保存为 .npz 文件
    np.savez(npz_path, **data)
    print(f"Converted data saved to {npz_path}")

# 示例用法
csv_input_path = "/scratch/fzv6en/liuzheng/dprl/results/maze2d-large-dense-v1/finetune/epsilon10_horizon32_curiosity0.3_rdp/state_final/sampled_trajectories.csv"
npz_output_path = "/scratch/fzv6en/liuzheng/dprl/results/maze2d-large-dense-v1/finetune/epsilon10_horizon32_curiosity0.3_rdp/state_final/sampled_trajectories.npz"

convert_diffusion_to_cql(csv_input_path, npz_output_path)
