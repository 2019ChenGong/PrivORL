import os
import glob
import concurrent.futures
import csv
import time
import subprocess

"""
    dataset:
        maze2d-open-dense-v0
        maze2d-umaze-dense-v1
        maze2d-medium-dense-v1
        maze2d-large-dense-v1

        antmaze-umaze-v1
        antmaze-medium-play-v1
        antmaze-large-play-v1

        kitchen-complete-v0
        kitchen-partial-v0
        kitchen-mixed-v0
"""

# datasets = ["halfcheetah"]
# datasets = ["hopper", "halfcheetah", "walker2d"]

# datasets = ['antmaze-umaze-v1', 'antmaze-medium-play-v1', 'antmaze-large-play-v1']
# datasets = ['antmaze-umaze-v1']
# datasets = ['antmaze-medium-play-v1', 'antmaze-umaze-v1']

# datasets = ["maze2d-open-dense-v0", "maze2d-umaze-dense-v1", "maze2d-medium-dense-v1", "maze2d-large-dense-v1"]
# datasets = ["maze2d-large-dense-v1"]
# datasets = ["maze2d-umaze-dense-v1"]

# datasets = [
#             "maze2d-umaze-dense-v1", 
#             "maze2d-medium-dense-v1", 
#             "maze2d-large-dense-v1",  
#             "kitchen-partial-v0",  
#             "halfcheetah-medium-replay-v2", 
#             "walker2d-medium-replay-v2",
#             ]

# datasets = ["kitchen-complete-v0", "kitchen-partial-v0", "kitchen-mixed-v0"]
datasets = ["halfcheetah-medium-replay-v2"]
# datasets = ["maze2d-medium-dense-v1"]

# datasets = ["halfcheetah-medium-v2", "walker2d-medium-v2"]

# datasets = ["maze2d-large-dense-v1", "halfcheetah-medium-replay-v2", "maze2d-medium-dense-v1"]
# datasets = ["kitchen-partial-v0", "halfcheetah-medium-v2"]

# datasets = ["maze2d-medium-dense-v1", "maze2d-umaze-dense-v1"]
# datasets = ["maze2d-medium-dense-v1"]

pretraining_rate = 1.0
finetuning_rate = 0.8

# curiosity_driven_rates = [0.1, 0.2 ,0.4 ,0.5]
curiosity_driven_rates = [0.3]

# dp_epsilons = [1, 5, 10, 15, 20]
# dp_epsilons = [1, 5, 10, 15]
dp_epsilons = [10]
accountant = 'rdp'  # 'prv' or 'rdp'

num_samples = [1e6]
seeds = [0, 1, 2, 3, 4, 5]
gpus = ['0', '1', '2']
max_workers = 6
# algos = ['td3_bc', 'iql', 'edac', 'cql']
algos = ['td3_bc', 'iql', 'edac']
# algos = ['iql']

# offline RL
name = "CurDPsynthER"
# name = "DPsynthER"

def get_directories(path):
    directories = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return directories

def run_command_on_gpu(command, gpu_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    subprocess.run(command)
    time.sleep(1)


with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = []
    gpu_index = 0
    
    for seed in seeds:
        for algo in algos:
            for dataset in datasets:
                for num_sample in num_samples:
                    for dp_epsilon in dp_epsilons:
                        for curiosity_driven_rate in curiosity_driven_rates:
                            # offline RL 
                            env, version = dataset.split('-', 1)
                            # checkpoints_path = f"corl_logs_{env}/"
                            checkpoints_path = f"corl_logs_param_analysis_v1_{env}_{accountant}_24/"
                            # checkpoints_path = f"corl_logs_param_analysis_v1_{env}_pretrain/"
                            # checkpoints_path = f"corl_logs_without_dp_{env}/"
                            # checkpoints_path = f"corl_logs_ablation_{env}/"
                            # checkpoints_path = f"corl_logs_real_{env}/"

                            config = f"synther/corl/yaml/{algo}/{env}/{version}.yaml"
                            # dataset = dataset + "-expert-v2"
                            # results_folder = f"./alter_curiosity_driven_results_{dataset}_{pretraining_rate}"
                            # results_folder = f"./alter_without_curiosity_driven_results_{dataset}_{pretraining_rate}"
                            # results_folder = f"./alter_without_pretraining_curiosity_driven_results_{dataset}_{pretraining_rate}"
                            # results_folder = f"./alter_{curiosity_driven_rate}curiosity_driven_results_{dataset}_{pretraining_rate}"
                            # results_folder = f"./results_{dataset}_{pretraining_rate}"
                            results_folder = f"./results_{dataset}_{curiosity_driven_rate}_{accountant}_24"
                            # results_folder = f"./results_{dataset}_{curiosity_driven_rate}"
                            # results_folder = f"./alter_whole_mujoco_full_results_{dataset}_{pretraining_rate}"

                            # offlineRL_load_path = os.path.join(results_folder, f"{dataset}_samples_{num_sample}_{dp_epsilon}dp_{finetuning_rate}.npz")
                            offlineRL_load_path = os.path.join(results_folder, f"cleaned_{dataset}_samples_{num_sample}_{dp_epsilon}dp_{finetuning_rate}_{accountant}.npz")
                            # offlineRL_load_path = os.path.join(results_folder, f"cleaned_{dataset}_samples_{num_sample}_{dp_epsilon}dp_{finetuning_rate}_real.npz")
                            # offlineRL_load_path = os.path.join(results_folder, f"cleaned_pretrain_samples.npz")
                            # offlineRL_load_path = os.path.join(results_folder, f"cleaned_without_dp_{dataset}_samples_{num_sample}_{dp_epsilon}dp_{finetuning_rate}.npz")
                            
                            prefix = f'{curiosity_driven_rate}CurRate'
                            # prefix = f'Without_pre'
                            # prefix = 'NEW_Without_pre'
                            save_checkpoints = False

                            if algo == "td3_bc" or algo == "iql" or algo == 'edac':
                                arguments = [
                                '--env', dataset,
                                '--seed', seed,
                                '--checkpoints_path',checkpoints_path,
                                '--config', config,
                                '--dp_epsilon', dp_epsilon,
                                '--diffusion.path', offlineRL_load_path,
                                '--name', name,
                                '--prefix', prefix,
                                '--save_checkpoints', save_checkpoints
                                ]
                            else:
                                arguments = [
                                '--env', dataset,
                                '--seed', seed,
                                '--checkpoints_path',checkpoints_path,
                                '--config', config,
                                '--dp_epsilon', dp_epsilon,
                                '--diffusion_path', offlineRL_load_path,
                                '--name', name,
                                '--prefix', prefix,
                                '--save_checkpoints', save_checkpoints
                                ]

                            if algo == "td3_bc":
                                script_path = 'evaluation/eval-agent/td3_bc.py'
                            elif algo == "iql":
                                script_path = 'evaluation/eval-agent/iql.py'
                            elif algo == "cql":
                                script_path = 'evaluation/eval-agent/cql.py'
                            elif algo == "awac":
                                script_path = 'evaluation/eval-agent/awac.py'
                            elif algo == "edac":
                                script_path = 'evaluation/eval-agent/edac.py'
                            
                            command = ['python', script_path] + [str(arg) for arg in arguments]
                            
                            futures.append(executor.submit(run_command_on_gpu, command, gpus[gpu_index]))

                            gpu_index = (gpu_index + 1) % len(gpus)
                            time.sleep(10)

    concurrent.futures.wait(futures)