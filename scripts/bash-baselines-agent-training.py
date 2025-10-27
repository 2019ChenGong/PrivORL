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

# datasets = ['antmaze-umaze-v1', 'antmaze-medium-play-v1', 
            # 'antmaze-large-play-v1'
            # ]

# datasets = [
            # "maze2d-open-dense-v0", 
            # "maze2d-umaze-dense-v1", 
            # "maze2d-medium-dense-v1", 
            # "maze2d-large-dense-v1"
            # ]

# datasets = ["kitchen-complete-v0", "kitchen-partial-v0", "kitchen-mixed-v0"]

# datasets = ["halfcheetah-medium-replay-v2", "walker2d-medium-replay-v2"]

# datasets = ["halfcheetah-medium-v2"]

# datasets = ["kitchen-complete-v0"]

datasets = [
            'antmaze-umaze-v1', 'antmaze-medium-play-v1', 'antmaze-large-play-v1',
            # "maze2d-open-dense-v0", "maze2d-umaze-dense-v1", 
            # "maze2d-medium-dense-v1", "maze2d-large-dense-v1",
            "kitchen-complete-v0", "kitchen-partial-v0", "kitchen-mixed-v0",
            # "halfcheetah-medium-replay-v2", "walker2d-medium-replay-v2"
            ]

pretraining_rate = 0.3
finetuning_rate = 0.99

dp_epsilons = [10.0]
num_samples = [1e6]
seeds = [0]
# gpus = ['0', '1', '2']
gpus = ['1', '2', '3']
max_workers = 100
# algos = ['cql', 'iql']
# algos = [
#         'edac', 'cql', 
#         'iql', 'td3_bc'
#         ]
algos = ['edac']

# offline RL
names = ['pategan', 'pgm', 'privsyn']

# names = ['pretraining_pategan']

baseline_test = "10iter"

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
    
    for name in names:
        for seed in seeds:
            for algo in algos:
                for dataset in datasets:
                    for num_sample in num_samples:
                        for dp_epsilon in dp_epsilons:
                            # offline RL 
                            env, version = dataset.split('-', 1)
                            checkpoints_path = f"corl_logs_{env}/"
                            config = f"synther/corl/yaml/{algo}/{env}/{version}.yaml"
                            # dataset = dataset + "-expert-v2"
                            results_folder = f"./results_{dataset}_{pretraining_rate}"

                            # offlineRL_load_path = f'baselines/samples/{dataset}/{name}_{dp_epsilon}/{baseline_test}_{dataset}.npz'
                            offlineRL_load_path = f'baselines/samples/{dataset}/{name}_{dp_epsilon}/{dataset}.npz'
                            
                            if algo == "td3_bc" or algo == "iql":
                                arguments = [
                                '--env', dataset,
                                '--seed', seed,
                                '--checkpoints_path',checkpoints_path,
                                '--config', config,
                                '--dp_epsilon', dp_epsilon,
                                '--diffusion.path', offlineRL_load_path,
                                '--name', name,
                                '--baseline_test', baseline_test
                                ]
                            elif algo == "awac" or algo == "cql":
                                arguments = [
                                '--env', dataset,
                                '--seed', seed,
                                '--checkpoints_path',checkpoints_path,
                                '--config', config,
                                '--dp_epsilon', dp_epsilon,
                                '--diffusion_path', offlineRL_load_path,
                                '--name', name,
                                '--baseline_test', baseline_test
                                ]
                            else:
                                arguments = [
                                '--env', dataset,
                                '--seed', seed,
                                '--checkpoints_path',checkpoints_path,
                                '--config', config,
                                '--dp_epsilon', dp_epsilon,
                                '--diffusion.path', offlineRL_load_path,
                                '--name', name,
                                '--baseline_test', baseline_test
                                ]
                            if algo == "td3_bc":
                                script_path = 'td3_bc.py'
                            elif algo == "iql":
                                script_path = 'iql.py'
                            elif algo == "cql":
                                script_path = 'cql.py'
                            elif algo == "edac":
                                script_path = 'edac.py'
                            elif algo == "awac":
                                script_path = 'awac.py'
                            
                            command = ['python', script_path] + [str(arg) for arg in arguments]

                            futures.append(executor.submit(run_command_on_gpu, command, gpus[gpu_index]))

                            gpu_index = (gpu_index + 1) % len(gpus)
                            time.sleep(10) # especially for fine-tuning

    concurrent.futures.wait(futures)