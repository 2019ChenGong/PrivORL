import os
import glob
import concurrent.futures
import csv
import time
import subprocess

# datasets = ["halfcheetah"]
datasets = ["hopper", "halfcheetah", "walker2d"]
# datasets = ["halfcheetah", "walker2d"]
datasets_name = {"hopper":      ['hopper-medium-replay-v2', 'hopper-full-replay-v2', 'hopper-medium-v2', 'hopper-random-v2'], 
                 "halfcheetah": ['halfcheetah-medium-replay-v2', 'halfcheetah-full-replay-v2', 'halfcheetah-medium-v2', 'halfcheetah-random-v2'], 
                 "walker2d":    ['walker2d-medium-replay-v2', 'walker2d-full-replay-v2', 'walker2d-medium-v2', 'walker2d-random-v2']}   

dp_epsilons = [5]
num_samples = [5e6]
seeds = [0]
gpus = ['0', '1', '2']
max_workers = 8
# algos = ['td3_bc', 'iql', 'cql', 'edac']
algos = ['td3_bc']

# offline RL
checkpoints_path = "corl_logs/"  
name = "DPsynthER"

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
                        # dp diffusion
                        dataset_name = datasets_name[dataset]
                        dataset = dataset + "-expert-v2"
                        results_folder = f"./results_{dataset}"
                        finetune_load_path = os.path.join(results_folder, "model-390000.pt")
                        store_path = f"{dataset}_samples_{num_sample}_{dp_epsilon}dp.npz"
                        
                        arguments = [
                            '--dataset', dataset,
                            '--datasets_name', dataset_name,
                            '--seed', seed,
                            # '--load_checkpoint',
                            '--full_pretrain', # make sure finetune one dataset using other complete datasets
                            '--dp_epsilon', dp_epsilon,
                            '--results_folder', results_folder,
                            '--load_path', finetune_load_path,
                            '--save_file_name', store_path,
                        ]
                        script_path = 'train_diffuser.py'
                        
                        command = ['python', script_path] + [str(arg) for arg in arguments]

                        futures.append(executor.submit(run_command_on_gpu, command, gpus[gpu_index]))

                        gpu_index = (gpu_index + 1) % len(gpus)
                        time.sleep(10) # especially for fine-tuning

    concurrent.futures.wait(futures)