import os
import glob
import concurrent.futures
import csv
import time
import subprocess


# datasets = ["halfcheetah"]
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

datasets = ["halfcheetah-medium-replay-v2", 
            # "walker2d-medium-replay-v2"
            ]

# datasets = ["maze2d-umaze-dense-v1", "maze2d-medium-dense-v1", "maze2d-large-dense-v1"]
# datasets = ["maze2d-large-dense-v1"]
# datasets = ["maze2d-open-dense-v0"]
# datasets = ["maze2d-umaze-dense-v1", "maze2d-medium-dense-v1", "maze2d-large-dense-v1"]
# datasets = ["maze2d-umaze-dense-v1", "maze2d-medium-dense-v1", "maze2d-large-dense-v1"]
# datasets = ["maze2d-open-dense-v0"]
# datasets = ["kitchen-complete-v0", "kitchen-partial-v0", "kitchen-mixed-v0"]
# datasets = ["kitchen-partial-v0"]

# datasets = ['antmaze-umaze-v1', 'antmaze-medium-play-v1']
# datasets = ["antmaze-large-play-v1"]

# datasets_name = {"halfcheetah-medium-v2": ["walker2d-medium-v2"], 
#                  "walker2d-medium-v2":    ["halfcheetah-medium-v2"]}   

datasets_name = {"halfcheetah-medium-replay-v2": ['walker2d-full-replay-v2', 'halfcheetah-full-replay-v2', 'walker2d-medium-v2'],
                 "walker2d-medium-replay-v2":    ['halfcheetah-expert-v2', 'walker2d-medium-v2', 'halfcheetah-full-replay-v2'] }   

# datasets_name = {
#                  "maze2d-open-dense-v0":      ['maze2d-umaze-dense-v1', 'maze2d-medium-dense-v1', 'maze2d-large-dense-v1'], 
#                  "maze2d-umaze-dense-v1":     ['maze2d-open-dense-v0', 'maze2d-medium-dense-v1', 'maze2d-large-dense-v1'], 
#                  "maze2d-medium-dense-v1":    ['maze2d-open-dense-v0', 'maze2d-umaze-dense-v1', 'maze2d-large-dense-v1'],
#                  "maze2d-large-dense-v1":    ['maze2d-open-dense-v0', 'maze2d-umaze-dense-v1', 'maze2d-medium-dense-v1']
#                 }   

# datasets_name = {"antmaze-umaze-v1":      ['antmaze-medium-play-v1', 'antmaze-large-play-v1'], 
#                  "antmaze-medium-play-v1": ['antmaze-umaze-v1', 'antmaze-large-play-v1'], 
#                  "antmaze-large-play-v1":    ['antmaze-medium-play-v1', 'antmaze-umaze-v1']                
#                  }   

#
# datasets_name = {"kitchen-complete-v0":      ["kitchen-partial-v0", "kitchen-mixed-v0"], 
#                  "kitchen-partial-v0": ['kitchen-complete-v0', 'kitchen-mixed-v0'], 
#                  "kitchen-mixed-v0":    ["kitchen-complete-v0", "kitchen-partial-v0"]                }  



dp_epsilons = [10]
num_samples = [1e6]
seeds = [0]
gpus = ['1', '2']
max_workers = 1

pretraining_rate = 1.0
finetuning_rate = 0.999

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
        for dataset in datasets:
            for num_sample in num_samples:
                for dp_epsilon in dp_epsilons:
                    # dp diffusion
                    dataset_name = datasets_name[dataset]
                    # results_folder = f"./results_{dataset}_{pretraining_rate}"
                    # results_folder = f"./same_environment_results_{dataset}_{pretraining_rate}"               
                    results_folder = f"./alter_curiosity_driven_results_{dataset}_{pretraining_rate}"
                    finetune_load_path = os.path.join(results_folder, "model-9.pt")
                    store_path = f"{dataset}_samples_{num_sample}_{dp_epsilon}dp_{finetuning_rate}.npz"

                    env, version = dataset.split('-', 1)
                    
                    arguments = [
                        '--dataset', dataset,
                        '--datasets_name', dataset_name,
                        '--seed', seed,
                        '--load_checkpoint',
                        # '--curiosity_driven',
                        '--dp_epsilon', dp_epsilon,
                        '--results_folder', results_folder,
                        '--load_path', finetune_load_path,
                        '--save_file_name', store_path,
                        '--pretraining_rate', pretraining_rate,
                        '--finetuning_rate', finetuning_rate,
                        '--save_num_samples', int(num_sample),
                    ]
                    script_path = 'train_diffuser.py'
                    
                    command = ['python', script_path] + [str(arg) for arg in arguments]

                    futures.append(executor.submit(run_command_on_gpu, command, gpus[gpu_index]))

                    gpu_index = (gpu_index + 1) % len(gpus)
                    time.sleep(10) # especially for fine-tuning

    concurrent.futures.wait(futures)
