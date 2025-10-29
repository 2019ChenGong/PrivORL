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

datasets = [
            # "halfcheetah-medium-replay-v2", 
            # "walker2d-medium-replay-v2",
            # "maze2d-umaze-dense-v1", "maze2d-medium-dense-v1", "maze2d-large-dense-v1",
            # "kitchen-partial-v0"
            ]

# datasets = ["maze2d-umaze-dense-v1", "maze2d-medium-dense-v1", "maze2d-large-dense-v1"]
# datasets = ["maze2d-large-dense-v1"]
datasets = ["maze2d-medium-dense-v1"]
# datasets = ["maze2d-umaze-dense-v1", "maze2d-medium-dense-v1"]

# datasets = ["maze2d-open-dense-v0"]
# datasets = ["kitchen-complete-v0", "kitchen-partial-v0", "kitchen-mixed-v0"]
# datasets = ["kitchen-partial-v0"]

# datasets = ['antmaze-umaze-v1', 'antmaze-medium-play-v1']
# datasets = ["antmaze-large-play-v1"]


dp_epsilons = [10]
num_samples = [1e6]
gpus = ['0', '1']
max_workers = 20


pretraining_rate = 1.0
finetuning_rates = [0.8]
cur_rate = 0.3

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
    
    for dataset in datasets:
        for num_sample in num_samples:
            for dp_epsilon in dp_epsilons:
                for finetuning_rate in finetuning_rates:

                    env, version = dataset.split('-', 1)
                    # load_path = f"alter_without_pretraining_curiosity_driven_results_{dataset}_{pretraining_rate}/cleaned_{dataset}_samples_{num_sample}_{dp_epsilon}dp_{finetuning_rate}.npz"
                    # load_path = f"alter_curiosity_driven_results_{dataset}_{pretraining_rate}/cleaned_without_dp_{dataset}_samples_{num_sample}_{dp_epsilon}dp_{finetuning_rate}.npz"
                    load_path = f"alter_{cur_rate}curiosity_driven_results_{dataset}_{pretraining_rate}/cleaned_{dataset}_samples_{num_sample}_{dp_epsilon}dp_{finetuning_rate}.npz"

                    arguments = [
                        '--dataset', dataset,
                        '--dp_epsilon', dp_epsilon,
                        '--pretraining_rate', pretraining_rate,
                        '--finetuning_rate', finetuning_rate,
                        '--samples', int(num_sample),
                        '--load_path', load_path,
                        '--cur_rate', cur_rate,
                    ]
                    script_path = 'evaluation/eval-fidelity/marginal.py'
                    
                    command = ['python', script_path] + [str(arg) for arg in arguments]
                    import pdb
                    pdb.set_trace()
                    futures.append(executor.submit(run_command_on_gpu, command, gpus[gpu_index]))

                    gpu_index = (gpu_index + 1) % len(gpus)
                    time.sleep(10) # especially for fine-tuning

    concurrent.futures.wait(futures)
