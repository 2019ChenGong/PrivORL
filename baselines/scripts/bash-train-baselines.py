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
            # 'antmaze-umaze-v1', 'antmaze-medium-play-v1', 'antmaze-large-play-v1',
            "maze2d-open-dense-v0", "maze2d-umaze-dense-v1", "maze2d-medium-dense-v1", "maze2d-large-dense-v1",
            # "kitchen-complete-v0", "kitchen-partial-v0", "kitchen-mixed-v0"
            ]

# datasets = ["halfcheetah-medium-v2", "walker2d-medium-v2"]
# datasets = ["halfcheetah-medium-replay-v2", "walker2d-medium-replay-v2"]
# datasets = ['maze2d-open-dense-v0']

# datasets = ["maze2d-open-dense-v0", "maze2d-umaze-dense-v1", "maze2d-medium-dense-v1", "maze2d-large-dense-v1"]

# datasets = ["kitchen-complete-v0", "kitchen-partial-v0", "kitchen-mixed-v0"]

models = ['pretraining_pategan']
# models = ['pategan']

# models = ['privsyn', 'pgm']
# models = ['privsyn', 'pgm', 'pategan']
epsilons = [10.0]
gpus = ['1', '2']
max_workers = 100


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
    
    for epsilon in epsilons:
        for model in models:
            for dataset in datasets:

                arguments = [
                    '--d', dataset,
                    '--m', model,
                    # '--epsilon', epsilon,
                    # '--finetuning',
                ]
                script_path = 'baselines/scripts/train_synthesizer.py'
                # script_path = 'baselines/scripts/syn_synthesizer.py'
                
                command = ['python', script_path] + [str(arg) for arg in arguments]

                futures.append(executor.submit(run_command_on_gpu, command, gpus[gpu_index]))

                gpu_index = (gpu_index + 1) % len(gpus)
                time.sleep(10) # especially for fine-tuning

    concurrent.futures.wait(futures)