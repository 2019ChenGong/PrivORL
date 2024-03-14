import os
import glob
import concurrent.futures
import csv
import time
import subprocess

datasets = ["hopper", "halfcheetah", "walker2d-medium"]

dp_epsilons = [0.3]
num_samples = [5e6]
seeds = [0]
gpus = ['0', '1', '2', '3', '4']

max_workers = 8


def get_directories(path):
    directories = [os.path.join(path, d) for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    return directories

def run_command_on_gpu(command, gpu_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    subprocess.run(command)
    time.sleep(1)
    # with open(output_file, "a", newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile)
    #     csv_writer.writerow([env, unlearning_rate, unlearning_step, algo, start_time, result.stdout.strip()])


with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = []
    gpu_index = 0

    for dataset in datasets:
        for seed in seeds:
            for num_sample in num_samples:
                for dp_epsilon in dp_epsilons:
                    dataset = dataset + "-medium-replay-v2"
                    store_path = f"{dataset}_samples_{num_sample}_{dp_epsilon}dp"
                    arguments = [
                        '--dataset', dataset,
                        '--seed', seed,
                        '--dp_epsilon', 0.3,
                        '--save_file_name', store_path,
                    ]
                    script_path = 'train_diffuser.py'
                    command = ['python', script_path] + [str(arg) for arg in arguments]

                    futures.append(executor.submit(run_command_on_gpu, command, gpus[gpu_index]))

                    gpu_index = (gpu_index + 1) % len(gpus)

    concurrent.futures.wait(futures)