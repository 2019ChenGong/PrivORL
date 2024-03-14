import os
import glob
import concurrent.futures
import csv
import time
import subprocess

datasets = ["hopper-medium-replay-v2", "halfcheetah-medium-replay-v2", "walker2d-medium-replay-v2"]

dp_epsilon = [0.3]
unlearning_steps = [10000, 100000]
seeds = [0, 1, 2]
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
            for unlearning_step in unlearning_steps:
                file_folder = f"./Total_trained/{env}/"
                folders = get_directories(file_folder)
                for folder in folders:
                    model_param = os.path.join(folder, 'params.json')
                    search_pattern = os.path.join(folder, '*_1000000.pt')
                    model_files = glob.glob(search_pattern)
                    if not model_files:
                        continue
                    model = os.path.join(folder, os.path.basename(model_files[0]))
                    

                    part = os.path.basename(folder).split('_')[0]
                    start_time = os.path.basename(folder).split('_')[1]
                    algo = "PLASP" if part == "PLASWithPerturbation" else part

                    arguments = [
                        '--dataset', env,
                        '--model', model_param,
                        '--model_params', model,
                        '--number_of_finetuning', unlearning_step,
                        '--seed', seed,
                        '--unlearning_rate', 0.3,
                        '--algo', algo,
                        '--gpu', gpus[gpu_index] 
                    ]
                    script_path = 'total_fine_tune.py'
                    command = ['python', script_path] + [str(arg) for arg in arguments]

                    futures.append(executor.submit(run_command_on_gpu, command, gpus[gpu_index]))

                    gpu_index = (gpu_index + 1) % len(gpus)

    concurrent.futures.wait(futures)