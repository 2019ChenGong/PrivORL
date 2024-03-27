import os
import glob
import concurrent.futures
import csv
import time
import subprocess

# datasets = ["halfcheetah"]
datasets = ["hopper", "halfcheetah", "walker2d"]
# datasets = ["halfcheetah", "walker2d"]
datasets_name = {"hopper":      ['hopper-expert-v2', 'hopper-full-replay-v2', 'hopper-medium-v2', 'hopper-random-v2'], 
                 "halfcheetah": ['halfcheetah-expert-v2', 'halfcheetah-full-replay-v2', 'halfcheetah-medium-v2', 'halfcheetah-random-v2'], 
                 "walker2d":    ['walker2d-expert-v2', 'walker2d-full-replay-v2', 'walker2d-medium-v2', 'walker2d-random-v2']}   

dp_epsilons = [8]
num_samples = [5e6]
seeds = [0]
gpus = ['0', '1', '2']
max_workers = 8
# algos = ['td3_bc', 'iql', 'cql', 'edac']
algos = ['cql']

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
    # with open(output_file, "a", newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile)
    #     csv_writer.writerow([env, unlearning_rate, unlearning_step, algo, start_time, result.stdout.strip()])


with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = []
    gpu_index = 0
    
    for seed in seeds:
        for algo in algos:
            for dataset in datasets:
                for num_sample in num_samples:
                    for dp_epsilon in dp_epsilons:
                        
                        # offline RL 
                        config = f"synther/corl/yaml/{algo}/{dataset}/medium_replay_v2.yaml"
                        dataset = dataset + "-medium-replay-v2"
                        results_folder = f"./results_{dataset}"
                        offlineRL_load_path = os.path.join(results_folder, f"{dataset}_samples_{num_sample}_{dp_epsilon}dp.npz")
                        
                        arguments = [
                            # '--dataset', dataset,
                            # '--datasets_name', dataset_name,
                            '--seed', seed,
                            '--checkpoints_path',checkpoints_path,
                            '--config', config,
                            '--dp_epsilon', dp_epsilon,
                            '--diffusion.path', offlineRL_load_path,
                            # '--name', name
                        ]
                        if algo == "td3_bc":
                            script_path = 'td3_bc.py'
                        # elif algo == "iql":
                        #     script_path = './synther/corl/algorithms/iql.py'
                        # elif algo == "cql":
                        #     script_path = './synther/corl/algorithms/cql.py'
                        # elif algo == "edac":
                        #     script_path = './synther/corl/algorithms/edac.py'
                        elif algo == "iql":
                            script_path = 'iql.py'
                        elif algo == "cql":
                            script_path = 'cql.py'
                        elif algo == "edac":
                            script_path = 'edac.py'
                        
                        command = ['python', script_path] + [str(arg) for arg in arguments]

                        futures.append(executor.submit(run_command_on_gpu, command, gpus[gpu_index]))

                        gpu_index = (gpu_index + 1) % len(gpus)
                        time.sleep(10) # especially for fine-tuning

    concurrent.futures.wait(futures)