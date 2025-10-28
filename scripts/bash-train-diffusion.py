import os
import glob
import concurrent.futures
import csv
import time
import subprocess

datasets = [
            # "kitchen-partial-v0",
            # "maze2d-umaze-dense-v1", 
            "maze2d-medium-dense-v1", 
            # "maze2d-large-dense-v1", 
            # "halfcheetah-medium-replay-v2", 
            ]


datasets_name = {"halfcheetah-medium-replay-v2": ['walker2d-full-replay-v2', 'halfcheetah-expert-v2', 'walker2d-medium-v2'],
                 "maze2d-umaze-dense-v1":     ['maze2d-open-dense-v0', 'maze2d-medium-dense-v1', 'maze2d-large-dense-v1'], 
                 "maze2d-medium-dense-v1":    ['maze2d-open-dense-v0', 'maze2d-umaze-dense-v1', 'maze2d-large-dense-v1'],
                 "maze2d-large-dense-v1":    ['maze2d-open-dense-v0', 'maze2d-umaze-dense-v1', 'maze2d-medium-dense-v1'],
                 "kitchen-partial-v0": ['kitchen-complete-v0', 'kitchen-mixed-v0']
                 }

dp_epsilons = [10]
num_samples = [1e6]
seeds = [0]
gpus = ['0', '1', '2']
max_workers = 20

pretraining_rate = 1.0
finetuning_rates = [0.8]
curiosity_driven_rates = [0.3]

accountant = 'prv'  # 'gdp' or 'rdp'


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
                    for finetuning_rate in finetuning_rates:
                        for curiosity_driven_rate in curiosity_driven_rates:
                            # dp diffusion
                            dataset_name = datasets_name[dataset]
                            # results_folder = f"./results_{dataset}_{pretraining_rate}"
                            results_folder = f"./results_{dataset}_{curiosity_driven_rate}_{accountant}"
                            # results_folder = f"./same_environment_results_{dataset}_{pretraining_rate}"               
                            # results_folder = f"./alter_curiosity_driven_results_{dataset}_{pretraining_rate}"
                            # results_folder = f"./alter_for_mia_curiosity_driven_results_{dataset}_{pretraining_rate}"
                            # results_folder = f"./alter_{curiosity_driven_rate}curiosity_driven_results_{dataset}_{pretraining_rate}"
                            # results_folder = f"./alter_without_pretraining_curiosity_driven_results_{dataset}_{pretraining_rate}"            
                            # results_folder = f"./alter_without_curiosity_driven_results_{dataset}_{pretraining_rate}"       
                            # results_folder = f"./alter_whole_mujoco_full_results_{dataset}_{pretraining_rate}"     
                            if dataset == 'maze2d-medium-dense-v1':
                                finetune_load_path = os.path.join(results_folder, "pretraining-model-4.pt")
                            else:
                                finetune_load_path = os.path.join(results_folder, "pretraining-model-9.pt")
                            # finetune_load_path = os.path.join(results_folder, "pretraining-model-4.pt")

                            # if dataset == 'maze2d-medium-dense-v1':
                            #     finetune_load_path = os.path.join(f"./alter_curiosity_driven_results_{dataset}_{pretraining_rate}", "pretraining-model-4.pt")
                            # else:
                            #     finetune_load_path = os.path.join(f"./alter_curiosity_driven_results_{dataset}_{pretraining_rate}", "pretraining-model-9.pt")
                            
                            # store_path = f"for_ablation_{dataset}_samples_{num_sample}_{dp_epsilon}dp_{finetuning_rate}.npz"
                            store_path = f"{dataset}_samples_{num_sample}_{dp_epsilon}dp_{finetuning_rate}_{accountant}.npz"
                            # store_path = f"2epoch_{dataset}_samples_{num_sample}_{dp_epsilon}dp_{finetuning_rate}.npz"
                            # store_path = f"without_dp_{dataset}_samples_{num_sample}_{dp_epsilon}dp_{finetuning_rate}.npz"
                            # store_path = f"pretraining_cur_syn_{dataset}_samples_{num_sample}_{dp_epsilon}dp_{finetuning_rate}.npz"
                            env, version = dataset.split('-', 1)
                            
                            arguments = [
                                '--dataset', dataset,
                                '--datasets_name', dataset_name,
                                '--seed', seed,
                                # '--load_checkpoint',
                                '--curiosity_driven',
                                '--curiosity_driven_rate', curiosity_driven_rate,
                                '--dp_epsilon', dp_epsilon,
                                '--results_folder', results_folder,
                                '--load_path', finetune_load_path,
                                '--save_file_name', store_path,
                                '--pretraining_rate', pretraining_rate,
                                '--finetuning_rate', finetuning_rate,
                                '--save_num_samples', int(num_sample),
                                '--accountant', accountant,
                                # '--save_data',
                            ]
                            script_path = 'synther/training/train_diffuser.py'
                            
                            command = ['python', script_path] + [str(arg) for arg in arguments]

                            futures.append(executor.submit(run_command_on_gpu, command, gpus[gpu_index]))

                            gpu_index = (gpu_index + 1) % len(gpus)
                            time.sleep(10) # especially for fine-tuning

    concurrent.futures.wait(futures)
