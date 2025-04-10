import subprocess
import os
import time
import concurrent.futures

# Configurations
gpus = ['0', '1']
gpus = ['7', '5', '6', '3']
gpus = ['0', '1', '2', '3']
gpus = ['0', '1', '2']

epsilons = [
            1,
            #  10
             ]
agents = ['cql', 'iql', 'edac', 'td3_bc']

datasets=[
    'maze2d-umaze-dense-v1',
    # 'maze2d-medium-dense-v1',
    'maze2d-large-dense-v1',
    'kitchen-partial-v0',
    'halfcheetah-medium-replay-v2',
]
seeds = [1,2,3]

max_gpus = len(gpus)
max_workers = 6  # Maximum concurrent jobs

def run_command(command, gpu_id):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    subprocess.run(command, shell=True, env=env)

with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = []
    gpu_index = 0

    for seed in seeds:
        for dataset in datasets:
            for epsilon in epsilons:
                for agent in agents:
                    gpu = gpus[gpu_index]

                    diffusion_path = f"/p/fzv6enresearch/liuzheng/MTDiff/results_unet/{dataset}/finetune/epsilon{epsilon}_horizon32/state_200000/sampled_trajectories.npz"
                    checkpoints_path = f"results_trajectory_unet/{dataset}/finetune/epsilon{epsilon}_horizon32/state{200000}/seed{seed}/{agent}"

                    dataset_prefix = dataset.split('-')[0]
                    dataset_suffix = '-'.join(dataset.split('-')[1:])

                    if agent == "cql":
                        command = (
                            f"python {agent}_trajectory.py "
                            f"--seed {seed} "
                            f"--env {dataset} "
                            f"--config synther/corl/yaml/{agent}/{dataset_prefix}/{dataset_suffix}.yaml "
                            f"--checkpoints_path {checkpoints_path} "
                            f"--diffusion_path {diffusion_path}"
                        )
                    else:
                        command = (
                            f"python {agent}_trajectory.py "
                            f"--seed {seed} "
                            f"--env {dataset} "
                            f"--config synther/corl/yaml/{agent}/{dataset_prefix}/{dataset_suffix}.yaml "
                            f"--checkpoints_path {checkpoints_path} "
                            f"--diffusion.path {diffusion_path}"
                        )

                    print(f"Submitting on GPU {gpu}: {command}")
                    futures.append(executor.submit(run_command, command, gpu))
                    gpu_index = (gpu_index + 1) % max_gpus
                    time.sleep(1)

    concurrent.futures.wait(futures)

print("All tasks have completed.")
