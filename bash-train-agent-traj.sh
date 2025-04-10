#!/bin/bash

gpus=(0 1)
epsilons=( 1 10 )
epsilons=( 1 )


agents=(
    'cql'
    'iql'
    'edac'
    'td3_bc'
)

datasets=(
    # 'maze2d-umaze-dense-v1'
    # 'maze2d-medium-dense-v1'
    'maze2d-large-dense-v1'
    # 'kitchen-partial-v0'
    # 'halfcheetah-medium-replay-v2'
)

seeds=(0)


max_gpus=${#gpus[@]} 
max_workers=4

# Initialize GPU index and process counter
gpu_index=0
current_workers=0


for seed in "${seeds[@]}"; do
    for dataset in "${datasets[@]}"; do
        for epsilon in "${epsilons[@]}"; do
            for agent in "${agents[@]}"; do
                gpu=${gpus[$gpu_index]}

                diffusion_path="/p/fzv6enresearch/liuzheng/MTDiff/results/${dataset}/finetune/epsilon${epsilon}_horizon32/state_200000/sampled_trajectories.npz"
                checkpoints_path="results_trajectory/${dataset}/finetune/epsilon${epsilon}_horizon32/state_200000/${agent}"

                dataset_prefix=$(echo $dataset | cut -d'-' -f1)   # e.g., maze2d
                dataset_suffix=$(echo $dataset | cut -d'-' -f2-)  # e.g., umaze-dense-v1

                if [ "$agent" == "cql" ]; then
                    command="CUDA_VISIBLE_DEVICES=${gpu} python ${agent}_trajectory.py \
                        --seed \"${seed}\" \
                        --env \"${dataset}\" \
                        --config synther/corl/yaml/${agent}/${dataset_prefix}/${dataset_suffix}.yaml \
                        --checkpoints_path \"${checkpoints_path}\" \
                        --diffusion_path \"${diffusion_path}\" "
                else
                    command="CUDA_VISIBLE_DEVICES=${gpu} python ${agent}_trajectory.py \
                        --seed \"${seed}\" \
                        --env \"${dataset}\" \
                        --config synther/corl/yaml/${agent}/${dataset_prefix}/${dataset_suffix}.yaml \
                        --checkpoints_path \"${checkpoints_path}\" \
                        --diffusion.path \"${diffusion_path}\" "
                fi

                echo "Running on GPU ${gpu}: ${command}"
                eval $command &

                gpu_index=$(( (gpu_index + 1) % max_gpus ))

                current_workers=$((current_workers + 1))

                if (( current_workers >= max_workers )); then
                    wait -n 
                    current_workers=$((current_workers - 1))
                fi

                sleep 1
            done
        done
    done
done


wait

echo "All tasks have completed."