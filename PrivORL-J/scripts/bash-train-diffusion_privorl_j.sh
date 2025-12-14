#!/bin/bash

# Define parameters
gpus=(0 1 2 3)
gpus=(2 1 0)
gpus=(0)

finetune="False"
finetune="True"

# horizons=( 16 32 64 128)
horizons=(32)
batch_size=32

curiosity_rates=(0.1 0.2 0.4 0.5)
curiosity_rates=(0.3)

# epsilons=(5 15 )
epsilons=(1 5 10 15 20 )
epsilons=(1 10 )
epsilons=(10  )

datasets=(
    'maze2d-umaze-dense-v1'
    'maze2d-medium-dense-v1'
    'maze2d-large-dense-v1'
    # 'kitchen-partial-v0'
    # 'halfcheetah-medium-replay-v2'
)

max_gpus=${#gpus[@]} 
max_workers=4

# Initialize GPU index and process counter
gpu_index=0
current_workers=0

# Run evaluation for each dataset
for horizon in "${horizons[@]}"; do
    for curiosity_rate in "${curiosity_rates[@]}"; do
        for dataset in "${datasets[@]}"; do
            for epsilon in "${epsilons[@]}"; do
                gpu=${gpus[$gpu_index]}
            
                if [ "$finetune" == "False" ]; then
                    command="CUDA_VISIBLE_DEVICES=${gpu} python PrivORL-J/scripts/training.py \
                        --dataset \"${dataset}\" \
                        --finetune \"${finetune}\" \
                        --curiosity_rate \"${curiosity_rate}\" \
                        --horizon \"${horizon}\" \
                        --batch_size \"${batch_size}\" \
                        --model models.TasksAug \
                        --diffusion models.AugDiffusion \
                        --loss_type statehuber \
                        --loader datasets.AugDataset\
                        --logbase 'PrivORL-J/logs_transition_cond_final'"
                else
                    checkpoint_path="PrivORL-J/logs_transition_cond_final/${dataset}/pretrain/horizon${horizon}_curiosity${curiosity_rate}/state_final.pt"
                    command="CUDA_VISIBLE_DEVICES=${gpu} python PrivORL-J/scripts/training.py \
                        --dataset \"${dataset}\" \
                        --finetune \"${finetune}\" \
                        --curiosity_rate \"${curiosity_rate}\" \
                        --horizon \"${horizon}\" \
                        --batch_size \"${batch_size}\" \
                        --checkpoint_path \"${checkpoint_path}\" \
                        --target_epsilon \"${epsilon}\" \
                        --model models.TasksAug \
                        --diffusion models.AugDiffusion \
                        --loss_type statehuber \
                        --loader datasets.AugDataset \
                        --logbase 'PrivORL-J/logs_transition_cond_final' \
                        --accountant 'rdp'"
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