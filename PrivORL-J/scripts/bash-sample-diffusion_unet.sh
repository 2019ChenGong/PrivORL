#!/bin/bash


epsilons=(5 15 )
epsilons=(5  )
epsilons=(1  10 )

# finetune='False'
finetune='True'

datasets=(
    'maze2d-umaze-dense-v1'
    # 'maze2d-medium-dense-v1'
    'maze2d-large-dense-v1'
    'kitchen-partial-v0'
    'halfcheetah-medium-replay-v2'
)

# Run evaluation for each dataset
for dataset in "${datasets[@]}"; do
    for epsilon in "${epsilons[@]}"; do
        if [ "$finetune" == "False" ]; then
            sample_checkpoint_path="logs_unet/${dataset}/pretrain/horizon32/state_500000.pt"
            output_csv_path="results_unet/${dataset}/pretrain/horizon32/state_500000/sampled_trajectories.csv"
            
            command="python scripts/sample.py \
                --dataset \"${dataset}\" \
                --finetune \"${finetune}\" \
                --sample_checkpoint_path \"${sample_checkpoint_path}\" \
                --output_csv_path \"${output_csv_path}\" \
                --model models.TasksAug \
                --diffusion models.AugDiffusion \
                --loss_type statehuber \
                --loader datasets.AugDataset"
        else
            sample_checkpoint_path="logs_unet/${dataset}/finetune/epsilon${epsilon}_horizon32/state_200000.pt"
            output_csv_path="results_unet/${dataset}/finetune/epsilon${epsilon}_horizon32/state_200000/sampled_trajectories.csv"

            command="python scripts/sample.py \
                --dataset \"${dataset}\" \
                --finetune \"${finetune}\" \
                --target_epsilon \"${epsilon}\" \
                --sample_checkpoint_path \"${sample_checkpoint_path}\" \
                --output_csv_path \"${output_csv_path}\" \
                --model models.TasksAug \
                --diffusion models.AugDiffusion \
                --loss_type statehuber \
                --loader datasets.AugDataset"
        fi
        
        echo "Running: ${command}"
        eval $command 

    done
done

echo "All tasks have completed."