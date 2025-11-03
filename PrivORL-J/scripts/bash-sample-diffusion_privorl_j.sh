#!/bin/bash


epsilons=(5 15 )
epsilons=(10  )
# epsilons=(1 10  )

curiosity_rates=(0.3)

accountant='rdp'

# finetune='False'
finetune='True'

datasets=(
    'maze2d-umaze-dense-v1'
    'maze2d-medium-dense-v1'
    'maze2d-large-dense-v1'
    # 'kitchen-partial-v0'
    'halfcheetah-medium-replay-v2'
)

# Run evaluation for each dataset
for dataset in "${datasets[@]}"; do
    for epsilon in "${epsilons[@]}"; do
        for curiosity_rate in "${curiosity_rates[@]}"; do
            if [ "$finetune" == "False" ]; then
                sample_checkpoint_path="PrivORL-J/logs_transition_cond/${dataset}/pretrain/horizon32/state_500000.pt"
                output_csv_path="PrivORL-J/results_transition_cond/${dataset}/pretrain/horizon32/state_500000/sampled_trajectories.csv"
                
                command="python PrivORL-J/scripts/sample.py \
                    --dataset \"${dataset}\" \
                    --finetune \"${finetune}\" \
                    --sample_checkpoint_path \"${sample_checkpoint_path}\" \
                    --output_csv_path \"${output_csv_path}\" \
                    --model models.TasksAug \
                    --diffusion models.AugDiffusion \
                    --loss_type statehuber \
                    --loader datasets.AugDataset"
            else
                sample_checkpoint_path="PrivORL-J/logs_transition_cond/${dataset}/finetune/epsilon${epsilon}_horizon32_curiosity${curiosity_rate}_${accountant}/state_final.pt"
                output_csv_path="PrivORL-J/results_transition_cond/${dataset}/finetune/epsilon${epsilon}_horizon32_curiosity${curiosity_rate}_${accountant}/state_final/sampled_trajectories.csv"

                command="python PrivORL-J/scripts/sample.py \
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
done

echo "All tasks have completed."