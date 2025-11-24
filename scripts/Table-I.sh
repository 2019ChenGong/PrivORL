#!/bin/bash
# This script reproduces the PrivORL-n utility results for maze2d-medium-dense-v1 in Table I
# Expected output: Normalized return approximately 90.7 ± 8.6
# Runtime: ~3.5 hours on NVIDIA A6000

# Step 1: Pre-training (1.5h)
python synther/training/train_diffuser.py \
  --dataset maze2d-medium-dense-v1 \
  --datasets_name "['maze2d-open-dense-v0', 'maze2d-umaze-dense-v1', 'maze2d-large-dense-v1']" \
  --curiosity_driven \
  --curiosity_driven_rate 0.3 \
  --results_folder ./results_maze2d-medium-dense-v1_0.3

# Step 2: Fine-tuning + Sampling (2h + 0.5h)
python synther/training/train_diffuser.py \
  --dataset maze2d-medium-dense-v1 \
  --dp_epsilon 10 \
  --results_folder ./results_maze2d-medium-dense-v1_0.3 \
  --load_path ./results_maze2d-medium-dense-v1_0.3/pretraining-model-4.pt \
  --save_file_name maze2d-medium-dense-v1_samples_1000000.0_10dp_0.8.npz \
  --load_checkpoint

# Step 3: Evaluation (1.5h)
python evaluation/eval-agent/iql.py \
  --env maze2d-medium-dense-v1 \
  --checkpoints_path corl_logs_param_analysis_v1_maze2d/ \
  --config synther/corl/yaml/iql/maze2d/medium-dense-v1.yaml \
  --dp_epsilon 10 \
  --diffusion.path ./results_maze2d-medium-dense-v1_0.3/maze2d-medium-dense-v1_samples_1000000.0_10dp_0.8.npz \
  --name CurDPsynthER \
  --prefix 0.3CurRate \
  --save_checkpoints False


# Validation: Check that output file contains normalized return in range [80.0, 120.0]