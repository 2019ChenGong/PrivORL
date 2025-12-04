#!/bin/bash
# This script reproduces the MIA results for maze2d-medium-dense-v1 in Table XII
# Expected output:
# ############ MIA Results: ############
# nonndp sigma: 0.100 TPR@10%FPR: 0.321 TPR@1%FPR: 0.1160 TPR@0.1%FPR: 0.07400 TPR@20%FPR: 0.467000
# dp1 sigma: 0.100 TPR@10%FPR: 0.089 TPR@1%FPR: 0.0110 TPR@0.1%FPR: 0.00300 TPR@20%FPR: 0.180000
# dp10 sigma: 0.100 TPR@10%FPR: 0.093 TPR@1%FPR: 0.0110 TPR@0.1%FPR: 0.00400 TPR@20%FPR: 0.189000
# ######################################
# The results are printed in the terminal after evaluation.
# Runtime: ~5 hours on NVIDIA A6000


# Step 1: Training nondp checkpoint (17h)
python synther/training/train_diffuser.py \
  --dataset maze2d-medium-dense-v1 \
  --datasets_name "['maze2d-open-dense-v0', 'maze2d-umaze-dense-v1', 'maze2d-large-dense-v1']" \
  --curiosity_driven \
  --curiosity_driven_rate 0.3 \
  --results_folder ./evaluation/eval-mia/mia_curiosity_driven_result_maze2d-medium-dense-v1 

python synther/training/train_diffuser_mia.py \
  --dataset maze2d-medium-dense-v1 \
  --results_folder ./evaluation/eval-mia/mia_curiosity_driven_result_maze2d-medium-dense-v1 \
  --load_path ./evaluation/eval-mia/mia_curiosity_driven_result_maze2d-medium-dense-v1/pretraining-model-4.pt \
  --save_file_name maze2d-medium-dense-v1_samples_1000000.0_nondp_0.8.npz \
  --load_checkpoint


# Step 2: Training dp1 checkpoint (2.5h)
python synther/training/train_diffuser_mia.py \
  --dataset maze2d-medium-dense-v1 \
  --dp_epsilon 1 \
  --results_folder ./evaluation/eval-mia/mia_curiosity_driven_result_maze2d-medium-dense-v1 \
  --load_path ./evaluation/eval-mia/mia_curiosity_driven_result_maze2d-medium-dense-v1/pretraining-model-4.pt \
  --save_file_name maze2d-medium-dense-v1_samples_1000000.0_1dp_0.8.npz \
  --load_checkpoint


# Step 3: Training dp10 checkpoint (2.5h)
python synther/training/train_diffuser_mia.py \
  --dataset maze2d-medium-dense-v1 \
  --dp_epsilon 10 \
  --results_folder ./evaluation/eval-mia/mia_curiosity_driven_result_maze2d-medium-dense-v1 \
  --load_path ./evaluation/eval-mia/mia_curiosity_driven_result_maze2d-medium-dense-v1/pretraining-model-4.pt \
  --save_file_name maze2d-medium-dense-v1_samples_1000000.0_10dp_0.8.npz \
  --load_checkpoint


# Step 4: Evaluation MIA (2min)
python evaluation/eval-mia/mia.py \
  --dataset maze2d-medium-dense-v1 \
  --nondp_weight finetuning_without_dp-model-149.pt \
  --dp1_weight finetuning_dp1.0-model-4.pt \
  --dp10_weight finetuning_dp10.0-model-4.pt \
  --repeat 64 \
  --sample_num 10000

