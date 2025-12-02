#!/bin/bash
# This script reproduces the PrivORL-n fidelity results for maze2d-medium-dense-v1 in Table III
# Expected output: 0.947 (for Marginal) and 0.983 (for Correlation)
# The results are printed in the terminal after evaluation.
# Runtime: ~2 mins on NVIDIA A6000

# Check if the required .npz file already exists
REQUIRED_FILE="./results_maze2d-medium-dense-v1_0.3/maze2d-medium-dense-v1_samples_1000000.0_10dp_0.8.npz"

if [ -f "$REQUIRED_FILE" ]; then
    echo "Found existing file: $REQUIRED_FILE"
    echo "Skipping Step 1 and Step 2, proceeding directly to Step 3..."
else
    echo "File not found: $REQUIRED_FILE"
    echo "Starting from Step 1..."

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
fi

# Step 3: Evaluation (2min)
python evaluation/eval-fidelity/marginal.py \
  --dataset maze2d-medium-dense-v1 \
  --dp_epsilon 10 \
  --load_path ./results_maze2d-medium-dense-v1_0.3/maze2d-medium-dense-v1_samples_1000000.0_10dp_0.8.npz \
  --cur_rate 0.3
