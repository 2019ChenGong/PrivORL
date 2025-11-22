#!/bin/bash
# This script reproduces the PrivORL-n fidelity results for maze2d-medium-dense-v1 in Table III
# Expected output: 0.947 (for Marginal) and 0.983 (for Correlation)
# Runtime: ~3.5 hours on NVIDIA A6000

# Evaluation (2min)
python evaluation/eval-fidelity/marginal.py \
  --dataset maze2d-medium-dense-v1 \
  --dp_epsilon 10 \
  --load_path alter_0.3curiosity_driven_results_maze2d-medium-dense-v1_1.0/cleaned_maze2d-medium-dense-v1_samples_1000000.0_10dp_0.8.npz \
  --cur_rate 0.3
