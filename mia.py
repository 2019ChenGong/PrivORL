import torch
import numpy as np
import argparse
import gym
import d4rl
import os
from accelerate import Accelerator



def make_inputs(
        env: gym.Env
) -> np.ndarray:
    dataset = d4rl.qlearning_dataset(env)
    
    return dataset

parser = argparse.ArgumentParser()
parser.add_argument("--model", "-m", type=str, default="pretraining_pategan")
parser.add_argument("--dataset", "-d", type=str, default="kitchen-complete-v0")
parser.add_argument("--finetuning_rate", type=float, default=0.6)
args = parser.parse_args()

results_folder = f"./alter_curiosity_driven_results_{args.dataset}_0.8"
ckpt_path = os.path.join(results_folder, "finetuning-model-9.pt")
syndata_path = os.path.join(results_folder, f"{args.dataset}_samples_{1e6}_10dp_{args.finetuning_rate}.npz")

# load orginal and syn data
env = gym.make(args.dataset)
original_data = make_inputs(env)

syn_data = np.load(syndata_path)

# load finetuning model
data = torch.load(ckpt_path)

# accelerator = Accelerator(
#             split_batches=True,
#             mixed_precision='fp16' if fp16 else 'no'
#         )
# model = self.accelerator.unwrap_model(self.model)
# model.load_state_dict(data['model'])
print(1)