import torch
import numpy as np
import argparse
import gym
import d4rl
import os
from typing import Optional, List

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from synther.diffusion.denoiser_network import ResidualMLPDenoiser
from synther.diffusion.elucidated_diffusion import ElucidatedDiffusion
from synther.diffusion.utils import construct_diffusion_model
from synther.diffusion.norm import normalizer_factory


def get_metrics(label, score, fixed_fpr=0.001):
    """
    Compute TPR at FPR
    """
    from sklearn.metrics import roc_curve
    
    fpr, tpr, _ = roc_curve(label, score)

    
    tpr_at_low_fpr = tpr[np.where(fpr <= fixed_fpr)[0][-1]]
    
    
    return tpr_at_low_fpr


def make_inputs(dataset):
    obs = dataset['observations']
    actions = dataset['actions']
    next_obs = dataset['next_observations']
    rewards = dataset['rewards']
    terminals = dataset['terminals'].astype(np.float32)
    inputs = np.concatenate([obs, actions, rewards[:, None], next_obs, terminals[:, None]], axis=1)
    return inputs


def construct_diffusion_model(
        inputs: torch.Tensor,
        normalizer_type: str,
        # denoising_network: nn.Module,
        disable_terminal_norm: bool = False,
        skip_dims: List[int] = [],
        cond_dim: Optional[int] = None,
) -> ElucidatedDiffusion:
    event_dim = inputs.shape[1]
    model = ResidualMLPDenoiser(d_in=event_dim,
                                dim_t=128, mlp_width=2048, num_layers=6,
                                learned_sinusoidal_cond=False,
                                random_fourier_features=True,
                                learned_sinusoidal_dim=16,
                                activation='relu',
                                layer_norm=False)

    if disable_terminal_norm:
        terminal_dim = event_dim - 1
        if terminal_dim not in skip_dims:
            skip_dims.append(terminal_dim)

    if skip_dims:
        print(f"Skipping normalization for dimensions {skip_dims}.")

    normalizer = normalizer_factory(normalizer_type, inputs, skip_dims=skip_dims)

    return ElucidatedDiffusion(
        net=model,
        normalizer=normalizer,
        event_shape=[event_dim],
    )


def get_data_and_model(config):
    results_folder = f"./evaluation/eval-mia/alter_for_mia_curiosity_driven_results_{config.dataset}"
    ckpt_path = os.path.join(results_folder, config.nondp_weight)
    # ckpt_path = os.path.join(f"./alter_curiosity_driven_results_{config.dataset}_{config.pretraining_rate}", config.nondp_weight)
    dp1_ckpt_path = os.path.join(results_folder, config.dp1_weight)
    dp10_ckpt_path = os.path.join(results_folder, config.dp10_weight)
    train_data_path = os.path.join(results_folder, "train_data_1000.npy")
    test_data_path = os.path.join(results_folder, "test_data_1000.npy")

    # load orginal and syn data
    train_data = np.load(train_data_path)
    train_data = torch.from_numpy(train_data).float().cuda()

    test_data = np.load(test_data_path)
    test_data = torch.from_numpy(test_data).float().cuda()

    train_data = train_data[:config.sample_num]
    test_data = test_data[:config.sample_num]

    print(train_data.shape, test_data.shape)

    # load finetuning model
    diffusion = construct_diffusion_model(inputs=train_data, 
                                          normalizer_type='standard', 
                                          disable_terminal_norm=True)
    dp1_diffusion = construct_diffusion_model(inputs=train_data, 
                                            normalizer_type='standard', 
                                            disable_terminal_norm=True)
    dp10_diffusion = construct_diffusion_model(inputs=train_data, 
                                            normalizer_type='standard', 
                                            disable_terminal_norm=True)

    ckpt = torch.load(ckpt_path)
    dp1_ckpt = torch.load(dp1_ckpt_path)
    dp10_ckpt = torch.load(dp10_ckpt_path)
    diffusion.load_state_dict(ckpt['model'])
    dp1_ckpt_model = {}
    for key, value in dp1_ckpt['model'].items():
        new_key = key.replace('_module.', '')
        dp1_ckpt_model[new_key] = value
    dp1_diffusion.load_state_dict(dp1_ckpt_model)
    dp10_ckpt_model = {}
    for key, value in dp10_ckpt['model'].items():
        new_key = key.replace('_module.', '')
        dp10_ckpt_model[new_key] = value
    dp10_diffusion.load_state_dict(dp10_ckpt_model)

    diffusion = diffusion.cuda()
    dp1_diffusion = dp1_diffusion.cuda()
    dp10_diffusion = dp10_diffusion.cuda()
    diffusion.eval()
    dp1_diffusion.eval()
    dp10_diffusion.eval()

    # maze2d 12
    config.sigma = [0.13, 0.12, 0.11, 0.1,]

    # # kitchen 41
    # config.sigma = [0.13, 0.12, 0.11, 0.1,]

    # 4
    # config.sigma = [0.003, 0.001, 0.0008]

    with torch.no_grad():
        for sigma in config.sigma:
            test_loss = 0
            train_loss = 0
            dp1_test_loss = 0
            dp1_train_loss = 0
            dp10_test_loss = 0
            dp10_train_loss = 0

            for _ in range(config.repeat):
                test_loss += diffusion.forward_mia(test_data, sigma=sigma)
                train_loss += diffusion.forward_mia(train_data, sigma=sigma)
                dp1_test_loss += dp1_diffusion.forward_mia(test_data, sigma=sigma)
                dp1_train_loss += dp1_diffusion.forward_mia(train_data, sigma=sigma)
                dp10_test_loss += dp10_diffusion.forward_mia(test_data, sigma=sigma)
                dp10_train_loss += dp10_diffusion.forward_mia(train_data, sigma=sigma)
            test_loss /= config.repeat
            train_loss /= config.repeat
            dp1_test_loss /= config.repeat
            dp1_train_loss /= config.repeat
            dp10_test_loss /= config.repeat
            dp10_train_loss /= config.repeat
            test_label = torch.ones_like(test_loss)
            train_label = torch.zeros_like(train_loss)
            print(test_loss.sum(), train_loss.sum())
            print(dp1_test_loss.sum(), dp1_train_loss.sum())
            print(dp10_test_loss.sum(), dp10_train_loss.sum())

            results = torch.cat([test_loss, train_loss]).cpu().detach().numpy()
            dp1_results = torch.cat([dp1_test_loss, dp1_train_loss]).cpu().detach().numpy()
            dp10_results = torch.cat([dp10_test_loss, dp10_train_loss]).cpu().detach().numpy()
            labels = torch.cat([test_label, train_label]).cpu().detach().numpy().astype(int)

            tpr_at_low_fpr_1 = get_metrics(labels, results, fixed_fpr=0.1)
            tpr_at_low_fpr_2 = get_metrics(labels, results, fixed_fpr=0.01)
            tpr_at_low_fpr_3 = get_metrics(labels, results, fixed_fpr=0.001)
            tpr_at_low_fpr_4 = get_metrics(labels, results, fixed_fpr=0.2)

            print("nondp sigma: %.3f TPR@10%%FPR: %.3f TPR@1%%FPR: %.4f TPR@0.1%%FPR: %.5f TPR@20%%FPR: %.6f" % (sigma, tpr_at_low_fpr_1, tpr_at_low_fpr_2, tpr_at_low_fpr_3, tpr_at_low_fpr_4))

            tpr_at_low_fpr_1 = get_metrics(labels, dp1_results, fixed_fpr=0.1)
            tpr_at_low_fpr_2 = get_metrics(labels, dp1_results, fixed_fpr=0.01)
            tpr_at_low_fpr_3 = get_metrics(labels, dp1_results, fixed_fpr=0.001)
            tpr_at_low_fpr_4 = get_metrics(labels, dp1_results, fixed_fpr=0.2)

            print("dp1 sigma: %.3f TPR@10%%FPR: %.3f TPR@1%%FPR: %.4f TPR@0.1%%FPR: %.5f TPR@20%%FPR: %.6f" % (sigma, tpr_at_low_fpr_1, tpr_at_low_fpr_2, tpr_at_low_fpr_3, tpr_at_low_fpr_4))

            tpr_at_low_fpr_1 = get_metrics(labels, dp10_results, fixed_fpr=0.1)
            tpr_at_low_fpr_2 = get_metrics(labels, dp10_results, fixed_fpr=0.01)
            tpr_at_low_fpr_3 = get_metrics(labels, dp10_results, fixed_fpr=0.001)
            tpr_at_low_fpr_4 = get_metrics(labels, dp10_results, fixed_fpr=0.2)

            print("dp10 sigma: %.3f TPR@10%%FPR: %.3f TPR@1%%FPR: %.4f TPR@0.1%%FPR: %.5f TPR@20%%FPR: %.6f" % (sigma, tpr_at_low_fpr_1, tpr_at_low_fpr_2, tpr_at_low_fpr_3, tpr_at_low_fpr_4))

    return train_data, test_data, diffusion

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, default="halfcheetah-medium-replay-v2")
    parser.add_argument("--pretraining_rate", type=float, default=1.0)
    parser.add_argument("--finetuning_rate", type=float, default=0.8)
    parser.add_argument("--nondp_weight", type=str, default="1e3data_300epoch_finetuning_without_dp-model-299.pt")
    # parser.add_argument("--nondp_weight", type=str, default="pretraining-model-9.pt")
    parser.add_argument("--dp1_weight", type=str, default="finetuning_dp1.0-model-4.pt")
    parser.add_argument("--dp10_weight", type=str, default="finetuning_dp10.0-model-4.pt")
    parser.add_argument("--sigma", type=list, default=[0.05, 0.01])
    parser.add_argument("--repeat", type=int, default=64)
    parser.add_argument("--sample_num", type=int, default=10000)
    args = parser.parse_args()

    get_data_and_model(config=args)