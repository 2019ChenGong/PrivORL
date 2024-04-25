import torch
import numpy as np
import argparse
import gym
import d4rl
import os
from typing import Optional, List


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
    results_folder = f"./alter_curiosity_driven_results_{config.dataset}_{config.pretraining_rate}"
    ckpt_path = os.path.join(results_folder, "pretraining-model-9.pt")
    dp_ckpt_path = os.path.join(results_folder, "finetuning-model-4.pt")
    syndata_path = os.path.join(results_folder, f"cleaned_{config.dataset}_samples_{1e6}_10dp_{config.finetuning_rate}.npz")

    # load orginal and syn data
    env = gym.make(config.dataset)
    original_data = d4rl.qlearning_dataset(env)
    original_data = make_inputs(original_data)
    original_data = torch.from_numpy(original_data).float().cuda()

    syn_data = np.load(syndata_path)
    syn_data = make_inputs(syn_data)
    syn_data = torch.from_numpy(syn_data).float()

    syn_data = syn_data[:len(original_data)].cuda()

    print(original_data.shape, syn_data.shape)

    # load finetuning model
    diffusion = construct_diffusion_model(inputs=original_data, 
                                          normalizer_type='standard', 
                                          disable_terminal_norm=True)
    dp_diffusion = construct_diffusion_model(inputs=original_data, 
                                            normalizer_type='standard', 
                                            disable_terminal_norm=True)

    ckpt = torch.load(ckpt_path)
    dp_ckpt = torch.load(dp_ckpt_path)
    diffusion.load_state_dict(ckpt['model'])
    dp_ckpt_model = {}
    for key, value in dp_ckpt['model'].items():
        new_key = key.replace('_module.', '')
        dp_ckpt_model[new_key] = value
    dp_diffusion.load_state_dict(dp_ckpt_model)

    diffusion = diffusion.cuda()
    dp_diffusion = dp_diffusion.cuda()
    diffusion.eval()
    dp_diffusion.eval()

    with torch.no_grad():
        for sigma in config.sigma:
            syn_loss = 0
            original_loss = 0
            dp_syn_loss = 0
            dp_original_loss = 0

            for _ in range(config.repeat):
                syn_loss += diffusion.forward_mia(syn_data, sigma=sigma)
                original_loss += diffusion.forward_mia(original_data, sigma=sigma)
                dp_syn_loss += dp_diffusion.forward_mia(syn_data, sigma=sigma)
                dp_original_loss += dp_diffusion.forward_mia(original_data, sigma=sigma)
            syn_loss /= config.repeat
            original_loss /= config.repeat
            dp_syn_loss /= config.repeat
            dp_original_loss /= config.repeat
            syn_label = torch.ones_like(syn_loss)
            original_label = torch.zeros_like(original_loss)
            print(syn_loss.sum(), original_loss.sum())
            print(dp_syn_loss.sum(), dp_original_loss.sum())

            results = torch.cat([syn_loss, original_loss]).cpu().detach().numpy()
            dp_results = torch.cat([dp_syn_loss, dp_original_loss]).cpu().detach().numpy()
            labels = torch.cat([syn_label, original_label]).cpu().detach().numpy().astype(int)

            tpr_at_low_fpr_1 = get_metrics(labels, results, fixed_fpr=0.1)
            tpr_at_low_fpr_2 = get_metrics(labels, results, fixed_fpr=0.01)
            tpr_at_low_fpr_3 = get_metrics(labels, results, fixed_fpr=0.001)
            tpr_at_low_fpr_4 = get_metrics(labels, results, fixed_fpr=0.0001)

            print("sigma: %.3f TPR@10%%FPR: %.2f TPR@1%%FPR: %.2f TPR@0.1%%FPR: %.2f TPR@0.01%%FPR: %.2f" % (sigma, tpr_at_low_fpr_1, tpr_at_low_fpr_2, tpr_at_low_fpr_3, tpr_at_low_fpr_4))

            tpr_at_low_fpr_1 = get_metrics(labels, dp_results, fixed_fpr=0.1)
            tpr_at_low_fpr_2 = get_metrics(labels, dp_results, fixed_fpr=0.01)
            tpr_at_low_fpr_3 = get_metrics(labels, dp_results, fixed_fpr=0.001)
            tpr_at_low_fpr_4 = get_metrics(labels, dp_results, fixed_fpr=0.0001)

            print("dpsigma: %.3f TPR@10%%FPR: %.2f TPR@1%%FPR: %.2f TPR@0.1%%FPR: %.2f TPR@0.01%%FPR: %.2f" % (sigma, tpr_at_low_fpr_1, tpr_at_low_fpr_2, tpr_at_low_fpr_3, tpr_at_low_fpr_4))

    return original_data, syn_data, diffusion, dp_diffusion

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, default="kitchen-complete-v0")
    parser.add_argument("--pretraining_rate", type=float, default=1.0)
    parser.add_argument("--finetuning_rate", type=float, default=0.8)
    parser.add_argument("--nondp_weight", type=str, default="retraining-model-9.pt")
    parser.add_argument("--dp_weight", type=str, default="finetuning-model-4.pt")
    parser.add_argument("--sigma", type=list, default=[0.01])
    parser.add_argument("--repeat", type=int, default=32)
    args = parser.parse_args()

    original_data, syn_data, diffusion, dpdiffusion = get_data_and_model(config=args)