import argparse
import pathlib
import os
import gin
import torch
import numpy as np
import gym
import d4rl

from synther.diffusion.elucidated_diffusion import Trainer
from synther.diffusion.norm import MinMaxNormalizer
from synther.diffusion.utils import (
    construct_diffusion_model,
    load_custom_dataset,
    split_diffusion_samples_epicare,
)
from synther.diffusion.delete_nan import remove_errors


@gin.configurable
class SimpleDiffusionGenerator:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        ema_model,
        num_sample_steps: int = 128,
        sample_batch_size: int = 100000,
        modelled_terminals: bool = True,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.diffusion = ema_model
        self.diffusion.eval()
        self.clamp_samples = isinstance(self.diffusion.normalizer, MinMaxNormalizer)
        self.num_sample_steps = num_sample_steps
        self.sample_batch_size = sample_batch_size
        self.modelled_terminals = modelled_terminals
        print(f"[INFO] Sampling using {self.num_sample_steps} steps, batch size {self.sample_batch_size}")

    def sample(self, num_samples: int):
        assert num_samples % self.sample_batch_size == 0, "num_samples must be a multiple of sample_batch_size"
        num_batches = num_samples // self.sample_batch_size
        all_obs, all_act, all_rew, all_next_obs, all_term = [], [], [], [], []

        for i in range(num_batches):
            print(f"[INFO] Generating batch {i + 1}/{num_batches}")
            sampled_outputs = self.diffusion.sample(
                batch_size=self.sample_batch_size,
                num_sample_steps=self.num_sample_steps,
                clamp=self.clamp_samples,
            ).cpu().numpy()

            transitions = split_diffusion_samples_epicare(
                sampled_outputs,
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                modelled_terminals=self.modelled_terminals,
            )

            if len(transitions) == 4:
                obs, act, rew, next_obs = transitions
                term = np.zeros_like(next_obs[:, 0])
            else:
                obs, act, rew, next_obs, term = transitions

            all_obs.append(obs)
            all_act.append(act)
            all_rew.append(rew)
            all_next_obs.append(next_obs)
            all_term.append(term)

        return (
            np.concatenate(all_obs, axis=0),
            np.concatenate(all_act, axis=0),
            np.concatenate(all_rew, axis=0),
            np.concatenate(all_next_obs, axis=0),
            np.concatenate(all_term, axis=0),
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="EpiCare/data/smart/train_seed_1.hdf5")
    parser.add_argument("--gin_config_files", nargs="*", type=str, default=["config/resmlp_denoiser.gin"])
    parser.add_argument("--gin_params", nargs="*", type=str, default=[])
    parser.add_argument("--load_path", type=str, default="results_epicare_0.3_rdp/pretraining-model-9.pt")
    parser.add_argument("--results_folder", type=str, default="results_epicare_0.3_rdp")
    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument("--num_sample_steps", type=int, default=128)
    parser.add_argument("--sample_batch_size", type=int, default=100000)
    parser.add_argument("--save_num_samples", type=int, default=int(1e6))
    parser.add_argument("--save_file_name", type=str, default="pretrain_samples.npz")
    args = parser.parse_args()

    # === Setup ===
    for config_file in args.gin_config_files:
        if not os.path.exists(config_file):
            print(f"[ERROR] Config file '{config_file}' not found.")
            exit(1)

    gin.parse_config_files_and_bindings(args.gin_config_files, args.gin_params)
    torch.manual_seed(0)
    np.random.seed(0)
    if args.use_gpu:
        torch.cuda.manual_seed(0)

    results_folder = pathlib.Path(args.results_folder)
    results_folder.mkdir(parents=True, exist_ok=True)

    # === Get dataset info ===
    dataset = load_custom_dataset(args.dataset)
    obs_dim = dataset["observations"].shape[1]
    action_dim = dataset["actions"].shape[1] if dataset["actions"].ndim > 1 else 1

    # === Construct and load pretrained model ===
    print("[INFO] Constructing diffusion model...")
    dummy_input = torch.zeros(10, obs_dim + action_dim + 1 + obs_dim + 1)
    diffusion = construct_diffusion_model(inputs=dummy_input)


    trainer = Trainer(
        dp_delta=1e-6,
        dp_epsilon=10.0,
        dp_max_grad_norm=1.0,
        dp_max_physical_batch_size=8192,
        dp_n_splits=4,
        load_checkpoint=True,
        load_path=args.load_path,
        curiosity_driven=False,
        curiosity_driven_rate=0.0,
        diffusion_model=diffusion,
        accountant="rdp",
        dataset=None,
        results_folder=args.results_folder,
    )

    print(f"[INFO] Loading pretrained checkpoint from {args.load_path}")
    trainer.load(milestone=trainer.train_num_steps)
    trainer.ema.to(trainer.accelerator.device)

    # === Sampling ===
    generator = SimpleDiffusionGenerator(
        obs_dim=obs_dim,
        action_dim=action_dim,
        ema_model=trainer.ema.ema_model,
        num_sample_steps=args.num_sample_steps,
        sample_batch_size=args.sample_batch_size,
        modelled_terminals=True,
    )

    print(f"[INFO] Start sampling {args.save_num_samples} transitions...")
    observations, actions, rewards, next_observations, terminals = generator.sample(
        num_samples=args.save_num_samples
    )

    save_path = results_folder / args.save_file_name
    np.savez_compressed(
        save_path,
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=terminals,
    )
    print(f"[INFO] Samples saved to {save_path}")

    remove_errors(results_folder, args.save_file_name)
    print("[INFO] NaN check completed.")