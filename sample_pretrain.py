import argparse
import pathlib
import os
import gym
import numpy as np
import torch
import gin
import d4rl

from synther.diffusion.elucidated_diffusion import Trainer
from synther.diffusion.norm import MinMaxNormalizer
from synther.diffusion.utils import split_diffusion_samples, construct_diffusion_model
from synther.diffusion.delete_nan import remove_errors


@gin.configurable
class SimpleDiffusionGenerator:
    def __init__(
        self,
        env: gym.Env,
        ema_model,
        num_sample_steps: int = 128,
        sample_batch_size: int = 100000,
    ):
        self.env = env
        self.diffusion = ema_model
        self.diffusion.eval()
        self.clamp_samples = isinstance(self.diffusion.normalizer, MinMaxNormalizer)
        self.num_sample_steps = num_sample_steps
        self.sample_batch_size = sample_batch_size
        print(f"[INFO] Sampling with {self.num_sample_steps} steps, batch size {self.sample_batch_size}")

    def sample(self, num_samples: int):
        assert num_samples % self.sample_batch_size == 0, "num_samples must be a multiple of sample_batch_size"
        num_batches = num_samples // self.sample_batch_size

        observations, actions, rewards, next_observations, terminals = [], [], [], [], []
        for i in range(num_batches):
            print(f"[INFO] Generating batch {i + 1}/{num_batches}")
            sampled_outputs = self.diffusion.sample(
                batch_size=self.sample_batch_size,
                num_sample_steps=self.num_sample_steps,
                clamp=self.clamp_samples,
            ).cpu().numpy()

            transitions = split_diffusion_samples(sampled_outputs, self.env)
            if len(transitions) == 4:
                obs, act, rew, next_obs = transitions
                terminal = np.zeros_like(next_obs[:, 0])
            else:
                obs, act, rew, next_obs, terminal = transitions

            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            next_observations.append(next_obs)
            terminals.append(terminal)

        return (
            np.concatenate(observations, axis=0),
            np.concatenate(actions, axis=0),
            np.concatenate(rewards, axis=0),
            np.concatenate(next_observations, axis=0),
            np.concatenate(terminals, axis=0),
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="maze2d-medium-dense-v1")
    parser.add_argument("--gin_config_files", nargs="*", type=str, default=["config/resmlp_denoiser.gin"])
    parser.add_argument("--gin_params", nargs="*", type=str, default=[])
    parser.add_argument("--results_folder", type=str, default="./results")
    parser.add_argument("--load_path", type=str, default="/bigtemp/fzv6en/SynthER/results_maze2d-large-dense-v1_0.3_prv/pretraining-model-9.pt")
    parser.add_argument("--use_gpu", action="store_true", default=True)
    parser.add_argument("--num_sample_steps", type=int, default=128)
    parser.add_argument("--sample_batch_size", type=int, default=100000)
    parser.add_argument("--save_num_samples", type=int, default=int(1e6))
    parser.add_argument("--save_file_name", type=str, default="pretrain_samples.npz")

    args = parser.parse_args()

    args.load_path = f"/bigtemp/fzv6en/SynthER/results_{args.dataset}_0.3_prv/pretraining-model-4.pt"
    args.results_folder = args.load_path.rsplit('/', 1)[0]
    print(f"[INFO] Load path set to {args.load_path}")
    print(f"[INFO] Results folder set to {args.results_folder}")

    # Load gin config
    for config_file in args.gin_config_files:
        if not os.path.exists(config_file):
            print(f"[ERROR] Config file {config_file} not found.")
            exit(1)
    gin.parse_config_files_and_bindings(args.gin_config_files, args.gin_params)

    torch.manual_seed(0)
    np.random.seed(0)
    if args.use_gpu:
        torch.cuda.manual_seed(0)

    # Create environment
    env = gym.make(args.dataset)
    results_folder = pathlib.Path(args.results_folder)
    results_folder.mkdir(parents=True, exist_ok=True)

    # Load pretrained model
    print("[INFO] Constructing diffusion model...")

    obs_dim = env.observation_space.shape[0]
    if hasattr(env.action_space, "shape") and env.action_space.shape is not None:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = 1

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

    # Start sampling
    generator = SimpleDiffusionGenerator(
        env=env,
        ema_model=trainer.ema.ema_model,
        num_sample_steps=args.num_sample_steps,
        sample_batch_size=args.sample_batch_size,
    )

    print("[INFO] Sampling transitions...")
    observations, actions, rewards, next_observations, terminals = generator.sample(
        num_samples=args.save_num_samples,
    )

    # Save results
    output_path = results_folder / args.save_file_name
    np.savez_compressed(
        output_path,
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        terminals=terminals,
    )
    print(f"[INFO] Samples saved to {output_path}")

    # Clean potential NaNs
    remove_errors(results_folder, args.save_file_name)
    print("[INFO] NaN check completed.")
