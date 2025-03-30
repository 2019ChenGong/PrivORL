import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import diffuser.utils as utils
import numpy as np
import pdb
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'maze2d-medium-dense-v1'
    config: str = 'config.locomotion'

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank, world_size, args):
    setup(rank, world_size)

    # Set device for this rank
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # Dataset setup
    dataset_config = utils.Config(
        args.loader,
        savepath=(args.savepath, 'dataset_config.pkl'),
        env=args.dataset,
        replay_dir_list=[],
        task_list=[args.dataset],
        horizon=args.horizon,
        normalizer=args.normalizer,
        preprocess_fns=args.preprocess_fns,
        use_padding=args.use_padding,
        max_path_length=args.max_path_length,
        discount=args.discount,
        termination_penalty=args.termination_penalty,
        normed=args.normed,
        meta_world=False,
        seq_length=5,
        use_d4rl=True,
    )
    dataset = dataset_config()

    # Model and diffusion setup
    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim
    model_config = utils.Config(
        args.model,
        savepath=(args.savepath, 'model_config.pkl'),
        horizon=args.horizon,
        transition_dim=observation_dim + 1,
        cond_dim=observation_dim,
        num_tasks=args.num_tasks,
        device=device,
        action_dim=action_dim,
    )
    diffusion_config = utils.Config(
        args.diffusion,
        savepath=(args.savepath, 'diffusion_config.pkl'),
        horizon=args.horizon,
        observation_dim=observation_dim,
        action_dim=action_dim,
        n_timesteps=args.n_diffusion_steps,
        loss_type=args.loss_type,
        clip_denoised=args.clip_denoised,
        predict_epsilon=args.predict_epsilon,
        action_weight=args.action_weight,
        loss_weights=args.loss_weights,
        loss_discount=args.loss_discount,
        device=device,
    )
    trainer_config = utils.Config(
        utils.AugTrainer,
        savepath=(args.savepath, 'trainer_config.pkl'),
        train_batch_size=args.batch_size,
        train_lr=args.learning_rate,
        gradient_accumulate_every=args.gradient_accumulate_every,
        ema_decay=args.ema_decay,
        sample_freq=args.sample_freq,
        save_freq=args.save_freq,
        label_freq=int(args.n_train_steps // args.n_saves),
        # label_freq=100,
        save_parallel=args.save_parallel,
        results_folder=args.savepath,
        bucket=args.bucket,
        n_reference=args.n_reference,
        trainer_device=device,
        horizon=args.horizon,
        privacy=args.privacy,
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=args.max_grad_norm,
        checkpoint_path=args.checkpoint_path
    )

    # Instantiate model and diffusion
    model = model_config()
    diffusion = diffusion_config(model)
    renderer = None

    # Wrap trainer with DDP compatibility
    trainer = trainer_config(diffusion, dataset, renderer)

    # Only rank 0 reports parameters and prints epoch info
    if rank == 0:
        utils.report_parameters(model)

    # Training loop
    n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)
    for i in range(n_epochs):
        if rank == 0:
            print(f'Epoch {i} / {n_epochs} | {args.savepath}')
        trainer.train(n_train_steps=args.n_steps_per_epoch)

    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    args = Parser().parse_args('diffusion', full_init=True) 
    # pdb.set_trace()
    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)