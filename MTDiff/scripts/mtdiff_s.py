import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import diffuser.utils as utils
import numpy as np
import pdb
import copy
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from tqdm import tqdm

from opacus import GradSampleModule, PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from opacus.accountants import RDPAccountant

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'maze2d-medium-dense-v1'
    config: str = 'config.locomotion'

def setup(device):
    torch.cuda.set_device(device)

def cleanup():
    dist.destroy_process_group()

def cycle(dl):
    while True:
        for data in dl:
            yield data

def main(args):
    device = args.device
    setup(device)
    # torch.cuda.set_device(device)

    if args.finetune:
        args.privacy = True
        args.n_train_steps = 2e5
    else:
        args.privacy = False
        args.n_train_steps = 5e5

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

    n_epochs = int(args.n_train_steps // args.n_steps_per_epoch)

    if args.finetune:
        trainer.load_for_finetune(args.checkpoint_path)

        trainer.privacy_engine = PrivacyEngine()
        # print("before ModuleValidator, model is:\n", self.model)
        trainer.model = ModuleValidator.fix(trainer.model)
        # print("after ModuleValidator, model is:\n", self.model)
            
        trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=trainer.train_lr)

        trainer.model, trainer.optimizer, trainer.raw_dataloader = trainer.privacy_engine.make_private_with_epsilon(
            module=trainer.model,
            optimizer=trainer.optimizer,
            data_loader=trainer.raw_dataloader,
            target_epsilon=args.target_epsilon,
            target_delta=args.target_delta,
            epochs=n_epochs,
            max_grad_norm=trainer.max_grad_norm,
        )
        # pdb.set_trace()
        trainer.ema_model = copy.deepcopy(trainer.model)

        trainer.dataloader = cycle(trainer.raw_dataloader)


    utils.report_parameters(model)

    # Training loop    
    for epoch in range(n_epochs):
        print(f'Epoch {epoch + 1} / {n_epochs} | {args.savepath}')
        trainer.train(n_train_steps=args.n_steps_per_epoch)
    trainer.save(int(args.n_train_steps))

if __name__ == "__main__":
    args = Parser().parse_args('diffusion', full_init=True)
    main(args)