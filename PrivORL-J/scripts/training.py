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

# datasets_name dictionary (unchanged)
datasets_name = {
    "halfcheetah-medium-replay-v2": ['walker2d-full-replay-v2', 'halfcheetah-full-replay-v2', 'walker2d-medium-v2'],
    "walker2d-medium-replay-v2": ['halfcheetah-expert-v2', 'walker2d-full-replay-v2', 'halfcheetah-full-replay-v2'],
    "maze2d-open-dense-v0": ['maze2d-umaze-dense-v1', 'maze2d-medium-dense-v1', 'maze2d-large-dense-v1'],
    "maze2d-umaze-dense-v1": ['maze2d-open-dense-v0', 'maze2d-medium-dense-v1', 'maze2d-large-dense-v1'],
    "maze2d-medium-dense-v1": ['maze2d-open-dense-v0', 'maze2d-umaze-dense-v1', 'maze2d-large-dense-v1'],
    "maze2d-large-dense-v1": ['maze2d-open-dense-v0', 'maze2d-umaze-dense-v1', 'maze2d-medium-dense-v1'],
    "antmaze-umaze-v1": ['antmaze-medium-play-v1', 'antmaze-large-play-v1'],
    "antmaze-medium-play-v1": ['antmaze-umaze-v1', 'antmaze-large-play-v1'],
    "antmaze-large-play-v1": ['antmaze-medium-play-v1', 'antmaze-umaze-v1'],
    "kitchen-complete-v0": ["kitchen-partial-v0", "kitchen-mixed-v0"],
    "kitchen-partial-v0": ['kitchen-complete-v0', 'kitchen-mixed-v0'],
    "kitchen-mixed-v0": ["kitchen-complete-v0", "kitchen-partial-v0"]
}

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'maze2d-medium-dense-v1'
    config: str = 'config.locomotion'
    # 新增参数
    max_fragments_per_trajectory: int = 128

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

    if args.finetune:
        args.privacy = True
        args.n_train_steps = 4e5
        task_list = [args.dataset]
        double_samples = True
    else:
        args.privacy = False
        args.n_train_steps = 5e5
        task_list = datasets_name[args.dataset]
        double_samples = False

    dataset_config = utils.Config(
        args.loader,
        savepath=(args.savepath, 'dataset_config.pkl'),
        env=args.dataset,
        replay_dir_list=[],
        task_list=task_list,
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
        double_samples=double_samples
    )
    dataset = dataset_config()
    
    if double_samples:
        dataset.double_samples(n=25)
        dataset.calculate_fragments_info()

    print(f"[DEBUG] Number of trajectories (n_episodes): {dataset.n_episodes}")
    print(f"[DEBUG] Total fragments: {dataset.total_fragments}")
    print(f"[DEBUG] Dataset size (len(dataset)): {len(dataset)}")
    print(f"[DEBUG] Max fragments per trajectory for training: {args.max_fragments_per_trajectory}")

    observation_dim = dataset.observation_dim
    action_dim = dataset.action_dim
    model_config = utils.Config(
        args.model,
        savepath=(args.savepath, 'model_config.pkl'),
        horizon=args.horizon,
        transition_dim=observation_dim + 1,
        cond_dim=observation_dim,
        num_tasks=1,
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
        save_parallel=args.save_parallel,
        results_folder=args.savepath,
        bucket=args.bucket,
        n_reference=args.n_reference,
        trainer_device=device,
        horizon=args.horizon,
        privacy=args.privacy,
        noise_multiplier=args.noise_multiplier,
        max_grad_norm=args.max_grad_norm,
        checkpoint_path=args.checkpoint_path,
        curiosity_rate=args.curiosity_rate,
        max_fragments_per_trajectory=args.max_fragments_per_trajectory
    )

    # Instantiate model and diffusion
    model = model_config()
    diffusion = diffusion_config(model)
    renderer = None

    # Create trainer
    trainer = trainer_config(diffusion, dataset, renderer)

    trajectory_dataloader_length = len(trainer.raw_dataloader)  # trajectory num / batch_size
    
    avg_fragments_per_trajectory = dataset.total_fragments / dataset.n_episodes
    fragment_steps_per_epoch = trajectory_dataloader_length * avg_fragments_per_trajectory
    n_epochs = int(np.ceil(args.n_train_steps / fragment_steps_per_epoch))
    
    gradient_updates_per_epoch = trajectory_dataloader_length
    total_gradient_updates = n_epochs * gradient_updates_per_epoch

    if args.finetune and args.privacy:
        print(f"[INFO] Trajectory-based DP training")
        print(f"[INFO] Number of trajectories: {dataset.n_episodes}")
        print(f"[INFO] Batch size: {args.batch_size}")
        print(f"[INFO] Trajectory dataloader length: {trajectory_dataloader_length}")
        print(f"[INFO] Max fragments per trajectory: {args.max_fragments_per_trajectory}")
        print(f"[INFO] Fragment steps per epoch: {fragment_steps_per_epoch}")
        print(f"[INFO] Total fragment steps planned: {args.n_train_steps}")
        print(f"[INFO] Epochs needed: {n_epochs}")
        print(f"[INFO] Gradient updates per epoch: {gradient_updates_per_epoch}")
        print(f"[INFO] Total gradient updates: {total_gradient_updates}")

    if args.finetune:
        trainer.load_for_finetune(args.checkpoint_path)
        trainer.privacy_engine = PrivacyEngine(accountant=args.accountant)
        trainer.model = ModuleValidator.fix(trainer.model)
        trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=trainer.train_lr)

        trainer.model, trainer.optimizer, trainer.raw_dataloader = trainer.privacy_engine.make_private_with_epsilon(
            module=trainer.model,
            optimizer=trainer.optimizer,
            data_loader=trainer.raw_dataloader,
            target_epsilon=args.target_epsilon,
            target_delta=args.target_delta,
            epochs=n_epochs,
            max_grad_norm=trainer.max_grad_norm,
            avg_fragments_per_trajectory=avg_fragments_per_trajectory
        )
        
        trainer.ema_model = copy.deepcopy(trainer.model)
        trainer.dataloader = cycle(trainer.raw_dataloader)

    utils.report_parameters(model)

    total_fragment_steps_completed = 0
    
    for epoch in range(n_epochs):
        print(f'Epoch {epoch + 1} / {n_epochs} | {args.savepath}')
        
        remaining_steps = args.n_train_steps - total_fragment_steps_completed
        steps_this_epoch = min(fragment_steps_per_epoch, remaining_steps)
        
        print(f"Fragment steps this epoch: {steps_this_epoch}")
        print(f"Total fragment steps completed: {total_fragment_steps_completed}")
        
        completed_steps = trainer.train(n_train_steps=steps_this_epoch)
        total_fragment_steps_completed += completed_steps
        
        print(f"Completed fragment steps this epoch: {completed_steps}")
        print(f"Total fragment steps completed so far: {total_fragment_steps_completed}")
        
        if total_fragment_steps_completed >= args.n_train_steps:
            print(f"Completed all {args.n_train_steps} fragment steps")
            break
        
    print(f"[INFO] Training completed. Saving final model as state_final.pt")
    trainer.save("final")

    print(f"[INFO] Training completed!")
    print(f"[INFO] Final fragment steps: {trainer.step}")
    print(f"[INFO] Final global steps: {trainer.global_step}")
    
    # if args.finetune and args.privacy:
    #     epsilon = trainer.privacy_engine.get_epsilon(args.target_delta)
    #     print(f"[INFO] Final privacy cost: ε={epsilon:.2f}, δ={args.target_delta}")
    #     print(f"[INFO] Total fragment steps completed: {total_fragment_steps_completed}")
    #     print(f"[INFO] Total gradient updates: {trainer.step}")

if __name__ == "__main__":
    args = Parser().parse_args('diffusion', full_init=True)
    main(args)