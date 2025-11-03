import os
import pickle
import glob
import torch
import pdb
import copy

from collections import namedtuple

from opacus import GradSampleModule, PrivacyEngine
from opacus.validators import ModuleValidator
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP

DiffusionExperiment = namedtuple('Diffusion', 'dataset renderer model trainermodel diffusion ema trainer epoch')
mtdtExperiment = namedtuple('mtdtExperiment', 'dataset model ema trainer epoch')


def cycle(dl):
    while True:
        for data in dl:
            yield data

def mkdir(savepath):
    """
        returns `True` iff `savepath` is created
    """
    if not os.path.exists(savepath):
        os.makedirs(savepath)
        return True
    else:
        return False

def get_latest_epoch(loadpath):
    states = glob.glob1(os.path.join(*loadpath), 'state_*')
    latest_epoch = -1
    for state in states:
        epoch = int(state.replace('state_', '').replace('.pt', ''))
        latest_epoch = max(epoch, latest_epoch)
    return latest_epoch

def load_config(*loadpath):
    loadpath = os.path.join(*loadpath)
    config = pickle.load(open(loadpath, 'rb'))
    print(f'[ utils/serialization ] Loaded config from {loadpath}')
    #print(config)
    return config

def load_diffusion(*loadpath, epoch='latest', device='cuda:0', seed=None, sample=False, args):
    dataset_config = load_config(*loadpath, 'dataset_config.pkl')
    model_config = load_config(*loadpath, 'model_config.pkl')
    diffusion_config = load_config(*loadpath, 'diffusion_config.pkl')
    trainer_config = load_config(*loadpath, 'trainer_config.pkl')

    ## @TODO : remove results folder from within trainer class
    trainer_config._dict['results_folder'] = os.path.join(*loadpath)

    dataset = dataset_config(seed=seed)
    #renderer = render_config()
    renderer = None
    model = model_config()
    diffusion = diffusion_config(model)
    
    trainer = trainer_config(diffusion, dataset, renderer, sample)
    print("in serialization.py, sample is ", sample)

    if epoch == 'latest':
        epoch = get_latest_epoch(loadpath)

    print(f'\n[ utils/serialization ] Loading model epoch: {epoch}\n')

    if args.privacy:
        trainer.privacy_engine = PrivacyEngine()
        # print("before ModuleValidator, model is:\n", self.model)
        trainer.model = ModuleValidator.fix(trainer.model)
        # print("after ModuleValidator, model is:\n", self.model)

        # trainer.model = DPDDP(trainer.model)
            
        trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=trainer.train_lr)
        trainer.model, trainer.optimizer, trainer.raw_dataloader = trainer.privacy_engine.make_private_with_epsilon_j(
            module=trainer.model,
            optimizer=trainer.optimizer,
            data_loader=trainer.raw_dataloader,
            target_epsilon=args.target_epsilon,
            target_delta=args.target_delta,
            epochs=int(args.n_train_steps // args.n_steps_per_epoch),
            max_grad_norm=trainer.max_grad_norm,
            avg_fragments_per_trajectory=1
        )
        trainer.dataloader = cycle(trainer.raw_dataloader)
        trainer.ema_model = copy.deepcopy(trainer.model)
    
    trainer.load_for_sample(epoch)

    return DiffusionExperiment(dataset, renderer, model, trainer.model, diffusion, trainer.ema_model, trainer, epoch)


def load_mtdt(*loadpath, epoch='latest', device='cuda:0', seed=None):
    dataset_config = load_config(*loadpath, 'dataset_config.pkl')
    model_config = load_config(*loadpath, 'model_config.pkl')
    trainer_config = load_config(*loadpath, 'trainer_config.pkl')

    ## remove absolute path for results loaded from azure
    ## @TODO : remove results folder from within trainer class
    trainer_config._dict['results_folder'] = os.path.join(*loadpath)

    dataset = dataset_config(seed=seed)
    model = model_config()
    trainer = trainer_config(model, dataset)
    if epoch == 'latest':
        epoch = get_latest_epoch(loadpath)

    print(f'\n[ utils/serialization ] Loading model epoch: {epoch}\n')

    trainer.load(epoch)
    return mtdtExperiment(dataset, model, trainer.model, trainer, epoch)
def load_model(*loadpath, dataset=None, epoch='latest', device='cuda:0', seed=None):
    model_config = load_config(*loadpath, 'model_config.pkl')
    trainer_config = load_config(*loadpath, 'trainer_config.pkl')
    ## @TODO : remove results folder from within trainer class
    trainer_config._dict['results_folder'] = os.path.join(*loadpath)

    model = model_config()
    trainer = trainer_config(model, dataset)
    if epoch == 'latest':
        epoch = get_latest_epoch(loadpath)

    print(f'\n[ utils/serialization ] Loading model epoch: {epoch}\n')

    trainer.load(epoch)
    return trainer.model
def check_compatibility(experiment_1, experiment_2):
    '''
        returns True if `experiment_1 and `experiment_2` have
        the same normalizers and number of diffusion steps
    '''
    normalizers_1 = experiment_1.dataset.normalizer.get_field_normalizers()
    normalizers_2 = experiment_2.dataset.normalizer.get_field_normalizers()
    for key in normalizers_1:
        norm_1 = type(normalizers_1[key])
        norm_2 = type(normalizers_2[key])
        assert norm_1 == norm_2, \
            f'Normalizers should be identical, found {norm_1} and {norm_2} for field {key}'

    n_steps_1 = experiment_1.diffusion.n_timesteps
    n_steps_2 = experiment_2.diffusion.n_timesteps
    assert n_steps_1 == n_steps_2, \
        ('Number of timesteps should match between diffusion experiments, '
        f'found {n_steps_1} and {n_steps_2}')
