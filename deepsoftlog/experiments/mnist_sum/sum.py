from pathlib import Path
import torch

from deepsoftlog.data.dataloader import DataLoader
from deepsoftlog.embeddings.initialize_vector import SPECIAL_MODELS
from deepsoftlog.training import load_program, load_config
from deepsoftlog.training.logger import WandbLogger
from deepsoftlog.training.loss import nll_loss, get_optimizer
from deepsoftlog.training.trainer import Trainer

from deepsoftlog.embeddings.nn_models import SumFunctor, SumFunctorAll
from deepsoftlog.experiments.mnist_sum.dataset import mnist_sum_dataset, generate_data, SumDataset, MnistQueryDataset

_EXP_ROOT = Path(__file__).parent


def get_pretrain_dataloader(cfg):
    pretrain_dataset = SumDataset(1, 2500).randomly_mutate_output()
    return DataLoader(pretrain_dataset, batch_size=cfg['batch_size'])


def get_train_dataloader(cfg):
    train_dataset = mnist_sum_dataset(cfg['digits'], "training")
    train_dataset = train_dataset.random_subset(cfg['data_subset'])
    dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'])
    return dataloader


def get_test_dataloader(cfg):
    eval_dataset = mnist_sum_dataset(cfg['digits'], "test")
    return DataLoader(eval_dataset, batch_size=1, shuffle=False)

def get_validation_dataloader(cfg):
    eval_dataset = mnist_sum_dataset(cfg['digits'], "validation")
    return DataLoader(eval_dataset, batch_size=1, shuffle=False)


def get_val_dataloader(cfg):
    eval_dataset = mnist_sum_dataset(cfg['digits'], "val").randomly_mutate_output()
    return DataLoader(eval_dataset, batch_size=cfg['batch_size'], shuffle=False)


def train(config_name):
    cfg = load_config(config_name)
    if cfg['sum_symb']:
        if cfg['digits'] == 2:
            SPECIAL_MODELS[('plus', 2)] = SumFunctor
        elif cfg['digits'] == 4:
            SPECIAL_MODELS[('plus4', 4)] = SumFunctor
        elif cfg['digits'] == 16:
            SPECIAL_MODELS[('plus16', 16)] = SumFunctorAll
        elif cfg['digits'] == 64:
            SPECIAL_MODELS[('plus64', 64)] = SumFunctorAll
        elif cfg['digits'] == 256:
            SPECIAL_MODELS[('plus256', 256)] = SumFunctorAll
        elif cfg['digits'] == 1024:
            SPECIAL_MODELS[('plus1024', 1024)] = SumFunctorAll

    # Training
    generate_data(cfg['digits'])
    val_dataloader = get_val_dataloader(cfg)
    vlidation_dataloader = get_validation_dataloader(cfg)
    test_dataloader = get_test_dataloader(cfg)
    program = load_program(cfg, val_dataloader)
    optimizer = get_optimizer(program.get_store(), cfg)
    logger = WandbLogger(cfg)
    trainer = Trainer(program, get_train_dataloader, vlidation_dataloader, test_dataloader, nll_loss, optimizer, logger=logger)
    trainer.train(cfg, nb_workers=cfg['nb_workers'])
    trainer.eval(get_test_dataloader(cfg))


def eval(folder: str, digits=None):
    cfg = load_config(f"results/{folder}/config.yaml")
    if digits is not None:
        cfg['digits'] = digits
    if cfg['sum_symb']:
        if cfg['digits'] == 2:
            SPECIAL_MODELS[('plus', 2)] = SumFunctor
        elif cfg['digits'] == 4:
            SPECIAL_MODELS[('plus4', 4)] = SumFunctor
        elif cfg['digits'] == 16:
            SPECIAL_MODELS[('plus16', 16)] = SumFunctorAll
        elif cfg['digits'] == 64:
            SPECIAL_MODELS[('plus64', 64)] = SumFunctorAll
        elif cfg['digits'] == 256:
            SPECIAL_MODELS[('plus256', 256)] = SumFunctorAll
        elif cfg['digits'] == 1024:
            SPECIAL_MODELS[('plus1024', 1024)] = SumFunctorAll
    print("EVALING", cfg['name'], cfg['digits'])

    test_dataloader = get_test_dataloader(cfg)
    program = load_program(cfg, test_dataloader)
    state_dict = torch.load(f"results/{folder}/store.pt")
    program.store.load_state_dict(state_dict, strict=False)
    trainer = Trainer(program, None, None, None)

    print("Digit accuracy:")
    digit_dataset = MnistQueryDataset("test")
    trainer.eval(DataLoader(digit_dataset, batch_size=4, shuffle=False))
    print("Predicate accuracy:")
    trainer.eval(test_dataloader)


if __name__ == "__main__":
    train("deepsoftlog/experiments/mnist_sum/config.yaml")
