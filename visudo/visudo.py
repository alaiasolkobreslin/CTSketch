import argparse
import time

import yaml
from torch.utils.data import DataLoader
import torch
import wandb

from anesi_visudo import ViSudoModel
from parse_data import get_datasets
from inference_models import NoPossibleActionsException
from torcheval.metrics import BinaryAUROC


SWEEP = False

def val_sudo(x, label, model, device):
    try:
        test_result = model.test(x, label, None)
    except NoPossibleActionsException:
        print("No possible actions during testing")
        test_result = val_sudo(x, label, model, device)
    acc = test_result[0].item()
    acc_prior = test_result[1].item()
    acc_clauses = test_result[2].item()
    prior_y = test_result[3]
    return acc, acc_prior, acc_clauses, prior_y

if __name__ == '__main__':
  for i in [10, 9, 8, 7]:
    config = {
        "N": 9,
        "DEBUG": False,
        "amt_samples": 500,
        "batch_size": 20,
        "batch_size_test": 200,
        "dirichlet_init": 0.02,
        "dirichlet_iters": 18,
        "dirichlet_L2": 250000,
        "dirichlet_lr": 0.00285,
        "epochs": 5000,
        "encoding": "pair",
        "fixed_alpha": None,
        "hidden_size": 100,
        "K_beliefs": 1600,
        "layers": 2,
        "log_per_epoch": 1,
        "P_source": "both",
        "percept_loss_pref": 1.0,
        "perception_lr": 0.000554,
        "perception_loss": "log-q",
        "policy": "off",
        "predict_only": True,
        "pretrain_epochs": 50,
        "prune": False,
        "q_lr": 0.00301,
        "q_loss": "mse",
        "split": 11,
        "test": True,
        "train_negatives": True,
        "use_cuda": True,
        "verbose": 0,
        "val_freq": 10,
    }
    config["split"] = i

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    known, unknown = parser.parse_known_args()
    config_file = known.config
    if config_file is not None:
        with open(config_file, 'r') as f:
            config.update(yaml.safe_load(f))

        run = wandb.init(config=config, project="visudo", entity="seewonchoi")
        config = wandb.config
        print(config)
    elif SWEEP:
        # TODO: I don't get how it's supposed to know what yaml file to open here.
        with open("visudo/sweeps/sweep4.yaml", 'r') as f:
            sweep_config = yaml.load(f, Loader=yaml.FullLoader)

        run = wandb.init(config=sweep_config)
        config.update(wandb.config)
        print(config)
    else:
        run = wandb.init(
            project=f'anesi-visudo{config["N"]}',
            name=f'{config["split"]}',
            config=config,
        )

    # Check for available GPUs
    use_cuda = config["use_cuda"] and torch.cuda.is_available()
    device = torch.device(1 if use_cuda else 'cpu')

    model = ViSudoModel(config).to(device)
    train, val, test = get_datasets(config["split"], dimension=config["N"], use_negative_train=config["train_negatives"])

    if config["test"]:
        val = test

    train_loader = DataLoader(train, config["batch_size"], shuffle=True)
    val_loader = DataLoader(val, config["batch_size_test"], False)

    log_iterations = len(train_loader) // config["log_per_epoch"]

    if config["DEBUG"]:
        torch.autograd.set_detect_anomaly(True)


    model.perception.requires_grad_(False)

    for epoch in range(config["epochs"]):
        cum_loss_percept = 0
        cum_loss_q = 0

        start_epoch_time = time.time()

        if epoch > config["pretrain_epochs"]:
            model.perception.requires_grad_(True)

        for i, batch in enumerate(train_loader):
            grid, label = batch

            x = grid.to(device)
            label = label.to(device)
            try:
                trainresult = model.train_all(x, label)
                loss_percept = trainresult.percept_loss
                loss_nrm = trainresult.q_loss
            except NoPossibleActionsException:
                print("No possible actions during training")
                continue

            cum_loss_percept += loss_percept.item()
            cum_loss_q += loss_nrm.item()

            if (i + 1) % log_iterations == 0:
                avg_alpha = torch.nn.functional.softplus(model.alpha).mean()

                log_q_weight = torch.sigmoid((config['percept_loss_pref'] * avg_alpha).log()).item()
                avg_alpha = avg_alpha.item()

                print(f"epoch: {epoch} "
                      f"actor: {cum_loss_percept / log_iterations:.4f} "
                      f"nrm: {cum_loss_q / log_iterations:.4f} " 
                      f"avg_alpha: {avg_alpha:.4f} ",
                      f"log_q_weight: {log_q_weight:.4f} ",)
                wandb.log({
                    # "epoch": epoch,
                    "percept_loss": cum_loss_percept / log_iterations,
                    "nrm_loss": cum_loss_q / log_iterations,
                    "avg_alpha": avg_alpha,
                    # "log_q_weight": log_q_weight,
                })
                cum_loss_percept = 0
                cum_loss_q = 0

        end_epoch_time = time.time()

        if epoch % config["val_freq"] != 0:
            continue

        if config['test']:
            print("----- TESTING -----")
        else:
            print("----- VALIDATING -----")
        val_acc = 0.
        val_acc_prior = 0.
        val_acc_clauses = 0.
        prior_y = []
        labels = []
        for i, batch in enumerate(val_loader):
            grid, label = batch
            label = label.to(device)
            test_result = val_sudo(grid.to(device), label, model, device)
            val_acc += test_result[0]
            val_acc_prior += test_result[1]
            val_acc_clauses += test_result[2]
            prior_y += [test_result[3]]
            labels += [label]

        val_accuracy = val_acc / len(val_loader)
        val_accuracy_prior = val_acc_prior / len(val_loader)
        val_accuracy_clauses = val_acc_clauses / len(val_loader)

        all_labels = torch.cat(labels, dim=0)
        all_prior_y = torch.cat(prior_y, dim=0)
        metric = BinaryAUROC()
        metric.update(all_prior_y, all_labels)
        val_auroc = metric.compute().item()

        epoch_time = end_epoch_time - start_epoch_time
        test_time = time.time() - end_epoch_time

        prefix = 'Test' if config['test'] else 'Val'

        print(f"{prefix} accuracy: {val_accuracy_prior} {prefix} clauses acc: {val_accuracy_clauses}"
                f" {prefix} auroc: {val_auroc} {prefix}"
              f" Epoch time: {epoch_time} {prefix} time: {test_time}")

        wdb_prefix = 'test' if config['test'] else 'val'
        wandb.log({
            # "epoch": epoch,
            f"{wdb_prefix}_accuracy": val_accuracy,
            f"{wdb_prefix}_accuracy_prior": val_accuracy_prior,
            f"{wdb_prefix}_accuracy_clauses": val_accuracy_clauses,
            f"{wdb_prefix}_auroc": val_auroc,
            f"{wdb_prefix}_time": test_time,
            "epoch_time": epoch_time,
        })
    run.finish()