import argparse
import time

import yaml
from torch.utils.data import DataLoader
from experiments.mnist_op import MNISTSumNModel
import torch
import wandb
import random

from experiments.mnist_op.data import sum_n
from inference_models import NoPossibleActionsException

SWEEP = False

def test(x, label, label_digits, model, device):
    l_digits = []
    for i in range(len(label_digits)):
        l_digits += label_digits[i]
    label_digits_l = list(map(lambda d: d.to(device), l_digits))
    try:
        test_result = model.test(x, label, label_digits_l)
    except NoPossibleActionsException:
        print("No possible actions during testing")
        test_result = test(x, label, label_digits, model, device)
    acc = test_result[0].item()
    acc_prior = test_result[1].item()
    explain_acc = test_result[2].item()
    digit_acc = test_result[3].item()
    return acc, acc_prior, explain_acc, digit_acc

if __name__ == '__main__':
    config = {
        "use_cuda": True,
        "DEBUG": False,
        "N": 1,
        "y_encoding": "base10",
        "w_encoding": "base10",
        "model": "full",
        "test": True,
        "batch_size": 16,
        "batch_size_test": 16,
        "amount_samples": 600,
        "predict_only": True,
        "use_prior": True,
        "q_lr": 0.001,
        "q_loss": "mse",
        "policy": "off",
        "perception_lr": 0.001,
        "perception_loss": "log-q",
        "percept_loss_pref": 1.,
        "epochs": 200,
        "log_per_epoch": 1,
        "layers": 3,
        "hidden_size": 800,
        "prune": False,
        "dirichlet_init": 0.1,
        "dirichlet_lr": 0.01,
        "dirichlet_iters": 50,
        "dirichlet_L2": 900000,
        "K_beliefs": 2500,
        "op":'sum_n',
        "arity": 2,
        "pretrain_epochs": 0
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    known, unknown = parser.parse_known_args()
    config_file = known.config
    if config_file is not None:
        with open(config_file, 'r') as f:
            config.update(yaml.safe_load(f))

        run = wandb.init(config=config, project="mnist-add", entity="blackbox-learning")
        config = wandb.config
        print(config)
    elif SWEEP:
        # TODO: I don't get how it's supposed to know what yaml file to open here.
        with open("anesi/experiments/mnist_op/sweeps/sweep.yaml", 'r') as f:
            sweep_config = yaml.load(f, Loader=yaml.FullLoader)

        run = wandb.init(config=sweep_config)
        config.update(wandb.config)
        print(config)
    else:
        run = wandb.init(
            project=f"anesi-sum{config['arity']}",
            config=config,
        )
        print(config)

    # Check for available GPUs
    use_cuda = config["use_cuda"] and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    op = None
    model = None
    if config["op"] == "sum_n":
        op = sum_n
        model = MNISTSumNModel(config).to(device)

    if config["test"]:
        train_set = op(config["N"], config["arity"], "train")
        val_set = op(config["N"], config["arity"], "test")
    else:
        train_set = op(config["N"], config["arity"], "train")
        val_set = op(config["N"], config["arity"], "val")

    train_loader = DataLoader(train_set, config["batch_size"], False)
    val_loader = DataLoader(val_set, config["batch_size_test"], False)

    log_iterations = len(train_loader) // config["log_per_epoch"]

    if config["DEBUG"]:
        torch.autograd.set_detect_anomaly(True)
    
    model.perception.requires_grad_(False)
    
    for epoch in range(config["epochs"]):
        
        if epoch > config["pretrain_epochs"]:
            model.requires_grad_(True)

        print("----------------------------------------")
        print("NEW EPOCH", epoch)
        cum_loss_percept = 0
        cum_loss_nrm = 0
        train_acc = 0
        train_acc_prior = 0
        train_explain_acc = 0
        train_digit_acc = 0

        start_epoch_time = time.time()

        for i, batch in enumerate(train_loader):
            # label_digits is ONLY EVER to be used during testing!!!
            label = batch[-2]
            label_digits = batch[-1]
            numbs = batch[:-2]

            x = torch.cat(numbs, dim=1).to(device)
            label = label.to(device)
            try:
                trainresult = model.train_all(x, label)
                loss_percept = trainresult.percept_loss
                loss_nrm = trainresult.q_loss
            except NoPossibleActionsException:
                print("No possible actions during training")
                continue

            cum_loss_percept += loss_percept.item()
            cum_loss_nrm += loss_nrm.item()

            test_result = test(x, label, label_digits, model, device)
            train_acc += test_result[0]
            train_acc_prior += test_result[1]
            train_explain_acc += test_result[2]
            train_digit_acc += test_result[3]

            if (i + 1) % log_iterations == 0:
                avg_alpha = torch.nn.functional.softplus(model.alpha).mean()

                log_q_weight = torch.sigmoid((config['percept_loss_pref'] * avg_alpha).log()).item()
                avg_alpha = avg_alpha.item()

                print(f"actor: {cum_loss_percept / log_iterations:.4f} "
                      f"nrm: {cum_loss_nrm / log_iterations:.4f} " 
                      f"avg_alpha: {avg_alpha:.4f} ",
                      f"log_q_weight: {log_q_weight:.4f} ",
                      f"train_acc: {train_acc / log_iterations:.4f}",
                      f"train_acc_prior: {train_acc_prior / log_iterations:.4f}",
                      f"train_explain_acc: {train_explain_acc / log_iterations:.4f}",
                      f"train_digit_acc: {train_digit_acc / log_iterations:.4f}")

                wandb.log({
                    # "epoch": epoch,
                    "percept_loss": cum_loss_percept / log_iterations,
                    "nrm_loss": cum_loss_nrm / log_iterations,
                    "train_accuracy": train_acc / log_iterations,
                    "train_accuracy_prior": train_acc_prior / log_iterations,
                    "train_explain_accuracy": train_explain_acc / log_iterations,
                    "train_digit_accuracy": train_digit_acc / log_iterations,
                    "avg_alpha": avg_alpha,
                    # "log_q_weight": log_q_weight,
                })
                cum_loss_percept = 0
                cum_loss_nrm = 0
                train_acc = 0
                train_acc_prior = 0
                train_explain_acc = 0
                train_digit_acc = 0

        end_epoch_time = time.time()

        if config['test']:
            print("----- TESTING -----")
        else:
            print("----- VALIDATING -----")
        val_acc = 0.
        val_acc_prior = 0.
        val_explain_acc = 0.
        val_digit_acc = 0.
        for i, batch in enumerate(val_loader):
            # numb1, numb2, label, label_digits = batch
            # x = torch.cat([numb1, numb2], dim=1)
            label = batch[-2]
            label_digits = batch[-1]
            numbs = batch[:-2]
            x = torch.cat(numbs, dim=1).to(device)

            test_result = test(x.to(device), label.to(device), label_digits, model, device)
            val_acc += test_result[0]
            val_acc_prior += test_result[1]
            val_explain_acc += test_result[2]
            val_digit_acc += test_result[3]

        val_accuracy = val_acc / len(val_loader)
        val_accuracy_prior = val_acc_prior / len(val_loader)
        val_explain_accuracy = val_explain_acc / len(val_loader)
        val_digit_accuracy = val_digit_acc / len(val_loader)
        epoch_time = end_epoch_time - start_epoch_time
        test_time = time.time() - end_epoch_time

        prefix = 'Test' if config['test'] else 'Val'

        print(f"{prefix} accuracy: {val_accuracy} {prefix} Explain: {val_explain_accuracy}",
              f"{prefix} Digit: {val_digit_accuracy} Epoch time: {epoch_time} {prefix} time: {test_time}")

        wdb_prefix = 'test' if config['test'] else 'val'
        wandb.log({
            # "epoch": epoch,
            f"{wdb_prefix}_accuracy": val_accuracy,
            f"{wdb_prefix}_accuracy_prior": val_accuracy_prior,
            f"{wdb_prefix}_digit_accuracy": val_digit_accuracy,
            f"{wdb_prefix}_time": test_time,
            "epoch_time": epoch_time,
        })
    run.finish()