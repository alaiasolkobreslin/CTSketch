import argparse
import time
import itertools
import pathlib
import sys

import yaml
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import wandb
from torcheval.metrics import BinaryAUROC

from parse_data import get_datasets
from MNISTNet import MNIST_Net
from sudoku_config import ij4, ij9, ij4_2, ij9_2, triples_9
from util import log_not

sys.path.insert(0, f'{pathlib.Path(__file__).parent.parent.absolute()}')
import tt_sketch.tensor
import tt_sketch.sketch

SWEEP = False
EPS = 1e-8

def all_distinct_pair(n):
    t = torch.ones(n, n).to(device)
    for i in range(n):
        t[i, i] = 0
    return t

def sub_program_pair(P, n):
    if n == 4: ij = ij4_2
    else: ij = ij9_2
    xs = [torch.tensor([x for (x, _) in ij])]
    ys = [torch.tensor([y for (_, y) in ij])]
    Pxs = P.permute(1, 0, 2)[xs]
    Pys = P.permute(1, 0, 2)[ys]
    output = torch.einsum('pia, pib, ab -> pi', Pxs, Pys, gt_pair)
    return output

def program_pair(P):
    rs = sub_program_pair(P, config["N"])
    p = (rs + EPS).log().sum(dim=0).to(device)
    return p

def loss_fn(output, ground_truth):
    output[ground_truth == 0] = log_not(output[ground_truth == 0])
    return (-output).mean()
    
def val_sudo(x, label, model, device):
    P = model(x)
    rs = sub_program_pair(P, config["N"])
    output = rs.prod(dim=0).to(device)
    if len(output.shape) == 1:
        pred = output.round()
        prior_y = output
    else:
        pred = output.argmax(dim=-1)
        prior_y = output[:, 1]
    acc = (pred == label).sum() / len(label)
    acc_prior = acc
    acc_clauses = 0
    return acc, acc_prior, acc_clauses, prior_y

if __name__ == '__main__':
    config = {
        "N": 4,
        "DEBUG": True,
        "batch_size": 20,
        "batch_size_test": 100,
        "epochs": 1000,
        "log_per_epoch": 1,
        "perception_lr": 1e-3,
        "pretrain_epochs": 0,
        "split": 1,
        "test": True,
        "train_negatives": False,
        "use_cuda": True,
        "val_freq": 10,
        "train": "00100"
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    run = wandb.init(
            project="tensor-visudo" + str(config["N"]),
            config=config,
            name=f'{config["split"]}'
    )
    print(config)

    # Check for available GPUs
    use_cuda = config["use_cuda"] and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    model = MNIST_Net(config["N"]).to(device)
    train, val, test = get_datasets(config["split"], numTrain=config["train"] ,dimension=config["N"], use_negative_train=config["train_negatives"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["perception_lr"])

    if config["test"]:
        val = test

    train_loader = DataLoader(train, config["batch_size"], shuffle=True)
    val_loader = DataLoader(val, config["batch_size_test"], False)

    log_iterations = len(train_loader) // config["log_per_epoch"]

    t0 = time.time()
    gt_pair = all_distinct_pair(config["N"])
    t1 = time.time()
    wandb.log({"pretrain": t1 - t0})
    
    if config["DEBUG"]:
        torch.autograd.set_detect_anomaly(True)

    for epoch in range(config["epochs"]):
        cum_loss_percept = 0
        cum_loss_q = 0

        start_epoch_time = time.time()

        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            grid, label = batch

            x = grid.to(device)
            label = label.to(device)
            P = model(x)
            output = program_pair(P)
            loss = loss_fn(output, label.float())
            loss.backward()
            optimizer.step()

            cum_loss_percept += loss.item()
            if (i + 1) % log_iterations == 0:
                print(f"epoch: {epoch} "
                      f"average loss: {cum_loss_percept / log_iterations:.4f} ")
                wandb.log({ "percept_loss": cum_loss_percept / log_iterations })
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

        print(f"{prefix} accuracy: {val_accuracy} {prefix}"
                f" {prefix} auroc: {val_auroc} {prefix}"
              f" Epoch time: {epoch_time} {prefix} time: {test_time}")

        wdb_prefix = 'test' if config['test'] else 'val'
        wandb.log({
            # "epoch": epoch,
            f"{wdb_prefix}_accuracy": val_accuracy,
            f"{wdb_prefix}_auroc": val_auroc,
            f"{wdb_prefix}_time": test_time,
            "epoch_time": epoch_time,
        })

    run.finish()