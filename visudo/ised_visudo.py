import argparse
import time

import yaml
from torch.utils.data import DataLoader
import torch
import wandb
from torcheval.metrics import BinaryAUROC

from parse_data import get_datasets
from MNISTNet import MNIST_Net
from sudoku_config import ij4, ij9, ij4_2, ij9_2
from util import log_not

import blackbox

SWEEP = True
EPS = 1e-8

def sub_program_pair(p1, p2):
    return int(p1 == p2)

def program_pair(logits, target, N):
    ised_ctx = blackbox.BlackBoxFunction(
       sub_program_pair,
       input_mappings=tuple([blackbox.DiscreteInputMapping(list(range(N)))]*2),
       output_mapping=blackbox.DiscreteOutputMapping(list(range(2))),
       sample_count=config["sample_count"],
       loss_aggregator='add_mult')

    if N == 4: ij = ij4_2
    else: ij = ij9_2
    a, b = list(zip(*ij))
    xs = logits.permute(1, 0, 2)[list(a)].flatten(0, 1)
    ys = logits.permute(1, 0, 2)[list(b)].flatten(0, 1)
    r = ised_ctx(xs, ys)
    rs = r[:, 1].reshape(-1, len(target)).t()
    rs = (rs + EPS).log().sum(dim=-1)
    rs[target == 0] = log_not(rs[target == 0])
    loss = -rs.mean()
    return loss

def sub_program4(p1, p2, p3, p4):
    l = [p1, p2, p3, p4]
    if len(l) == len(set(l)): return 1
    return 0

def sub_program9(p1, p2, p3, p4, p5, p6, p7, p8, p9):
    l = [p1, p2, p3, p4, p5, p6, p7, p8, p9]
    if len(l) == len(set(l)): return 1
    return 0

def program(logits, target, N):
    if N == 4: sub_program = sub_program4
    else: sub_program = sub_program9
    ised_ctx = blackbox.BlackBoxFunction(
       sub_program,
       input_mappings=tuple([blackbox.DiscreteInputMapping(list(range(N)))]*N),
       output_mapping=blackbox.DiscreteOutputMapping(list(range(2))),
       sample_count=config["sample_count"],
       loss_aggregator='add_mult')    
    
    rs = []
    if N == 4:
        a, b, c, d = list(zip(*ij4))
        xs = logits.permute(1, 0, 2)[list(a)].flatten(0, 1)
        ys = logits.permute(1, 0, 2)[list(b)].flatten(0, 1)
        zs = logits.permute(1, 0, 2)[list(c)].flatten(0, 1)
        ws = logits.permute(1, 0, 2)[list(d)].flatten(0, 1)
        r = ised_ctx(xs, ys, zs, ws)
        rs = r[:, 1].reshape(-1, len(target)).t()
    else:
        a, b, c, d, e, f, g, h, j = list(zip(*ij9))
        xs = logits.permute(1, 0, 2)[list(a)].flatten(0, 1)
        ys = logits.permute(1, 0, 2)[list(b)].flatten(0, 1)
        zs = logits.permute(1, 0, 2)[list(c)].flatten(0, 1)
        ws = logits.permute(1, 0, 2)[list(d)].flatten(0, 1)
        es = logits.permute(1, 0, 2)[list(e)].flatten(0, 1)
        fs = logits.permute(1, 0, 2)[list(f)].flatten(0, 1)
        gs = logits.permute(1, 0, 2)[list(g)].flatten(0, 1)
        hs = logits.permute(1, 0, 2)[list(h)].flatten(0, 1)
        js = logits.permute(1, 0, 2)[list(j)].flatten(0, 1)
        ised_ctx(xs, ys, zs, ws, es, fs, gs, hs, js)
        rs = r[:, 1].reshape(-1, len(target)).t()
    
    rs = (rs + EPS).log().sum(dim=-1)
    rs[target == 0] = log_not(rs[target == 0])
    loss = -rs.mean()
    return loss

def all_distinct_pair(n):
    t = torch.ones(n, n).to(device)
    for i in range(n):
        t[i, i] = 0
    return t

def gt_program(P, n):
    # gt = all_distinct_pair(n)
    if n == 4: ij = ij4_2
    else: ij = ij9_2
    xs = [torch.tensor([x for (x, _) in ij])]
    ys = [torch.tensor([y for (_, y) in ij])]
    Pxs = P.permute(1, 0, 2)[xs]
    Pys = P.permute(1, 0, 2)[ys]
    output = torch.einsum('pia, pib, ab -> pi', Pxs, Pys, gt_pair)
    return output

def val_sudo(x, label, model, device):
    P = model(x)
    rs = gt_program(P, config['N'])
    output = rs.prod(dim=0).to(device)
    if len(output.shape) == 1:
        pred = output.round()
        prior_y = output
    else:
        pred = output.argmax(dim=-1)
        prior_y = output[:, 1]
    acc = (pred == label).sum() / len(label)
    acc_prior = acc
    return acc, acc_prior, prior_y

if __name__ == '__main__':
    config = {
        "N": 4,
        "DEBUG": True,
        "batch_size": 20,
        "batch_size_test": 100,
        "epochs": 1000,
        "log_per_epoch": 1,
        "perception_lr": 1e-3,
        "split": 1,
        "test": True,
        "train_negatives": True,
        "use_cuda": True,
        "val_freq": 10,
        "sample_count": 50,
        "train": "00100"
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    run = wandb.init(
            project=f"ised-visudo{config['N']}",
            name=f'{config["split"]}',
            config=config,
        )
    print(config)

    # Check for available GPUs
    use_cuda = config["use_cuda"] and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    gt_pair = all_distinct_pair(config["N"])
    model = MNIST_Net(config["N"]).to(device)
    train, val, test = get_datasets(config["split"], numTrain=config["train"], dimension=config["N"], use_negative_train=config["train_negatives"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["perception_lr"])

    if config["test"]:
        val = test

    train_loader = DataLoader(train, config["batch_size"], shuffle=True)
    val_loader = DataLoader(val, config["batch_size_test"], False)

    log_iterations = len(train_loader) // config["log_per_epoch"]

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
            loss = program(P, label, config["N"])
            loss.backward()
            optimizer.step()

            cum_loss_percept += loss.item()
            if (i + 1) % log_iterations == 0:
                print(f"epoch: {epoch} "
                      f"average loss: {cum_loss_percept / log_iterations:.4f} ")
                wandb.log({
                    "percept_loss": cum_loss_percept / log_iterations
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
        prior_y = []
        labels = []
        for i, batch in enumerate(val_loader):
            grid, label = batch
            label = label.to(device)
            test_result = val_sudo(grid.to(device), label, model, device)
            val_acc += test_result[0]
            val_acc_prior += test_result[1]
            prior_y += [test_result[2]]
            labels += [label]

        val_accuracy = val_acc / len(val_loader)

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
            f"{wdb_prefix}_accuracy": val_accuracy,
            f"{wdb_prefix}_auroc": val_auroc,
            f"{wdb_prefix}_time": test_time,
            "epoch_time": epoch_time,
        })
