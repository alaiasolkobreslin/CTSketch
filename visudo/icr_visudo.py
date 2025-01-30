import argparse
import time

import yaml
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import wandb
from torcheval.metrics import BinaryAUROC

from parse_data import get_datasets
from MNISTNet import MNIST_Net
from sudoku_config import ij4, ij9, ij4_2, ij9_2
from util import log_not

SWEEP = True
EPS = 1e-8

def indecater_multiplier(batch_size, N, pair, sample_count):
    icr_mult = torch.zeros((pair, N, sample_count, batch_size, N**2))
    icr_replacement = torch.zeros((pair, N, sample_count, batch_size, N**2))
    for i in range(pair):
      for j in range(N):
        icr_mult[i,j,:,:,i] = 1
        icr_replacement[i,j,:,:,i] = j
    return icr_mult.to(device), icr_replacement.to(device)

def sub_program(logits, outer_samples, fn):
    outer_loss = fn(outer_samples.long())
    variable_loss = outer_loss.mean(dim=2).permute(2,0,1)
    indecater_expression = variable_loss.detach() * F.softmax(logits, dim=-1)
    indecater_expression = indecater_expression.mean(dim=-1).mean(dim=-1)
    return torch.log(indecater_expression + EPS)

def program_pair(P, target, N):
    sample_count = config["amt_samples"]
    d = torch.distributions.Categorical(logits=P)
    samples = d.sample((sample_count,))
    outer_samples = torch.stack([samples] * N, dim=0)
    outer_samples = torch.stack([outer_samples] * N**2, dim=0)
    m, r = indecater_multiplier(P.shape[0], N, N**2, sample_count)
    outer_samples = outer_samples * (1 - m) + r

    results = []
    if N == 4: ij = ij4_2
    else: ij = ij9_2
    for (i, j) in ij:
        t = torch.stack((outer_samples[:, :, :, :, i], outer_samples[:, :, :, :, j]), dim=-1)
        indecater_expression = sub_program(torch.stack((P[:, i], P[:, j]), dim=1),  torch.stack((t[i], t[j]), dim=0), pair_fn)
        results.append(indecater_expression)
    output = torch.stack(results, dim=-1).sum(dim=-1)
    output[target == 0] = log_not(output[target == 0])
    loss = (-output).mean()
    return loss

def program(P, target, N):
    sample_count = config["amt_samples"]
    d = torch.distributions.Categorical(logits=P)
    samples = d.sample((sample_count,))
    outer_samples = torch.stack([samples] * N, dim=0)
    outer_samples = torch.stack([outer_samples] * N**2, dim=0)
    m, r = indecater_multiplier(P.shape[0], N, N**2, sample_count)
    outer_samples = outer_samples * (1 - m) + r

    results = []
    if N == 4:
        for (i, j, k, l) in ij4:
            t = torch.stack((outer_samples[:, :, :, :, i], outer_samples[:, :, :, :, j], outer_samples[:, :, :, :, k], outer_samples[:, :, :, :, l]), dim=-1)
            indecater_expression = sub_program(torch.stack((P[:, i], P[:, j], P[:, k], P[:, l]), dim=1), torch.stack((t[i], t[j], t[k], t[l]), dim=0), four_fn)
            results.append(indecater_expression)
    else:
        for (i, j, k, l, m, n, o, p, q) in ij9:
            t = torch.stack((outer_samples[:, :, :, :, i], outer_samples[:, :, :, :, j], outer_samples[:, :, :, :, k], outer_samples[:, :, :, :, l], outer_samples[:, :, :, :, m], outer_samples[:, :, :, :, n], outer_samples[:, :, :, :, o], outer_samples[:, :, :, :, p], outer_samples[:, :, :, :, q]), dim=-1)
            indecater_expression = sub_program(torch.stack((P[:, i], P[:, j], P[:, k], P[:, l], P[:, m], P[:, n], P[:, o], P[:, p], P[:, q]), dim=1), torch.stack((t[i], t[j], t[k], t[l], t[m], t[n], t[o], t[p], t[q]), dim=0), nine_fn)
            results.append(indecater_expression)
    output = torch.stack(results, dim=-1).sum(dim=-1)
    output[target == 0] = log_not(output[target == 0])
    loss = (-output).mean()
    return loss

def pair_fn(data):
    data = data.movedim(-1, 0)
    return (data[0] != data[1]).float()

def four_fn(data):
    data = data.movedim(-1, 0)
    return ((data[0] != data[1]) & (data[0] != data[2]) & (data[0] != data[3]) & (data[1] != data[2]) & (data[1] != data[3]) & (data[2] != data[3])).float()

def nine_fn(data):
    data = data.movedim(-1, 0)
    return ((data[0] != data[1]) & (data[0] != data[2]) & (data[0] != data[3]) & (data[0] != data[4]) & (data[0] != data[5]) & (data[0] != data[6]) & (data[0] != data[7]) & (data[0] != data[8]) & 
            (data[1] != data[2]) & (data[1] != data[3]) & (data[1] != data[4]) & (data[1] != data[5]) & (data[1] != data[6]) & (data[1] != data[7]) & (data[0] != data[8]) & 
            (data[2] != data[3]) & (data[2] != data[4]) & (data[2] != data[5]) & (data[2] != data[6]) & (data[2] != data[7]) & (data[2] != data[8]) &
            (data[3] != data[4]) & (data[3] != data[5]) & (data[3] != data[6]) & (data[3] != data[7]) & (data[3] != data[8]) &
            (data[4] != data[5]) & (data[4] != data[6]) & (data[4] != data[7]) & (data[4] != data[8]) & 
            (data[5] != data[6]) & (data[5] != data[7]) & (data[5] != data[8]) &
            (data[6] != data[7]) & (data[6] != data[8]) &
            (data[7] != data[8])).float()

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
    P = F.softmax(P, dim=-1)
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
        "amt_samples": 10,
        "batch_size": 20,
        "batch_size_test": 100,
        "epochs": 1000,
        "log_per_epoch": 1,
        "perception_lr": 1e-4,
        "split": 11,
        "test": True,
        "train_negatives": True,
        "use_cuda": True,
        "val_freq": 10,
        "train": "00100",
        "program_fn": "program_pair"
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    run = wandb.init(
            project=f"icr-visudo{config['N']}",
            name=f"{config['split']}",
            config=config,
        )
    print(config)

    # Check for available GPUs
    use_cuda = config["use_cuda"] and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    
    gt_pair = all_distinct_pair(config["N"])
    model = MNIST_Net(config["N"], with_softmax=False).to(device)
    train, val, test = get_datasets(config["split"], numTrain=config["train"], dimension=config["N"], use_negative_train=config["train_negatives"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["perception_lr"])
    if config["program_fn"] == 'program': program_f = program
    else: program_f = program_pair

    if config["test"]:
        val = test

    train_loader = DataLoader(train, config["batch_size"], shuffle=True)
    val_loader = DataLoader(val, config["batch_size_test"], False)

    log_iterations = len(train_loader) // config["log_per_epoch"]

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
            loss = program_f(P, label, config['N'])
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
        val_accuracy_prior = val_acc_prior / len(val_loader)

        all_labels = torch.cat(labels, dim=0)
        all_prior_y = torch.cat(prior_y, dim=0)
        metric = BinaryAUROC()
        metric.update(all_prior_y, all_labels)
        val_auroc = metric.compute().item()

        epoch_time = end_epoch_time - start_epoch_time
        test_time = time.time() - end_epoch_time

        prefix = 'Test' if config['test'] else 'Val'

        print(f"{prefix} accuracy: {val_accuracy} {prefix} accuracy prior: {val_accuracy_prior} {prefix}"
                f" {prefix} auroc: {val_auroc} {prefix}"
              f" Epoch time: {epoch_time} {prefix} time: {test_time}")

        wdb_prefix = 'test' if config['test'] else 'val'
        wandb.log({
            f"{wdb_prefix}_accuracy": val_accuracy,
            f"{wdb_prefix}_accuracy_prior": val_accuracy_prior,
            f"{wdb_prefix}_auroc": val_auroc,
            f"{wdb_prefix}_time": test_time,
            "epoch_time": epoch_time,
        })

    run.finish()
