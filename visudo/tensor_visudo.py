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

def all_distinct(n):
    t = torch.zeros(tuple([n]*n)).to(device)
    inds = list(itertools.permutations(list(range(n))))
    for inds_i in inds:
        t[tuple(inds_i)] = 1
    t = F.one_hot(t.long(), num_classes=2).float()
    return t

def sub_program4(P):
    gt = all_distinct(4)
    a, b, c, d = list(zip(*ij4))
    xs = P.permute(1, 0, 2)[list(a)]
    ys = P.permute(1, 0, 2)[list(b)]
    zs = P.permute(1, 0, 2)[list(c)]
    ws = P.permute(1, 0, 2)[list(d)]
    output = torch.einsum('pia, pib, pic, pid, abcde -> pie', xs, ys, zs, ws, gt).to(device)
    return output

def sub_program9(P):
    gt = all_distinct(9)
    outputs = []
    for ij9_i in (ij9):
        a, b, c, d, e, f, g, h, i = list(zip(*([ij9_i])))
        xs = P.permute(1, 0, 2)[list(a)]
        ys = P.permute(1, 0, 2)[list(b)]
        zs = P.permute(1, 0, 2)[list(c)]
        ws = P.permute(1, 0, 2)[list(d)]
        ms = P.permute(1, 0, 2)[list(e)]
        ns = P.permute(1, 0, 2)[list(f)]
        os = P.permute(1, 0, 2)[list(g)]
        ps = P.permute(1, 0, 2)[list(h)]
        qs = P.permute(1, 0, 2)[list(i)]
        eqn = 'pia, pib, pic, pid, pie, pif, pig, pih, pij, abcdefghjk -> pik'
        outputs.append(torch.einsum(eqn, xs, ys, zs, ws, ms, ns, os, ps, qs, gt))
    output = torch.cat(outputs, dim=0)
    return output

def all_distinct_pair(n):
    t = torch.ones(n, n).to(device)
    for i in range(n):
        t[i, i] = 0
    return t

def sub_program_2(P, n):
    # gt = all_distinct_pair(n)
    if n == 4: ij = ij4_2
    else: ij = ij9_2
    xs = [torch.tensor([x for (x, _) in ij])]
    ys = [torch.tensor([y for (_, y) in ij])]
    Pxs = P.permute(1, 0, 2)[xs]
    Pys = P.permute(1, 0, 2)[ys]
    output = torch.einsum('pia, pib, ab -> pi', Pxs, Pys, gt_pair)
    return output

def all_triple():
    output = torch.zeros((9, 9, 9, len(triples_9))).to(device)
    for n, (i, j, k) in enumerate(triples_9):
        ns = F.one_hot(torch.tensor(n).long(), num_classes=len(triples_9))
        output[i, j, k] = ns
        output[i, k, j] = ns
        output[j, i, k] = ns
        output[j, k, i] = ns
        output[k, i, j] = ns      
        output[k, j, i] = ns
    return output

def combine_triples():
    output = torch.zeros(84, 84, 84).to(device)
    rs = list(itertools.permutations(list(range(9))))
    for (a, b, c, d, e, f, g, h, i) in rs:
        if sorted(list((a, b, c))) != list((a, b, c)) or sorted(list((d, e, f))) != list((d, e, f)) or sorted(list((g, h, i))) != list((g, h, i)): continue
        n, m, o = triples_9.index((a, b, c)), triples_9.index((d, e, f)), triples_9.index((g, h, i))
        output[n, m, o] = 1
    output = F.one_hot(output.long(), num_classes=2).float()
    return output       

def program_triple(P):
    a, b, c, d, e, f, g, h, i = list(zip(*ij9))
    xs = P.permute(1, 0, 2)[list(a)] # 27 b 9
    ys = P.permute(1, 0, 2)[list(b)]
    zs = P.permute(1, 0, 2)[list(c)]
    ws = P.permute(1, 0, 2)[list(d)]
    ms = P.permute(1, 0, 2)[list(e)]
    ns = P.permute(1, 0, 2)[list(f)]
    os = P.permute(1, 0, 2)[list(g)]
    ps = P.permute(1, 0, 2)[list(h)]
    qs = P.permute(1, 0, 2)[list(i)]
    x = torch.stack((xs, ys, zs), dim=0)
    y = torch.stack((ws, ms, ns), dim=0)
    z = torch.stack((os, ps, qs), dim=0)
    output = torch.einsum('pqai, pqaj, pqak, ijkl -> pqal', x, y, z, gt)
    output = torch.einsum('pqa, pqb, pqc, abcr -> pqr', output[0], output[1], output[2], gt2)
    rs = (output[:, :, 1] + EPS).log().sum(dim=0)
    # rs = output.prod(dim=0)
    return rs

def program(P):
    rs = sub_program4(P)
    # output = rs.prod(dim=0)
    # output = torch.stack((1 - output[:, 1], output[:, 1]), dim=-1)
    output = (rs[:, :, 1] + EPS).log().sum(dim=0).to(device)
    return output

def program_2(P):
    rs = sub_program_2(P, 9)
    p = (rs + EPS).log().sum(dim=0).to(device)
    return p

def loss_fn(output, ground_truth):
    output[ground_truth == 0] = log_not(output[ground_truth == 0])
    return (-output).mean()
    
def val_sudo(x, label, model, device):
    P = model(x)
    rs = sub_program_2(P, 9)
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
        "amt_samples": 100,
        "batch_size": 20,
        "batch_size_test": 100,
        "epochs": 5000,
        "log_per_epoch": 1,
        "perception_lr": 5e-4,
        "pretrain_epochs": 0,
        "split": 1,
        "test": True,
        "train_negatives": True,
        "use_cuda": True,
        "val_freq": 10,
    }

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
        with open("visudo/sweeps/sweep9.yaml", 'r') as f:
            sweep_config = yaml.load(f, Loader=yaml.FullLoader)

        run = wandb.init(config=sweep_config)
        config.update(wandb.config)
        print(config)
    else:
        run = wandb.init(
            project="tensor-visudo" + str(config["N"]),
            entity="seewonchoi",
            config=config,
            name=f'{config["split"]}'
        )

    # Check for available GPUs
    use_cuda = config["use_cuda"] and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    model = MNIST_Net(config["N"]).to(device)
    train, val, test = get_datasets(config["split"], dimension=config["N"], use_negative_train=config["train_negatives"])
    # train, val, test = get_datasets(dimension=config["N"], use_negative_train=config["train_negatives"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config["perception_lr"])

    if config["test"]:
        val = test

    train_loader = DataLoader(train, config["batch_size"], shuffle=True)
    val_loader = DataLoader(val, config["batch_size_test"], False)

    log_iterations = len(train_loader) // config["log_per_epoch"]

    t0 = time.time()
    gt = all_triple()
    gt2 = combine_triples()
    # all_distinct(config["N"])
    t1 = time.time()
    wandb.log({"pretrain": t1 - t0})

    gt_pair = all_distinct_pair(config["N"])

    if config["DEBUG"]:
        torch.autograd.set_detect_anomaly(True)

    for epoch in range(config["epochs"]):
        cum_loss_percept = 0
        cum_loss_q = 0

        start_epoch_time = time.time()

        #if epoch > config["pretrain_epochs"]:
        #    model.requires_grad_(True)

        for i, batch in enumerate(train_loader):
            # print(i)
            optimizer.zero_grad()
            grid, label = batch

            x = grid.to(device)
            label = label.to(device)
            P = model(x)
            output = program_triple(P)
            loss = loss_fn(output, label.float())
            loss.backward()
            optimizer.step()

            cum_loss_percept += loss.item()
            if (i + 1) % log_iterations == 0:
                print(f"epoch: {epoch} "
                      f"average loss: {cum_loss_percept / log_iterations:.4f} ")
                wandb.log({
                    # "epoch": epoch,
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

def all_distinct_sketch(n, r):
    gt = all_distinct(n)[:, :, :, :, 1]
    inds = list(itertools.permutations(list(range(n))))
    t = tt_sketch.tensor.SparseTensor([n]*n, torch.tensor(inds).t().long(), torch.tensor([1.0]*len(inds)))
    assert(torch.all(t.to_numpy() == gt))
    s = tt_sketch.sketch.hmt_sketch(t, r)
    assert(torch.all(s.to_numpy().round() == gt))
    return s

def all_distinct_sketch9(n, r):
    gt = all_distinct(n)[:, :, :, :, :, :, :, :, :, 1] * 10
    inds = list(itertools.permutations(list(range(n))))
    inds = torch.tensor(inds).t()
    t = tt_sketch.tensor.SparseTensor([n]*n, inds[:, :1].long(), torch.tensor([0.]*1))
    s = tt_sketch.sketch.stream_sketch(tt_sketch.tensor.DenseTensor(gt), 5)
    assert(not (torch.any(torch.isnan(s.to_numpy()))))
    for i in range(0, 126):
       ind_i = inds[:, i*2880:(i+1)*2880]
       t = tt_sketch.tensor.SparseTensor([n]*n, ind_i.long(), torch.tensor([1.0]*2880))
       s = s + t
    s = s.to_numpy().round()
    print(torch.abs(s - gt).sum())
    print(s.max())
    assert(torch.all(s == gt))
    return s

def sub_program_sketch9(P, n):
    r = 6
    batch_size = P[0].shape[0]
    output = torch.zeros(batch_size).to(device)
    cs = all_distinct_sketch(n, r)
    rs = [list(range(min(r, 4)))] + [list(range(r))] * (n - 3) + [list(range(min(r, 4)))]
    rs = list(itertools.product(*rs))
    for rs_i in rs:
      temp = torch.ones(batch_size).to(device)
      temp *= (P[0] * cs[0][:, :, rs_i[0]].to(device)).sum(dim=-1)
      for i in range(1, n - 1):
        temp *= (P[i] * cs[i][rs_i[i - 1], :, rs_i[i]].unsqueeze(0).to(device)).sum(dim=-1)
      temp *= (P[-1] * cs[-1][rs_i[-1], :, :].t().to(device)).sum(dim=-1)
      output += temp
    return output

def program_sketch(P):
    rs = all_distinct_sketch9(P, 9)
    p = (rs + EPS).log().sum(dim=0).to(device)
    return p