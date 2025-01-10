import argparse
import time
import itertools

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

import scallopy

SWEEP = True
EPS = 1e-8

def sub_program_pair(p1, p2, N, provenance, k, dispatch):
    scl_ctx = scallopy.ScallopContext(provenance=provenance, k=k)
    scl_ctx.add_relation(f"digit_1", int, input_mapping=list(range(N)))
    scl_ctx.add_relation(f"digit_2", int, input_mapping=list(range(N)))
    scl_ctx.add_rule("sum(c) :- digit_1(a), digit_2(b), c == (a != b)")
    sum_n = scl_ctx.forward_function("sum", output_mapping=[(i,) for i in range(2)], dispatch=dispatch)
    output = sum_n(digit_1=p1, digit_2=p2)
    return output

def program_pair(logits, target, N, provenance, k, dispatch):
    if N == 4: ij = ij4_2
    else: ij = ij9_2

    rs = []
    for (i, j) in ij:
        r = sub_program_pair(logits[:, i], logits[:, j], N, provenance, k, dispatch)
        rs.append(r[:, 1])

    rs = torch.stack(rs, dim=-1)
    rs = (rs + EPS).log().sum(dim=-1)
    rs[target == 0] = log_not(rs[target == 0])
    loss = -rs.mean()
    return loss

def sum_program4_fn(N, provenance, k, dispatch):
    scl_ctx = scallopy.ScallopContext(provenance=provenance, k=k)
    scl_ctx.import_file("visudo/sub4.scl")
    scl_ctx.add_relation(f"digit_1", int, input_mapping=list(range(N)))
    scl_ctx.add_relation(f"digit_2", int, input_mapping=list(range(N)))
    scl_ctx.add_relation(f"digit_3", int, input_mapping=list(range(N)))
    scl_ctx.add_relation(f"digit_4", int, input_mapping=list(range(N)))
    # scl_ctx.add_rule("sum(e) :- digit_1(a), digit_2(b), digit_3(c), digit_4(d), e == ((a != b) && (a != c) &&(a != d) && (b != c) && (b != d) && (c != d))")
    sum_n = scl_ctx.forward_function("sum", output_mapping=[(i,) for i in range(2)], dispatch=dispatch)
    return sum_n

def sub_program4(p1, p2, p3, p4, sum_n):
    output = sum_n(digit_1=p1, digit_2=p2, digit_3=p3, digit_4=p4)
    return output.to(device)

def program(logits, target, N, provenance, k, dispatch):
    rs = []
    sum_n = sum_program4_fn(N, provenance, k, dispatch)

    for (i, j, k, l) in ij4:
        print((i, j, k, l))
        r = sub_program4(logits[:, i].cpu(), logits[:, j].cpu(), logits[:, k].cpu(), logits[:, l].cpu(), sum_n)
        rs.append(r[:, 1])

    rs = torch.stack(rs, dim=-1)
    rs = (rs + EPS).log().sum(dim=-1)
    rs[target == 0] = log_not(rs[target == 0])
    loss = -rs.mean()
    return loss

def program_all(logits, target, N, provenance, k, dispatch):
    logits = logits.permute(1, 0, 2).cpu()
    scl_ctx = scallopy.ScallopContext(provenance=provenance, k=k)
    scl_ctx.import_file("visudo/four.scl")
    scl_ctx.add_relation(f"digit_1", int, input_mapping=list(range(N)))
    scl_ctx.add_relation(f"digit_2", int, input_mapping=list(range(N)))
    scl_ctx.add_relation(f"digit_3", int, input_mapping=list(range(N)))
    scl_ctx.add_relation(f"digit_4", int, input_mapping=list(range(N)))
    scl_ctx.add_relation(f"digit_5", int, input_mapping=list(range(N)))
    scl_ctx.add_relation(f"digit_6", int, input_mapping=list(range(N)))
    scl_ctx.add_relation(f"digit_7", int, input_mapping=list(range(N)))
    scl_ctx.add_relation(f"digit_8", int, input_mapping=list(range(N)))
    scl_ctx.add_relation(f"digit_9", int, input_mapping=list(range(N)))
    scl_ctx.add_relation(f"digit_10", int, input_mapping=list(range(N)))
    scl_ctx.add_relation(f"digit_11", int, input_mapping=list(range(N)))
    scl_ctx.add_relation(f"digit_12", int, input_mapping=list(range(N)))
    scl_ctx.add_relation(f"digit_13", int, input_mapping=list(range(N)))
    scl_ctx.add_relation(f"digit_14", int, input_mapping=list(range(N)))
    scl_ctx.add_relation(f"digit_15", int, input_mapping=list(range(N)))
    scl_ctx.add_relation(f"digit_16", int, input_mapping=list(range(N)))
    visudo = scl_ctx.forward_function("visudo", output_mapping=[(i,) for i in range(2)], dispatch=dispatch)
    
    output = visudo(digit_1=logits[0], digit_2=logits[1], digit_3=logits[2], digit_4=logits[3],
                    digit_5=logits[4], digit_6=logits[5], digit_7=logits[6], digit_8=logits[7],
                    digit_9=logits[8], digit_10=logits[9], digit_11=logits[10], digit_12=logits[11],
                    digit_13=logits[12], digit_14=logits[13], digit_15=logits[14], digit_16=logits[15])
    
    rs = (output + EPS).log().sum(dim=-1)
    rs[target == 0] = log_not(rs[target == 0])
    loss = -rs.mean()
    return loss

def program_all9(logits, target, N, provenance, k, dispatch):
    logits = logits.permute(1, 0, 2).cpu()
    scl_ctx = scallopy.ScallopContext(provenance=provenance, k=k)
    scl_ctx.import_file("visudo/nine.scl")
    for i in range(1, 82):
        scl_ctx.add_relation(f"digit_{i}", int, input_mapping=list(range(N)))
    visudo = scl_ctx.forward_function("visudo", output_mapping=[(i,) for i in range(2)], dispatch=dispatch)
    
    output = visudo(
        digit_1 = logits[0], digit_2 = logits[1], digit_3 = logits[2], digit_4 = logits[3], digit_5 = logits[4], digit_6 = logits[5], digit_7 = logits[6], digit_8 = logits[7], digit_9 = logits[8], 
        digit_10 = logits[9], digit_11 = logits[10], digit_12 = logits[11], digit_13 = logits[12], digit_14 = logits[13], digit_15 = logits[14], digit_16 = logits[15], digit_17 = logits[16], digit_18 = logits[17], 
        digit_19 = logits[18], digit_20 = logits[19], digit_21 = logits[20], digit_22 = logits[21], digit_23 = logits[22], digit_24 = logits[23], digit_25 = logits[24], digit_26 = logits[25], digit_27 = logits[26], 
        digit_28 = logits[27], digit_29 = logits[28], digit_30 = logits[29], digit_31 = logits[30], digit_32 = logits[31], digit_33 = logits[32], digit_34 = logits[33], digit_35 = logits[34], digit_36 = logits[35], 
        digit_37 = logits[36], digit_38 = logits[37], digit_39 = logits[38], digit_40 = logits[39], digit_41 = logits[40], digit_42 = logits[41], digit_43 = logits[42], digit_44 = logits[43], digit_45 = logits[44], 
        digit_46 = logits[45], digit_47 = logits[46], digit_48 = logits[47], digit_49 = logits[48], digit_50 = logits[49], digit_51 = logits[50], digit_52 = logits[51], digit_53 = logits[52], digit_54 = logits[53], 
        digit_55 = logits[54], digit_56 = logits[55], digit_57 = logits[56], digit_58 = logits[57], digit_59 = logits[58], digit_60 = logits[59], digit_61 = logits[60], digit_62 = logits[61], digit_63 = logits[62], 
        digit_64 = logits[63], digit_65 = logits[64], digit_66 = logits[65], digit_67 = logits[66], digit_68 = logits[67], digit_69 = logits[68], digit_70 = logits[69], digit_71 = logits[70], digit_72 = logits[71], 
        digit_73 = logits[72], digit_74 = logits[73], digit_75 = logits[74], digit_76 = logits[75], digit_77 = logits[76], digit_78 = logits[77], digit_79 = logits[78], digit_80 = logits[79], digit_81 = logits[80])
    
    rs = (output + EPS).log().sum(dim=-1)
    rs[target == 0] = log_not(rs[target == 0])
    loss = -rs.mean()
    return loss

def program_all9_pair(logits, target, N, provenance, k, dispatch):
    logits = logits.permute(1, 0, 2).cpu()
    scl_ctx = scallopy.ScallopContext(provenance=provenance, k=k)
    scl_ctx.import_file("visudo/nine_pair.scl")
    for i in range(0, 81):
        scl_ctx.add_relation(f"digit_{i}", int, input_mapping=list(range(N)))
    visudo = scl_ctx.forward_function("visudo", output_mapping=[(i,) for i in range(2)], dispatch=dispatch)
    
    output = visudo(
        digit_0 = logits[0], digit_1 = logits[1], digit_2 = logits[2], digit_3 = logits[3], digit_4 = logits[4], digit_5 = logits[5], digit_6 = logits[6], digit_7 = logits[7], digit_8 = logits[8], 
        digit_9 = logits[9], digit_10 = logits[10], digit_11 = logits[11], digit_12 = logits[12], digit_13 = logits[13], digit_14 = logits[14], digit_15 = logits[15], digit_16 = logits[16], digit_17 = logits[17], 
        digit_18 = logits[18], digit_19 = logits[19], digit_20 = logits[20], digit_21 = logits[21], digit_22 = logits[22], digit_23 = logits[23], digit_24 = logits[24], digit_25 = logits[25], digit_26 = logits[26], 
        digit_27 = logits[27], digit_28 = logits[28], digit_29 = logits[29], digit_30 = logits[30], digit_31 = logits[31], digit_32 = logits[32], digit_33 = logits[33], digit_34 = logits[34], digit_35 = logits[35], 
        digit_36 = logits[36], digit_37 = logits[37], digit_38 = logits[38], digit_39 = logits[39], digit_40 = logits[40], digit_41 = logits[41], digit_42 = logits[42], digit_43 = logits[43], digit_44 = logits[44], 
        digit_45 = logits[45], digit_46 = logits[46], digit_47 = logits[47], digit_48 = logits[48], digit_49 = logits[49], digit_50 = logits[50], digit_51 = logits[51], digit_52 = logits[52], digit_53 = logits[53], 
        digit_54 = logits[54], digit_55 = logits[55], digit_56 = logits[56], digit_57 = logits[57], digit_58 = logits[58], digit_59 = logits[59], digit_60 = logits[60], digit_61 = logits[61], digit_62 = logits[62], 
        digit_63 = logits[63], digit_64 = logits[64], digit_65 = logits[65], digit_66 = logits[66], digit_67 = logits[67], digit_68 = logits[68], digit_69 = logits[69], digit_70 = logits[70], digit_71 = logits[71], 
        digit_72 = logits[72], digit_73 = logits[73], digit_74 = logits[74], digit_75 = logits[75], digit_76 = logits[76], digit_77 = logits[77], digit_78 = logits[78], digit_79 = logits[79], digit_80 = logits[80])
    
    rs = (output + EPS).log().sum(dim=-1)
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
        "perception_lr": 5e-4,
        "split": 11,
        "test": True,
        "train_negatives": True,
        "use_cuda": True,
        "val_freq": 10,
        "k": 3,
        "provenance": "difftopkproofs",
        "dispatch": "parallel",
        "train": "00100"
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
    else:
        run = wandb.init(
            project=f"scallop-visudo{config['N']}",
            name=f'{config["split"]}',
            config=config,
        )
        print(config)

    # Check for available GPUs
    use_cuda = config["use_cuda"] and torch.cuda.is_available()
    device = torch.device(1 if use_cuda else 'cpu')

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

        #if epoch > config["pretrain_epochs"]:
        #    model.requires_grad_(True)

        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            grid, label = batch

            x = grid.to(device)
            label = label.to(device)
            P = model(x)
            loss = program_all(P, label, config["N"], config["provenance"], config["k"], config["dispatch"])
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
            # "epoch": epoch,
            f"{wdb_prefix}_accuracy": val_accuracy,
            f"{wdb_prefix}_auroc": val_auroc,
            f"{wdb_prefix}_time": test_time,
            "epoch_time": epoch_time,
        })
