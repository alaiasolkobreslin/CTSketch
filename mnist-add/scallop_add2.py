import argparse
import time

import yaml
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import wandb
import scallopy

from add_config import addition, MNIST_Net

def test(x, label, label_digits, model, device):
    label_digits_l = list(map(lambda d: d.to(device), label_digits[0] + label_digits[1]))
    label_digits_l = torch.stack(label_digits_l, dim=-1)
    test_result = model(x)
    output = test_result.argmax(dim=-1)
    N = len(label_digits[0])
    n1 = torch.stack([10 ** (N - 1 - i) * output[:, i] for i in range(N)], -1)
    n2 = torch.stack([10 ** (N - 1 - i) * output[:, N + i] for i in range(N)], -1)
    pred = n1.sum(dim=-1) + n2.sum(dim=-1)
    acc = (pred == label).sum()
    digit_acc = (output == label_digits_l).float().mean()
    return acc, digit_acc

# no decomposition
# compare digit-wise for the loss computation
def program(logits, target, N, provenance, k, dispatch):
    output_dim = N*99 + 1

    scl_ctx = scallopy.ScallopContext(provenance=provenance, k=k)
    scl_ctx.add_relation(f"digit_1", int, input_mapping=list(range(10)))
    scl_ctx.add_relation(f"digit_2", int, input_mapping=list(range(10)))
    scl_ctx.add_relation(f"digit_3", int, input_mapping=list(range(10)))
    scl_ctx.add_relation(f"digit_4", int, input_mapping=list(range(10)))
    scl_ctx.add_rule("sum(e) :- digit_1(a), digit_2(b), digit_3(c), digit_4(d), e == a * 10 + b + c * 10 + d")
    sum_n = scl_ctx.forward_function("sum", output_mapping=[(i,) for i in range(output_dim)], dispatch=dispatch)
    output = sum_n(digit_1=logits[:, 0], digit_2=logits[:, 1], digit_3=logits[:, 2], digit_4=logits[:, 3])
    
    rs = []
    xs = torch.arange(output_dim)
    batch_size = output.shape[0]
    for i in range(N + 1):
        inds = (torch.floor_divide(xs, 10**i) % 10).repeat(batch_size, 1)
        t_i = torch.floor_divide(target, 10**i) % 10
        if i == N: 
            t = torch.zeros(batch_size, 2).scatter_add_(1, inds, output)
            l = F.one_hot(t_i, num_classes=2).float()
        else: 
            t = torch.zeros(batch_size, 10).scatter_add_(1, inds, output)
            l = F.one_hot(t_i, num_classes=10).float()
        # bce loss
        r = F.binary_cross_entropy(t, l)
        rs.append(r)
    rs = torch.stack(rs, dim=0)
    return rs.mean(dim=0)

# composed program for scallop
# currently uses NLLloss
def program_compose(logits, target, N, provenance, k, dispatch):
    n1 = logits[:, :N]
    n2 = logits[:, N:]
    
    rs = []
    for i in range(N + 1):
        l = torch.floor_divide(target, 10**i) % 10
        if i == 0:
            t = sub_program_2(torch.stack((n1[:, N-1], n2[:, N-1]), dim=1), provenance, k, dispatch)
            # loh = F.one_hot(l, num_classes=10)
        elif i == N:
            t = sub_program_1(torch.stack((n1[:, 0], n2[:, 0]), dim=1), provenance, k, dispatch)
            # loh = F.one_hot(l, num_classes=2)
        else:
            t = sub_program_4(torch.stack((n1[:, N-i], n2[:, N-i], n1[:, N-(i+1)], n2[:, N-(i+1)]), dim=1), provenance, k, dispatch)
            # loh = F.one_hot(l, num_classes=10)
        r = F.nll_loss(t, l)
        # r = F.binary_cross_entropy(t, loh)
        rs.append(r)
    rs = torch.stack(rs, dim=0).mean(dim=0)
    return rs

def sub_program_2(logits, provenance, k, dispatch):
    scl_ctx = scallopy.ScallopContext(provenance=provenance, k=k)
    scl_ctx.add_relation(f"digit_1", int, input_mapping=list(range(10)))
    scl_ctx.add_relation(f"digit_2", int, input_mapping=list(range(10)))
    scl_ctx.add_rule("sum(c) :- digit_1(a), digit_2(b), c == (a + b) % 10")
    sum_n = scl_ctx.forward_function("sum", output_mapping=[(i,) for i in range(10)], dispatch=dispatch)
    output = sum_n(digit_1=logits[:, 0], digit_2=logits[:, 1])
    return output

def sub_program_1(logits, provenance, k, dispatch):
    scl_ctx = scallopy.ScallopContext(provenance=provenance, k=k)
    scl_ctx.add_relation(f"digit_1", int, input_mapping=list(range(10)))
    scl_ctx.add_relation(f"digit_2", int, input_mapping=list(range(10)))
    scl_ctx.add_rule("sum(c) :- digit_1(a), digit_2(b), c == (a + b) / 10")
    sum_n = scl_ctx.forward_function("sum", output_mapping=[(i,) for i in range(2)], dispatch=dispatch)
    output = sum_n(digit_1=logits[:, 0], digit_2=logits[:, 1])
    return output

def sub_program_4(logits, provenance, k, dispatch):
    scl_ctx = scallopy.ScallopContext(provenance=provenance, k=k)
    scl_ctx.add_relation(f"digit_1", int, input_mapping=list(range(10)))
    scl_ctx.add_relation(f"digit_2", int, input_mapping=list(range(10)))
    scl_ctx.add_relation(f"digit_3", int, input_mapping=list(range(10)))
    scl_ctx.add_relation(f"digit_4", int, input_mapping=list(range(10)))
    scl_ctx.add_rule("sum(e) :- digit_1(a), digit_2(b), digit_3(c), digit_4(d), e == (c + d) % 10 + (a + b) / 10")
    sum_n = scl_ctx.forward_function("sum", output_mapping=[(i,) for i in range(10)], dispatch=dispatch)
    output = sum_n(digit_1=logits[:, 0], digit_2=logits[:, 1], digit_3=logits[:, 2], digit_4=logits[:, 3])
    return output

if __name__ == '__main__':
    config = {
        "use_cuda": True,
        "DEBUG": False,
        "N": 2,
        "op": "add",
        "model": "full",
        "test": False,
        "batch_size": 16,
        "batch_size_test": 16,
        "amt_samples": 100,
        "perception_lr": 1e-3,
        "epochs": 30,
        "log_per_epoch": 10,
        "k": 3,
        "provenance": "difftopkproofs",
        "dispatch": "parallel"
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    known, unknown = parser.parse_known_args()
    config_file = known.config
    if config_file is not None:
        with open(config_file, 'r') as f:
            config.update(yaml.safe_load(f))

        run = wandb.init(config=config, project="mnist-add", entity="seewonchoi")
        config = wandb.config
        print(config)
    else:
        name = "addition_" + str(config["N"])
        wandb.init(
            project=f"scallop-{config['op']}",
            name = name,
            config=config,
        )
        print(config)

    # Check for available GPUs
    use_cuda = config["use_cuda"] and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    op = addition
    model = MNIST_Net()
    percept_optimizer = torch.optim.Adam(model.parameters(), lr=config["perception_lr"])

    if config["test"]:
        train_set = op(config["N"], "full_train")
        val_set = op(config["N"], "test")
    else:
        train_set = op(config["N"], "train")
        val_set = op(config["N"], "val")

    train_loader = DataLoader(train_set, config["batch_size"], False)
    val_loader = DataLoader(val_set, config["batch_size_test"], False)

    print(len(val_loader))

    log_iterations = len(train_loader) // config["log_per_epoch"]

    if config["DEBUG"]:
        torch.autograd.set_detect_anomaly(True)

    for epoch in range(config["epochs"]):
        print("----------------------------------------")
        print("NEW EPOCH", epoch)
        cum_loss_percept = 0
        train_acc = 0
        train_digit_acc = 0

        start_epoch_time = time.time()

        for i, batch in enumerate(train_loader):
            percept_optimizer.zero_grad()
            # label_digits is ONLY EVER to be used during testing!!!
            numb1, numb2, label, label_digits = batch

            x = torch.cat([numb1, numb2], dim=1).to(device)
            label = label.to(device)
            output = model(x)
            loss = program(output, label, config["N"], config["provenance"], config["k"], config["dispatch"])
            loss.backward()
            percept_optimizer.step()

            cum_loss_percept += loss.item()

            test_result = test(x, label, label_digits, model, device)
            train_acc += test_result[0]
            train_digit_acc += test_result[1]

            if (i + 1) % log_iterations == 0:
                print(f"actor: {cum_loss_percept / log_iterations:.4f} "
                      f"train_acc: {train_acc / log_iterations:.4f}",
                      f"train_digit_acc: {train_digit_acc / log_iterations:.4f}")

                wandb.log({
                    # "epoch": epoch,
                    "percept_loss": cum_loss_percept / log_iterations,
                    "train_accuracy": train_acc / log_iterations,
                    "train_digit_accuracy": train_digit_acc / log_iterations,
                })
                cum_loss_percept = 0
                train_acc = 0
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
            numb1, numb2, label, label_digits = batch
            x = torch.cat([numb1, numb2], dim=1)

            test_result = test(x.to(device), label.to(device), label_digits, model, device)
            val_acc += test_result[0]
            val_digit_acc += test_result[1]

        val_accuracy = val_acc / len(val_loader)
        val_digit_accuracy = val_digit_acc / len(val_loader)
        epoch_time = end_epoch_time - start_epoch_time
        test_time = time.time() - end_epoch_time

        prefix = 'Test' if config['test'] else 'Val'

        print(f"{prefix} accuracy: {val_accuracy} {prefix}",
              f"{prefix} Digit: {val_digit_accuracy} Epoch time: {epoch_time} {prefix} time: {test_time}")

        wdb_prefix = 'test' if config['test'] else 'val'
        wandb.log({
            # "epoch": epoch,
            f"{wdb_prefix}_accuracy": val_accuracy,
            f"{wdb_prefix}_digit_accuracy": val_digit_accuracy,
            f"{wdb_prefix}_time": test_time,
            f"{wdb_prefix}_target": val_accuracy + val_digit_accuracy,
            "epoch_time": epoch_time,
        })