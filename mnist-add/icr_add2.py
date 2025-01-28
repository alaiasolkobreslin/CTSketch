import argparse
import time

import yaml
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import wandb

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

def indecater_multiplier(batch_size, N, pair, sample_count):
    icr_mult = torch.zeros((pair, N, sample_count, batch_size, pair))
    icr_replacement = torch.zeros((pair, N, sample_count, batch_size, pair))
    for i in range(pair):
      for j in range(N):
        icr_mult[i,j,:,:,i] = 1
        icr_replacement[i,j,:,:,i] = j
    return icr_mult.to(device), icr_replacement.to(device)

def sub_program_2(logits, samples, label):
    outer_samples = torch.stack([samples] * 10, dim=0)
    outer_samples = torch.stack([outer_samples] * 2, dim=0)
    m, r = indecater_multiplier(logits.shape[0], 10, 2, samples.shape[0])
    outer_samples = outer_samples * (1 - m) + r
    outer_loss = torch.where(outer_samples.sum(dim=-1)%10 == label, 1., 0.)

    variable_loss = outer_loss.mean(dim=2).permute(2,0,1)
    indecater_expression = variable_loss.detach() * F.softmax(logits, dim=-1)
    indecater_expression = indecater_expression.sum(dim=-1).sum(dim=-1)
    return indecater_expression

def sub_program_1(logits, samples, label):
    outer_samples = torch.stack([samples] * 10, dim=0)
    outer_samples = torch.stack([outer_samples] * 2, dim=0)
    m, r = indecater_multiplier(logits.shape[0], 10, 2, samples.shape[0])
    outer_samples = outer_samples * (1 - m) + r
    outer_loss = torch.where(torch.floor_divide(outer_samples.sum(dim=-1), 10) == label, 1., 0.)

    variable_loss = outer_loss.mean(dim=2).permute(2,0,1)
    indecater_expression = variable_loss.detach() * F.softmax(logits, dim=-1)
    indecater_expression = indecater_expression.sum(dim=-1).sum(dim=-1)
    return indecater_expression

def digit_fn4(data, target):
    n1 = torch.floor_divide(data[:, :, :, :, 0] + data[:, :, :, :, 1], 10)
    n2 = (data[:, :, :, :, 2] + data[:, :, :, :, 3])%10
    pred = n1 + n2
    acc = torch.where(pred == target, 1., 0.)
    return acc

def sub_program_4(logits, samples, target):
    outer_samples = torch.stack([samples] * 10, dim=0)
    outer_samples = torch.stack([outer_samples] * 4, dim=0)
    m, r = indecater_multiplier(logits.shape[0], 10, 4, samples.shape[0])
    outer_samples = outer_samples * (1 - m) + r
    outer_loss = digit_fn4(outer_samples, target)

    variable_loss = outer_loss.mean(dim=2).permute(2,0,1)
    indecater_expression = variable_loss.detach() * F.softmax(logits, dim=-1)
    indecater_expression = indecater_expression.sum(dim=-1).sum(dim=-1)
    return indecater_expression

def program_compose(P, target, n):
    sample_count = config["amt_samples"]
    d = torch.distributions.Categorical(logits=P)
    samples = d.sample((sample_count,))
    samples1 = samples[:, :, :n]
    samples2 = samples[:, :, n:]

    n1 = P[:, :n]
    n2 = P[:, n:]
    
    results = []
    indecater_expression = sub_program_2(torch.stack((n1[:, n-1], n2[:, n-1]), dim=1), torch.stack((samples1[:, :, n-1], samples2[:, :, n-1]), dim=-1), target%10)
    results.append(indecater_expression)
    for i in range(1, n):
        target = torch.floor_divide(target, 10)
        n_i = torch.stack((n1[:, n-i], n2[:, n-i], n1[:, n-(i+1)], n2[:, n-(i+1)]), dim=1)
        s_i = torch.stack((samples1[:, :, n-i], samples2[:, :, n-i], samples1[:, :, n-(i+1)], samples2[:, :, n-(i+1)]), dim=-1)
        indecater_expression = sub_program_4(n_i, s_i, target%10)
        results.append(indecater_expression)
    target = torch.floor_divide(target, 10)
    indecater_expression = sub_program_1(torch.stack((n1[:, 0], n2[:, 0]), dim=1), torch.stack((samples1[:, :, 0], samples2[:, :, 0]), dim=-1), target)
    results.append(indecater_expression)
    
    results = torch.stack(results, dim=0)
    loss = - torch.log(results + 1e-8).sum(dim=0)
    loss = loss.mean(dim=-1)
    return loss

if __name__ == '__main__':
    config = {
        "use_cuda": True,
        "DEBUG": False,
        "N": 1,
        "op": "add",
        "model": "full",
        "test": True,
        "batch_size": 16,
        "batch_size_test": 16,
        "amt_samples": 100,
        "perception_lr": 1e-3,
        "epochs": 30,
        "log_per_epoch": 10
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
        wandb.init(
            project=f"icr-add{config['N']}",
            config=config,
        )
        print(config)

    # Check for available GPUs
    use_cuda = config["use_cuda"] and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    op = addition
    model = MNIST_Net(with_softmax=False)
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
            loss = program_compose(output, label, config["N"])
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