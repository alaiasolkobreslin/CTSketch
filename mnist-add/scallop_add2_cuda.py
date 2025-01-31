import argparse
import time

import yaml
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
from torch import nn
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
    acc = (pred == label.to(device)).sum()
    digit_acc = (output == label_digits_l).float().mean()
    return acc, digit_acc

class ScallopAddNNet(nn.Module):
    
    def __init__(self, N, provenance, k, dispatch):
        super(ScallopAddNNet, self).__init__()
        self.N = N
        self.provenance = provenance
        self.k = k
        self.dispatch = dispatch
        self.MNIST_Net = MNIST_Net()
        self.scallop_ctx_init()
        
    
    def scallop_ctx_init(self):
        self.output_dim = self.N * 99 + 1
        self.scl_ctx = scallopy.ScallopContext(provenance=self.provenance, k=self.k)
        self.scl_ctx.add_relation(f"digit_1", int, input_mapping=list(range(10)))
        self.scl_ctx.add_relation(f"digit_2", int, input_mapping=list(range(10)))
        self.scl_ctx.add_relation(f"digit_3", int, input_mapping=list(range(10)))
        self.scl_ctx.add_relation(f"digit_4", int, input_mapping=list(range(10)))
        self.scl_ctx.add_rule("sum(e) :- digit_1(a), digit_2(b), digit_3(c), digit_4(d), e == a * 10 + b + c * 10 + d")
        self.addN = self.scl_ctx.forward_function("sum", output_mapping=[(i,) for i in range(self.output_dim)], dispatch=self.dispatch)
        
    def forward(self, x):
        logits = self.MNIST_Net(x)
        output = self.addN(digit_1=logits[:, 0], digit_2=logits[:, 1], digit_3=logits[:, 2], digit_4=logits[:, 3])
        batch_size = output.shape[0]
        xs = torch.arange(self.output_dim).to(device)
        ts = []
        for i in range(self.N + 1):
            inds = (torch.floor_divide(xs, 10**i) % 10).repeat(batch_size, 1)
            if i == self.N: 
                ts.append(torch.zeros(batch_size, 2).to(device).scatter_add_(1, inds, output))
            else: 
                ts.append(torch.zeros(batch_size, 10).to(device).scatter_add_(1, inds, output))
        return ts

class Trainer():
    def __init__(self, train_loader, test_loader, config):
        self.N = config["N"]
        self.provenance = config["provenance"]
        self.k = config["k"]
        self.dispatch = config["dispatch"]
        self.network = ScallopAddNNet(self.N, self.provenance, self.k, self.dispatch).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config["perception_lr"])
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_fn = F.binary_cross_entropy
        
    def train_epoch(self, epoch, log_iterations):
        self.network.train()
        print("----------------------------------------")
        print("NEW EPOCH", epoch)
        cum_loss_percept = 0
        train_acc = 0
        train_digit_acc = 0

        start_epoch_time = time.time()

        for i, batch in enumerate(train_loader):            
            self.optimizer.zero_grad()
            # label_digits is ONLY EVER to be used during testing!!!
            numb1, numb2, target, label_digits = batch

            x = torch.cat([numb1.to(device), numb2.to(device)], dim=1)
            ts = self.network(x)
            rs = []
            for j in range(self.N + 1):
                t_i = torch.floor_divide(target.to(device), 10**j) % 10
                if j == self.N:
                    l = F.one_hot(t_i, num_classes=2).float()
                else:
                    l = F.one_hot(t_i, num_classes=10).float()
                r = F.binary_cross_entropy(ts[j], l)
                rs.append(r)
            rs = torch.stack(rs, dim=0)
            loss = rs.mean(dim=0)
            loss.backward()
            self.optimizer.step()

            cum_loss_percept += loss.item()
            
            test_result = test(x, target, label_digits, self.network.MNIST_Net, device)
            train_acc += test_result[0]
            train_digit_acc += test_result[1]
            
            if (i + 1) % log_iterations == 0:
                print(f"train_acc: {train_acc}")
                print(f"train_digit_acc: {train_digit_acc}")
                
                wandb.log({
                    # "epoch": epoch,
                    "percept_loss": cum_loss_percept / log_iterations,
                    "train_accuracy": train_acc / log_iterations,
                    "train_digit_accuracy": train_digit_acc / log_iterations,
                })  
            
        end_epoch_time = time.time()

    def test_epoch(self, epoch):
        print("----- TESTING -----")
        val_acc = 0.
        val_acc_prior = 0.
        val_explain_acc = 0.
        val_digit_acc = 0.
        
        for i, batch in enumerate(self.test_loader):
            numb1, numb2, label, label_digits = batch
            x = torch.cat([numb1, numb2], dim=1)

            test_result = test(x.to(device), label.to(device), label_digits, self.network.MNIST_Net, device)
            val_acc += test_result[0]
            val_digit_acc += test_result[1]
    
    def train(self, n_epochs, log_iterations):
        for epoch in range(1, n_epochs + 1):
            self.train_epoch(epoch, log_iterations)
            self.test_epoch(epoch)

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
    # model = MNIST_Net().to(device)
    # percept_optimizer = torch.optim.Adam(model.parameters(), lr=config["perception_lr"])

    if config["test"]:
        train_set = op(config["N"], "full_train")
        val_set = op(config["N"], "test")
    else:
        train_set = op(config["N"], "train")
        val_set = op(config["N"], "val")

    train_loader = DataLoader(train_set, config["batch_size"], False)
    val_loader = DataLoader(val_set, config["batch_size_test"], False)
    log_iterations = len(train_loader) // config["log_per_epoch"]
    if config["DEBUG"]:
        torch.autograd.set_detect_anomaly(True)
    trainer = Trainer(train_loader, val_loader, config)
    trainer.train(config["epochs"], log_iterations)
