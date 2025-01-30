import os
import random
from typing import *
from tqdm import tqdm
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from argparse import ArgumentParser
import wandb

import blackbox
import sys
import pathlib
sys.path.insert(0, f'{pathlib.Path(__file__).parent.parent.absolute()}')
from mnist_config import mnist_sum_loader, MNISTSumNet

EPS = 1e-8

def sum_n(*nums):
  return sum(nums)

class Trainer():
  def __init__(self, model, digits, dims, train_loader, test_loader, val_loader, model_dir, learning_rate, save_model=True):
    self.model_dir = model_dir
    self.all_digit = dims[-1]
    self.network = model(self.all_digit).to(device)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.val_loader = val_loader
    self.digits = digits
    self.dims = [10] + [dim_i * 9 + 1 for dim_i in dims]
    self.digit_acc = 0
    self.best_acc = 0
    self.save_model = save_model
  
  def loss(self, output, ground_truth):
    (_, dim) = output.shape
    gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in ground_truth]).to(device)
    return F.binary_cross_entropy(output, gt)

  def sub_product(self, i, *base_inputs):
    input_mapping = [blackbox.DiscreteInputMapping(list(range(self.dims[i])))]*2
    sum_2 = blackbox.BlackBoxFunction(
      sum_n,
      tuple(input_mapping),
      blackbox.DiscreteOutputMapping(list(range(self.dims[i+1]))),
      sample_count=config["sample_count"],
      loss_aggregator='add_mult')
    return sum_2(*base_inputs)

  def program(self, *inputs):
    ps = inputs
    iters = self.all_digit
    
    for i, n_i in enumerate(self.digits):  
      ps2 = []
      iters = iters // n_i
      for j in range(iters):
        output = self.sub_product(i, *ps[j*n_i : (j+1)*n_i])
        ps2.append(output)
      ps = tuple(ps2)
    output = ps[0]
    return output

  def train_epoch(self, epoch):
    self.network.train()
    train_loss = 0
    print(f"[Train Epoch {epoch}]")
    for (data, target, _) in train_loader:
      self.optimizer.zero_grad()
      data = tuple([data_i.to(device) for data_i in data])
      target = target.to(device)
      output = self.network(data)
      output = self.program(*output)
      loss = self.loss(output.to(device), target)
      train_loss += loss
      loss.backward()
      self.optimizer.step()
      torch.cuda.empty_cache()
    return train_loss

  def test_epoch(self, epoch):
    self.network.eval()
    num_items = len(self.val_loader.dataset)
    correct = 0
    digit_correct = 0
    with torch.no_grad():
      for (data, target, digits) in self.val_loader:
        output_t = self.network(tuple([data_i.to(device) for data_i in data]))
        output = torch.stack(output_t, dim=0).argmax(dim=-1)
        pred = output.sum(dim=0).to(device)
        target = target.to(device)
        correct += (pred == target).sum()
        digit_correct += (output.t().flatten() == digits.flatten().to(device)).sum()
    perc = float(correct/num_items)
    digit_perc = float(digit_correct / (num_items * self.all_digit))
    print(f"[Test Epoch {epoch}] Accuracy: {correct}/{num_items} ({perc * 100:.2f}%) Digit Accuracy ({digit_perc * 100:.2f}%)")
    torch.cuda.empty_cache()
    return perc, digit_perc

  def test(self):
    # self.network.load_state_dict(torch.load(self.model_dir+"/best.pth"))
    self.network.eval()
    num_items = len(self.test_loader.dataset)
    correct = 0
    digit_correct = 0
    with torch.no_grad():
      for (data, target, digits) in self.test_loader:
        data = tuple([data_i.to(device) for data_i in data])
        target = target.to(device)
        output_t = self.network(data)
        output = torch.stack(output_t, dim=0).argmax(dim=-1)
        pred = output.sum(dim=0).to(device)
        correct += (pred == target).sum()
        digit_correct += (output.t().flatten() == digits.flatten().to(device)).sum()
      perc = float(correct/num_items)
      digit_perc = float(digit_correct / (num_items * self.all_digit))
      print(f"[Final Acc] {int(correct)}/{int(num_items)} ({perc * 100:.2f})% Digit Accuracy ({digit_perc * 100:.2f}%)")
    wandb.log({"final acc": perc, "final digit acc": digit_perc})
    torch.cuda.empty_cache()
    return perc, digit_perc

  def train(self, n_epochs):  
    for epoch in range(1, n_epochs+1):
      t0 = time()
      train_loss = self.train_epoch(epoch)
      t1 = time()
      test_acc, test_digit_acc = self.test_epoch(epoch)
      t2 = time()
      
      if self.best_acc < test_acc:
        self.best_acc = test_acc
        perc, digit_perc = self.test()
        wandb.log({
        "train_loss": train_loss,
        "test_acc": test_acc,
        "test_digit_acc": test_digit_acc,
        "train time": t1 - t0,
        "test time": t2 - t1,
        "epoch time": t2 - t0,
        "final acc": perc,
        "final digit acc": digit_perc})  
      else:
        wandb.log({
        "train_loss": train_loss,
        "test_acc": test_acc,
        "test_digit_acc": test_digit_acc,
        "train time": t1 - t0,
        "test time": t2 - t1,
        "epoch time": t2 - t0}) 

if __name__ == "__main__":
    # Argument parser
    parser = ArgumentParser("mnist_sum")
    parser.add_argument("--n-epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--sample-count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--depth", type=int, default=1)
    args = parser.parse_args()

    # Parameters
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    depth = args.depth
    sample_count = args.sample_count
    seed = args.seed

    digits = [2] * depth
    dims = [2 ** (1 + i) for i in range(depth)]

    torch.manual_seed(seed)
    random.seed(seed)

    if torch.cuda.is_available(): device = torch.device('cuda')
    else: device = torch.device('cpu')

    # Data
    data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
    model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), f"../../model/ised_tree{depth}_{seed}"))
    os.makedirs(model_dir, exist_ok=True)

    # Dataloaders
    train_loader, val_loader, test_loader = mnist_sum_loader(data_dir, batch_size, dims[-1])
    print(len(train_loader.dataset))
    print(len(val_loader.dataset))
    print(len(test_loader.dataset))
    trainer = Trainer(MNISTSumNet, digits, dims, train_loader, test_loader, val_loader, model_dir, learning_rate)

    # setup wandb
    config = vars(args)
    config["seed"] = seed
    config['sample_count'] = sample_count
    run = wandb.init(
      project=f"ised_tree{depth}",
      name=f"{seed}",
      config=config)
    print(config)
    
    # Create trainer and train
    trainer.train(n_epochs)
    run.finish()