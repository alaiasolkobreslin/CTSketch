import os
import random
from typing import *
from tqdm import tqdm
from time import time

import torch
import torch.nn.functional as F
import torch.optim as optim

from argparse import ArgumentParser
import wandb

import sys
import pathlib
sys.path.insert(0, f'{pathlib.Path(__file__).parent.parent.absolute()}')
from mnist_config import MNISTSumNet, mnist_sum_loader
from tensor_sketch import TensorSketch
from sketch_configs import full_theta2, full_theta3, full_theta4, sample_theta

EPS = 1e-8

def full_theta(digits, input_dim, output_dim, samples):
  if digits == 4:
    return full_theta4(digits, input_dim, output_dim, samples)
  elif digits == 3:
    return full_theta3(digits, input_dim, output_dim, samples)
  elif digits == 2:
    return full_theta2(digits, input_dim, output_dim, samples)
  else: 
    raise Exception("not implemented")

class Trainer():
  def __init__(self, model, tensor_method, digits, dims, train_loader, test_loader, val_loader, model_dir, learning_rate, save_model=True):
    self.model_dir = model_dir
    self.all_digit = dims[-1]
    self.network = model(self.all_digit).to(device)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.val_loader = val_loader
    self.digits = digits
    self.dims = [10] + [dim_i * 9 + 1 for dim_i in dims]
    self.tensorsketch = TensorSketch(tensor_method)
    self.t = [None] * len(digits)
    self.digit_acc = 0
    self.best_acc = 0
    self.save_model = save_model
    self.loss_fn = F.l1_loss

  def pretrain(self):
    t0 = time()
    err = 0
    for i, digit_i in enumerate(self.digits):
      gt = full_theta(digit_i, self.dims[i], self.dims[i + 1], 0)
      rerr1, cores1, _ = self.tensorsketch.approx_theta({'gt': gt, 'rank': 2})
      rerr2, cores2, _ = self.tensorsketch.approx_theta({'gt': gt, 'rank': 2})
      if rerr2 is None or rerr1 < rerr2: rerr, cores = rerr1, cores1
      else: rerr, cores = rerr2, cores2
      self.t[i] = [cores_i.to(device) for cores_i in cores]
      err += rerr
    t1 = time()
    wandb.log({"pretrain": t1 - t0, "rerr": err})
    return rerr

  def sub_product(self, i, *base_inputs):
    t = self.t[i]
    output = torch.einsum('ijk, bj -> bk', t[0], base_inputs[0])
    for c1, c2 in zip(t[1:], base_inputs[1:]):
      output = torch.einsum('bi, ika, bk -> ba', output, c1, c2)
    return output

  def program(self, *inputs):
    ps = inputs
    iters = self.all_digit
    batch_size = inputs[0].shape[0]
    
    for i, n_i in enumerate(self.digits):  
      ps2 = []
      iters = iters // n_i
      for j in range(iters):
        output = self.sub_product(i, *tuple(ps[j*n_i : (j+1)*n_i]))
        if i == len(self.digits) - 1: ps2.append(output[:, 0])
        else:
          xs = torch.arange(self.dims[i+1]).repeat(batch_size, 1).to(device)
          sigma = max((1 / ps[j*n_i : (j+1)*n_i][0].max()), 4)
          distr_output = torch.exp(- (xs - output)**2 / sigma)
          distr_output = F.normalize(distr_output, dim=-1, p=1)
          ps2.append(distr_output)
      ps = tuple(ps2)
    output = ps[0]
    return output

  def train_epoch(self, epoch):
    self.network.train()
    num_items = 0
    total_correct = 0
    train_loss = 0
    iter = tqdm(self.train_loader, total=len(self.train_loader))
    for e, (data, target, _) in enumerate(iter):
      self.optimizer.zero_grad()
      output_t = self.network(tuple([data_i.to(device) for data_i in data]))
      output = self.program(*tuple(output_t))
      loss = self.loss_fn(output, target.to(device).float())
      train_loss += loss
      loss.backward()
      self.optimizer.step()
      total_correct += (output.round()==target.to(device)).float().sum()
      num_items += output.shape[0]
      correct_perc = 100. * total_correct / num_items
      iter.set_description(f"[Train {epoch}] Loss: {loss.item():.4f} Accuracy: {correct_perc:.4f}%")
    return train_loss

  def test_epoch(self, epoch):
    self.network.eval()
    num_items = len(self.val_loader.dataset)
    correct = 0
    digit_correct = 0
    digits_pred = []
    digits_gt = []
    with torch.no_grad():
      iter = tqdm(self.val_loader, total=len(self.val_loader))
      for (data, target, digits) in iter:
        output_t = self.network(tuple([data_i.to(device) for data_i in data]))
        output = torch.stack(output_t, dim=0).argmax(dim=-1)
        pred = output.sum(dim=0).to(device)
        digits_pred.extend(output.t().flatten().cpu())
        digits_gt.extend(digits.flatten().cpu())
        target = target.to(device)
        correct += (pred == target).sum()
        digit_correct += (output.t().flatten() == digits.flatten().to(device)).sum()
        perc = float(correct/num_items)
        digit_perc = float(digit_correct / (num_items * self.all_digit))
        iter.set_description(f"[Test Epoch {epoch}] Accuracy: {correct}/{num_items} ({perc * 100:.2f}%) Digit Accuracy ({digit_perc * 100:.2f}%)")
    
    return perc, digit_perc

  def test(self):
    self.network.eval()
    num_items = len(self.test_loader.dataset)
    correct = 0
    digit_correct = 0
    with torch.no_grad():
      for (data, target, digits) in self.test_loader:
        target = target.to(device)
        output_t = self.network(tuple([data_i.to(device) for data_i in data]))
        output = torch.stack(output_t, dim=0).argmax(dim=-1)
        pred = output.sum(dim=0).to(device)
        correct += (pred == target).sum()
        digit_correct += (output.t().flatten() == digits.flatten().to(device)).sum()
      perc = float(correct/num_items)
      digit_perc = float(digit_correct / (num_items * self.all_digit))
      print(f"[Final Acc] {int(correct)}/{int(num_items)} ({perc * 100:.2f})% Digit Accuracy ({digit_perc * 100:.2f}%)")
    return perc, digit_perc

  def train(self, n_epochs):
    err = self.pretrain()
    if not (err > 0): n_epochs = 0
    for epoch in range(1, n_epochs+1):
      t0 = time()
      train_loss = self.train_epoch(epoch)
      t1 = time()
      test_acc, test_digit_acc = self.test_epoch(epoch)
      t2 = time()

      if self.save_model and self.best_acc < test_acc:
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
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--method", type=str, default='tt')
    parser.add_argument("--loss", type=str, default='l1')  
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--depth", type=int, default=1)
    args = parser.parse_args()

    # Parameters
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    depth = args.depth
    method = args.method
    loss = args.loss
    seed = args.seed

    digits = [2] * depth
    dims = [2 ** (1 + i) for i in range(depth)]

    torch.manual_seed(seed)
    random.seed(seed)

    if torch.cuda.is_available(): device = torch.device('cuda')
    else: device = torch.device('cpu')

    # Data
    data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
    model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), f"../../model/tensor_tree{depth}_{seed}"))
    os.makedirs(model_dir, exist_ok=True)

    # Dataloaders
    train_loader, val_loader, test_loader = mnist_sum_loader(data_dir, batch_size, dims[-1])

    # setup wandb
    config = vars(args)
    config["seed"] = seed
    run = wandb.init(
      project=f"sketch_tree{depth}",
      name=f"{seed}",
      config=config)
    print(config)

    # Create trainer and train
    trainer = Trainer(MNISTSumNet, method, digits, dims, train_loader, test_loader, val_loader, model_dir, learning_rate)
    trainer.train(n_epochs)
    run.finish()