import os
import random
from typing import *
from time import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from argparse import ArgumentParser
from tqdm import tqdm
import wandb
from create_scallop_sum import CreateMNISTScallopSum

import scallopy
from mnist_config import mnist_sum_loader, MNISTSumNet

class MNISTSumNNet(nn.Module):
  def __init__(self, provenance, k, digit, dispatch):
    super(MNISTSumNNet, self).__init__()
    
    # MNIST Digit Recognition Network
    self.mnist_net = MNISTSumNet(digit)
    self.digit = digit

    # Scallop Context
    scallop_context_creator = CreateMNISTScallopSum(digit, provenance, k)
    self.scl_ctx = scallop_context_creator.add_sum_rule()

    # The `sum_n` logical reasoning module
    self.sum_n = self.scl_ctx.forward_function(f"sum_{self.digit}", output_mapping=[(i,) for i in range(digit*9 + 1)], dispatch=dispatch)

  def forward(self, x):
    # First recognize the digits
    distrs = self.mnist_net(x)
    
    # Call the sum-n reasoning module
    args = {f"digit_{str(i+1)}" : distrs[i] for i in range(self.digit)}
    output = self.sum_n(**args)

    return output

def bce_loss(output, ground_truth):
  (_, dim) = output.shape
  gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in ground_truth]).to(device)
  return F.binary_cross_entropy(output, gt)

def nll_loss(output, ground_truth):
  return F.nll_loss(output, ground_truth)

class Trainer():
  def __init__(self, train_loader, test_loader, model_dir, learning_rate, loss, k, provenance, digits, dispatch, save_model = False):
    self.model_dir = model_dir
    self.network = MNISTSumNNet(provenance, k, digits, dispatch).to(device)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.best_acc = 0
    self.best_loss = None
    self.save_model = save_model
    self.digits = digits

    if loss == "nll": self.loss = nll_loss
    elif loss == "bce": self.loss = bce_loss
    else: raise Exception(f"Unknown loss function `{loss}`")

  def train_epoch(self, epoch):
    self.network.train()
    iter = tqdm(self.train_loader, total=len(self.train_loader))
    train_loss = 0
    for (data, target, _) in iter:
      self.optimizer.zero_grad()
      data = tuple([data_i.to(device) for data_i in data])
      target = target.to(device)
      output = self.network(data)
      loss = self.loss(output, target)
      train_loss += loss.item()
      loss.backward()
      self.optimizer.step()
      iter.set_description(f"[Train Epoch {epoch}] Loss: {loss.item():.4f}")
    return train_loss

  def test_epoch(self, epoch):
    self.network.eval()
    num_items = len(self.test_loader.dataset)
    test_loss = 0
    correct = 0
    with torch.no_grad():
      iter = tqdm(self.test_loader, total=len(self.test_loader))
      for (data, target, _) in iter:
        data = tuple([data_i.to(device) for data_i in data])
        target = target.to(device)        
        output = self.network(data)
        test_loss += self.loss(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        perc = float(correct/num_items)
        iter.set_description(f"[Test Epoch {epoch}] Total loss: {test_loss:.4f}, Accuracy: {correct}/{num_items} ({perc * 100:.2f}%)")
      
      if self.save_model and self.best_acc < perc:
        self.best_loss = test_loss
        self.best_acc = perc
        torch.save(self.network, os.path.join(model_dir, f"sum{digits}_best.pt"))
    return perc, test_loss

  def train(self, n_epochs):
    for epoch in range(1, n_epochs + 1):
      t0 = time()
      train_loss = self.train_epoch(epoch)
      t1 = time()
      test_acc, test_loss = self.test_epoch(epoch)
      t2 = time()

      wandb.log({
        "train_loss": train_loss,
        "test_acc": test_acc,
        "test_loss": test_loss,
        "train time": t1 - t0,
        "test time": t2 - t1,
        "epoch time": t2 - t0})

if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("mnist_sum")
  parser.add_argument("--n-epochs", type=int, default=100)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--digits", type=int, default=2)
  parser.add_argument("--learning-rate", type=float, default=0.0005)
  parser.add_argument("--loss-fn", type=str, default="bce")
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--provenance", type=str, default="difftopkproofs")
  parser.add_argument("--top-k", type=int, default=3)
  parser.add_argument("--dispatch", type=str, default="parallel")
  args = parser.parse_args()

  # Parameters
  n_epochs = args.n_epochs
  batch_size = args.batch_size
  digits = args.digits
  learning_rate = args.learning_rate
  loss_fn = args.loss_fn
  k = args.top_k
  provenance = args.provenance

  if torch.cuda.is_available(): device = torch.device(1)
  else: device = torch.device('cpu')

  torch.manual_seed(args.seed)
  random.seed(args.seed)

  # Data
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../model/scallop_sum"))
  os.makedirs(model_dir, exist_ok=True)

  # Dataloaders
  train_loader, val_loader, test_loader = mnist_sum_loader(data_dir, batch_size, digits)
  trainer = Trainer(train_loader, test_loader, model_dir, learning_rate, loss_fn, k, provenance, digits, args.dispatch)
  print(len(train_loader.dataset))
  print(len(test_loader.dataset))

  # setup wandb
  config = vars(args)
  wandb.init(
    project=f"scallop_sum",
    name = f"{digits}_{args.seed}",
    config=config)
  print(config)
  
  trainer.train(n_epochs)