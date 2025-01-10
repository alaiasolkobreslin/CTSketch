import os
import random
from typing import *
from time import time

import torch
import torch.nn.functional as F
import torch.optim as optim

from argparse import ArgumentParser
import wandb

from mnist_config import mnist_sum_loader, MNISTSumNet

device = torch.device(0) if torch.cuda.is_available() else torch.device('cpu')

class Trainer():
  def __init__(self, model, train_loader, test_loader, model_dir, learning_rate, grad_type, dim, sample_count, task, task_type, seed):
    self.model_dir = model_dir
    self.network = model(dim, with_softmax=False).to(device)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.best_acc = None
    self.grad_type = grad_type
    self.sample_count = sample_count
    self.task = task
    self.task_type = task_type
    self.seed = seed
    self.log_it = 10
    self.digits = [dim]

  def indecater_multiplier(self, batch_size, dims, n):
    icr_mult = torch.zeros((dims, n, self.sample_count, batch_size, dims))
    icr_replacement = torch.zeros((dims, n, self.sample_count, batch_size, dims))
    for i in range(dims):
      for j in range(n):
        icr_mult[i,j,:,:,i] = 1
        icr_replacement[i,j,:,:,i] = j
    return icr_mult.to(device), icr_replacement.to(device)

  def reinforce_grads(self, logits, target):
    d = torch.distributions.Categorical(logits=logits)
    samples = d.sample((self.sample_count,))
    f_sample = (samples.sum(dim=-1) == target)
    log_p_sample = d.log_prob(samples).sum(dim=-1)
    reinforce = -(f_sample.detach() * log_p_sample).mean(dim=0)
    loss = reinforce.mean(dim=0)
    return loss
  
  def sub_indecater_grads(self, logits, dims):
    logits = torch.stack(logits, dim=1)
    output_dim = logits.shape[-1]*logits.shape[1] - 1
    d = torch.distributions.Categorical(logits=logits)
    samples = d.sample((self.sample_count,))

    outer_samples = torch.stack([samples] * logits.shape[-1], dim=0)
    outer_samples = torch.stack([outer_samples] * dims, dim=0)
    m, r = self.indecater_multiplier(logits.shape[0], dims, logits.shape[-1])
    outer_samples = outer_samples * (1 - m) + r
    outer_distr = F.one_hot(outer_samples.sum(dim=-1).long(), num_classes=output_dim).float()

    variable_distr = outer_distr.mean(dim=2).permute(2, 0, 1, 3)
    indecater_distr = variable_distr.detach() * F.softmax(logits, dim=-1).unsqueeze(-1)
    indecater_distr = indecater_distr.sum(dim=-2).sum(dim=-2)
    return indecater_distr
  
  def sub_advanced_indecater_grads(self, logits, dims):
    logits = torch.stack(logits, dim=1)
    batch_size = logits.shape[0]
    output_dim = logits.shape[-1]*logits.shape[1] - 1
    d = torch.distributions.Categorical(logits=logits)
    samples = d.sample((self.sample_count * dims,))
    samples = samples.reshape((dims, self.sample_count, batch_size, dims))
    
    outer_samples = torch.stack([samples] * logits.shape[-1], dim=1)
    m, r = self.indecater_multiplier(logits.shape[0], dims, logits.shape[-1])
    outer_samples = outer_samples * (1 - m) + r
    outer_distr = F.one_hot(outer_samples.sum(dim=-1).long(), num_classes=output_dim).float()
    
    variable_loss = outer_distr.mean(dim=2).permute(2, 0, 1, 3)
    indecater_expression = variable_loss.detach() * F.softmax(logits, dim=-1).unsqueeze(-1)
    indecater_expression = indecater_expression.sum(dim=-2).sum(dim=-2)

    return indecater_expression
  
  def indecater_grads(self, logits, grad_type):
    if grad_type == 'icr': grad_fn = self.sub_indecater_grads
    else: grad_fn = self.sub_advanced_indecater_grads

    ps = logits
    iters = len(logits)
    for i, n_i in enumerate(self.digits):
      ps2 = []
      iters = iters // n_i
      for j in range(iters):
        x = grad_fn(tuple(ps[j*n_i : (j+1)*n_i]), n_i)
        ps2.append(x)
      ps = tuple(ps2)
    return ps[0]
  
  def nll_loss(self, ps, target):
    icr_expr = torch.where(torch.arange(ps.shape[-1]).unsqueeze(0).to(device) == target.unsqueeze(-1), ps, 0).sum(dim=-1)
    loss = -torch.log(icr_expr + 1e-8)
    loss = loss.mean(dim=0)
    return loss
  
  def bce_loss(self, ps, target):
    gt = F.one_hot(target, num_classes = ps.shape[-1]).float()
    return F.binary_cross_entropy(ps, gt)

  def grads(self, logits):
    if self.grad_type == 'reinforce':
      return self.reinforce_grads(logits)
    elif self.grad_type == 'icr' or self.grad_type == 'advanced_icr':
      return self.indecater_grads(logits, grad_type)

  def train_epoch(self, epoch):
    counter = 1
    train_loss = 0
    self.network.train()
    for (data, target, _) in self.train_loader:
      self.optimizer.zero_grad()
      data = tuple([data_i.to(device) for data_i in data])
      target = target.to(device)
      logits = self.network(data)
      output = self.grads(logits)
      loss = self.nll_loss(output, target)
      train_loss += loss
      loss.backward()
      self.optimizer.step()
    #if counter % self.log_it == 0:
    #    acc = self.test()
    #    print(f"Epoch {epoch} iterations {counter}: Test accuracy: {acc}")
    print(f"[Epoch {epoch}] Loss {train_loss:.4f}")
    return train_loss

  def test(self):
    num_items = len(self.test_loader.dataset)
    correct = 0
    with torch.no_grad():
      for (data, target, _) in self.test_loader:
        data = tuple([data_i.to(device) for data_i in data])
        target = target.to(device)
        output_t = self.network(data)
        output = torch.stack(output_t, dim=0)
        output = output.argmax(dim=-1)
        pred = output.sum(dim=0)
        correct += (pred == target).sum()
      perc = float(correct/num_items)
    
    if self.best_acc is None or self.best_acc < perc:
      self.best_acc = perc
      torch.save(self.network, model_dir+f"/{self.grad_type}_{self.seed}_best.pkl")

    return perc

  def train(self, n_epochs):
    for epoch in range(1, n_epochs + 1):
      t0 = time()
      train_loss = self.train_epoch(epoch)
      t1 = time()
      test_acc = self.test()
      t2 = time()

      wandb.log({
        "train_loss": train_loss,
        "test_acc": test_acc,
        "train time": t1 - t0,
        "test time": t2 - t1,
        "epoch time": t2 - t0})
      
      print(f"Test accuracy: {test_acc}, TIME: {(t1 - t0):.4f}")

if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("mnist_r")
  parser.add_argument("--n-epochs", type=int, default=100)
  parser.add_argument("--batch-size", type=int, default=10)
  parser.add_argument("--learning-rate", type=float, default=5e-4)
  parser.add_argument("--sample-count", type=int, default=40)
  parser.add_argument("--grad_type", type=str, default='advanced_icr')
  parser.add_argument("--loss", type=str, default='nll_sum')
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--digits", type=int, default=256)
  args = parser.parse_args()

  # Parameters
  n_epochs = args.n_epochs
  batch_size = args.batch_size
  learning_rate = args.learning_rate
  sample_count = args.sample_count
  grad_type = args.grad_type
  seed = args.seed
  digits = args.digits

  torch.manual_seed(seed)
  random.seed(seed)
  task = sum
  task_type = f'sum_{digits}'

  if grad_type == 'reinforce':
    sample_count = sample_count * digits * 10

  # Data
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), f"../../model/{task_type}"))
  os.makedirs(model_dir, exist_ok=True)

  # Dataloaders
  train_loader, val_loader, test_loader = mnist_sum_loader(data_dir, batch_size, digits)
  trainer = Trainer(MNISTSumNet, train_loader, test_loader, model_dir, learning_rate, grad_type, digits, sample_count, task, task_type, seed)

  # setup wandb
  config = vars(args)
  wandb.init(
    project=f"icr_sum{digits}",
    name=f"{seed}",
    config=config)
  print(config)

  # Run
  trainer.train(n_epochs)