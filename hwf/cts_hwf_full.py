import os
import random
from typing import *
from time import time
import itertools

import torch
import torch.nn.functional as F
import torch.optim as optim

from argparse import ArgumentParser
import wandb

import sys
import pathlib
sys.path.insert(0, f'{pathlib.Path(__file__).parent.parent.absolute()}')
from hwf_config import hwf_loader, SymbolNet, hwf_eval

EPS = 1e-8
THS = 1e-3
ss = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', "+", "-", "*", "/"]

def all_hwf1():
  output = torch.ones(14).to(device) * -1
  rs = [list(range(d_i)) for d_i in [15]]
  rs = list(itertools.product(*rs))
  for (a,) in rs:
    if a > 9: continue
    output[a] = a
  return output

def all_hwf3():
  output = torch.zeros(14, 14, 14).to(device)
  rs = [list(range(d_i)) for d_i in [15]*3]
  rs = list(itertools.product(*rs))
  for (a, b, c) in rs:
    if b < 10 or a > 9 or c > 9: continue
    try: 
      r =  hwf_eval([ss[a], ss[b], ss[c]])
      output[a, b, c] = r
    except: continue
  return output

def all_hwf5():
  output = torch.zeros(14, 14, 14, 14, 14).to(device)
  rs = [list(range(d_i)) for d_i in [15]*5]
  rs = list(itertools.product(*rs))
  for (a, b, c, d, e) in rs:
    if b < 10 or d < 10 or a > 9 or c > 9 or e > 9: continue
    try: 
      r =  hwf_eval([ss[a], ss[b], ss[c], ss[d], ss[e]])
      output[a, b, c, d, e] = r
    except: continue
  return output

def all_hwf():
  output = torch.zeros(14, 14, 14, 14, 14, 14, 14).to(device)
  rs = [list(range(d_i)) for d_i in [15]*7]
  rs = list(itertools.product(*rs))
  for (a, b, c, d, e, f, g) in rs:
    if b < 10 or d < 10 or f < 10 or a > 9 or c > 9 or e > 9 or g > 9: continue
    try: 
      r =  hwf_eval([ss[a], ss[b], ss[c], ss[d], ss[e], ss[f], ss[g]])
      output[a, b, c, d, e, f, g] = r
    except: continue
  
  output = torch.round(output, decimals=4)
  return output

def gt_program(logits, eqn_len):
  P = logits.argmax(dim=-1)
  results = []
  for i in range(len(eqn_len)):
    p_i = P[eqn_len[:i].sum():eqn_len[:i+1].sum()]
    s_i = [ss[j] for j in p_i]
    try: r = hwf_eval(s_i)
    except: r = 7000.0
    results.append(r)
  return torch.tensor(results).to(device)

class Trainer():
  def __init__(self, train_loader, test_loader, model_dir, learning_rate, seed, save_model=False):
    self.model_dir = model_dir
    self.network = SymbolNet().to(device)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.best_acc = 0
    self.seed = seed
    self.save_model = save_model
    self.loss_fn = F.l1_loss
    self.max_len = 7

  def pretrain(self):
    t0 = time()
    self.cs1 = all_hwf1()
    if self.max_len >= 3: 
      self.cs3 = all_hwf3()
    if self.max_len >= 5: 
      self.cs5 = all_hwf5()
    if self.max_len >= 7: 
      self.cs7 = all_hwf()
    t1 = time()
    print(t1 - t0)
    wandb.log({"pretrain": t1 - t0})

  def program(self, d, eqn_len):
    data = [] # b x l x 14
    for i, l in enumerate(eqn_len):
      data.append(d[eqn_len[:i].sum():eqn_len[:i+1].sum()]) 
    output = torch.zeros(len(data)).to(device)
    for i, (data_i) in enumerate(data):
      if data[i].shape[0] == 1: 
        output[i] = torch.einsum('a, ai -> i', data_i[0], self.cs1.unsqueeze(-1))
      elif data[i].shape[0] == 3: 
        output[i] = torch.einsum('a, b, c, abci -> i', data_i[0], data_i[1], data_i[2], self.cs3.unsqueeze(-1))
      elif data[i].shape[0] == 5: 
        output[i] = torch.einsum('a, b, c, d, e, abcdei -> i', data_i[0], data_i[1], data_i[2], data_i[3], data_i[4], self.cs5.unsqueeze(-1))
      else: 
        output[i] = torch.einsum('a, b, c, d, e, f, g, abcdefgi -> i', data_i[0], data_i[1], data_i[2], data_i[3], data_i[4], data_i[5], data_i[6], self.cs7.unsqueeze(-1))
    return output

  def train_epoch(self, epoch):
    train_loss = 0
    print(f"Epoch {epoch}")
    self.network.train()
    for i, (data, eqn_len, target) in enumerate(self.train_loader):
      self.optimizer.zero_grad()
      data = data.to(device)
      target = target.to(device).float()
      eqn_len = eqn_len.to(device)
      output_t = self.network(data)
      output_t = F.softmax(output_t, dim=-1)
      output = self.program(output_t, eqn_len)
      loss = self.loss_fn(output, target)
      train_loss += loss
      loss.backward()
      self.optimizer.step()
    print(train_loss.item())
    return train_loss

  def test(self):
    num_items = len(self.test_loader.dataset)
    correct = 0
    self.network.eval()
    with torch.no_grad():
      for (data, eqn_len, target) in self.test_loader:
        target = target.to(device)
        data = data.to(device)
        eqn_len = eqn_len.to(device)
        output_t = self.network(data)
        output_t = F.softmax(output_t, dim=-1)
        output = gt_program(output_t, eqn_len)
        correct += torch.where((output - target).abs() < THS, 1, 0).sum()
      perc = float(correct / num_items)
      print(f"Correct: {correct} / {num_items} ({perc * 100:.4f})")
    return perc

  def train(self, n_epochs):
    self.pretrain()
    for epoch in range(1, n_epochs + 1):
      t0 = time()
      train_loss = self.train_epoch(epoch)
      t1 = time()
      acc = self.test()
      t2 = time()

      wandb.log({
        "train_loss": float(train_loss),
        "test_acc": float(acc),
        "train time": t1 - t0,
        "test time": t2 - t1,
        "epoch time": t2 - t0})

if __name__ == "__main__":
    # Argument parser
    parser = ArgumentParser("hwf")
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-len", type=int, default=7)
    parser.add_argument("--rank", type=int, default=14)
    args = parser.parse_args()
  
    # Parameters
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    seed = args.seed
    max_len = args.max_len

    if torch.cuda.is_available(): device = torch.device('cuda')
    else: device = torch.device('cpu')

    torch.manual_seed(seed)
    random.seed(seed)

    # Data
    data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
    model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), f"../../model/hwf"))
    os.makedirs(model_dir, exist_ok=True)

    # Dataloaders
    train_loader, test_loader = hwf_loader(data_dir, batch_size, "expr", max_len)
    trainer = Trainer(train_loader, test_loader, model_dir, learning_rate, seed)
    
    # Setup wandb
    config = vars(args)
    run = wandb.init(
        project=f"tensor_hwf_rank",
        name=f'{args.rank}_{seed}',
        config=config)
    print(config)

    # Run
    trainer.train(n_epochs)
    run.finish()
