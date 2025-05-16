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
from tensor_sketch import TensorSketch
from hwf_config import hwf_loader, SymbolNet, hwf_eval

EPS = 1e-8
THS = 1e-3
ss = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', "+", "-", "*", "/"]

def all_hwf1():
  output = torch.ones(14) * -1
  rs = [list(range(d_i)) for d_i in [15]]
  rs = list(itertools.product(*rs))
  for (a,) in rs:
    if a > 9: continue
    output[a] = a
  return 0, output.to(device)

def all_hwf3(tensorsketch):
  output = torch.zeros(14, 14, 14)
  rs = [list(range(d_i)) for d_i in [15]*3]
  rs = list(itertools.product(*rs))
  for (a, b, c) in rs:
    if b < 10 or a > 9 or c > 9: continue
    try: 
      r =  hwf_eval([ss[a], ss[b], ss[c]])
      output[a, b, c] = r
    except: continue
  output = torch.round(output, decimals=5)
  rerr1, cores1, _ = tensorsketch.approx_theta({'gt': output, 'rank': ranks[1]})
  rerr2, cores2, _ = tensorsketch.approx_theta({'gt': output, 'rank': ranks[1]})
  if rerr1 < rerr2: 
    cores = cores1
    rerr = rerr1
  else:
    cores = cores2
    rerr = rerr2
  cores = [cores_i.to(device) for cores_i in cores]  
  return rerr, cores

def all_hwf5(tensorsketch):
  output = torch.zeros(14, 14, 14, 14, 14)
  rs = [list(range(d_i)) for d_i in [15]*5]
  rs = list(itertools.product(*rs))
  for (a, b, c, d, e) in rs:
    if b < 10 or d < 10 or a > 9 or c > 9 or e > 9: continue
    try: 
      r =  hwf_eval([ss[a], ss[b], ss[c], ss[d], ss[e]])
      output[a, b, c, d, e] = r
    except: continue
  rerr1, cores1, _ = tensorsketch.approx_theta({'gt': output, 'rank': ranks[2]})
  rerr2, cores2, _ = tensorsketch.approx_theta({'gt': output, 'rank': ranks[2]})
  if rerr1 < rerr2: 
    cores = cores1
    rerr = rerr1
  else:
    cores = cores2
    rerr = rerr2
  cores = [cores_i.to(device) for cores_i in cores]
  return rerr, cores

def all_hwf(tensorsketch):
  output = torch.zeros(14, 14, 14, 14, 14, 14, 14)
  rs = [list(range(d_i)) for d_i in [15]*7]
  rs = list(itertools.product(*rs))
  for (a, b, c, d, e, f, g) in rs:
    if b < 10 or d < 10 or f < 10 or a > 9 or c > 9 or e > 9 or g > 9: continue
    try: 
      r =  hwf_eval([ss[a], ss[b], ss[c], ss[d], ss[e], ss[f], ss[g]])
      output[a, b, c, d, e, f, g] = r
    except: continue
  output = torch.round(output, decimals=5)
  rerr1, cores1, _ = tensorsketch.approx_theta({'gt': output, 'rank': ranks[3]})
  rerr2, cores2, _ = tensorsketch.approx_theta({'gt': output, 'rank': ranks[3]})
  if rerr1 < rerr2: cores, rerr = cores1, rerr1
  else: cores, rerr = cores2, rerr2
  cores = [cores_i.to(device) for cores_i in cores]
  return rerr, cores

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
  def __init__(self, tensor_method, train_loader, test_loader, model_dir, learning_rate, max_len, save_model=False):
    self.model_dir = model_dir
    self.network = SymbolNet().to(device)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.best_acc = 0
    self.seed = seed
    self.save_model = save_model
    self.loss_fn = F.l1_loss
    self.max_len = max_len
    self.tensorsketch = TensorSketch(tensor_method)
    self.tensor_method = tensor_method

  def pretrain(self):
    t0 = time()
    r, self.cs1 = all_hwf1()
    if self.max_len >= 3: 
      r3, self.cs3 = all_hwf3(self.tensorsketch)
      r += r3
    if self.max_len >= 5: 
      r5, self.cs5 = all_hwf5(self.tensorsketch)
      r += r5
    if self.max_len >= 7: 
      r7, self.cs7 = all_hwf(self.tensorsketch)
      r += r7
    t1 = time()
    print(t1 - t0)
    wandb.log({"pretrain": t1 - t0, "rerr": r})

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
  
  def program_sketch(self, d, eqn_len):
    data = [] # b x l x 14
    for i, l in enumerate(eqn_len):
      data.append(d[eqn_len[:i].sum():eqn_len[:i+1].sum()]) 
    
    output = torch.zeros(len(data)).to(device)
    for i, (data_i) in enumerate(data):
      if data[i].shape[0] == 1: 
        output[i] = torch.einsum('a, ai -> i', data_i[0], self.cs1.unsqueeze(-1))
      else:
        if data[i].shape[0] == 3: 
          cs = self.cs3
        elif data[i].shape[0] == 5: 
          cs = self.cs5
        elif data[i].shape[0] == 7: 
          cs = self.cs7

        if self.tensor_method == 'tt':
          temp = torch.einsum('ijk, j -> k', cs[0], data_i[0])
          for c1, c2 in zip(cs[1:], data_i[1:]):
            temp = torch.einsum('i, ika, k -> a', temp, c1, c2)
            output[i] = temp[0]
        
        elif self.tensor_method == 'cp':
          base_indices = [f"{chr(i+97)}" for i in range(len(data_i))]
          t_indices = [f"{chr(i+97)}r" for i in range(len(data_i))]
          equation = f"{', '.join(base_indices + t_indices)}, r ->"
          temp = torch.einsum(equation, *data_i, *cs)
          output[i] = temp

        elif self.tensor_method == 'tucker':
          cores = [f"{chr(i+97)}" for i in range(len(data_i))]
          cores = ''.join(cores)
          base_indices = [f"{chr(i+97+len(data_i))}{chr(i+97)}" for i in range(len(data_i))]
          t_indices = [f"{chr(i+97+len(data_i))}" for i in range(len(data_i))]
          equation = cores + ', ' + f"{', '.join(base_indices + t_indices)} ->"
          temp = torch.einsum(equation, *cs, *data_i)
          output[i] = temp
        
        elif self.tensor_method == 'tensor_ring':
          base_indices = [f"{chr(i+97)}" for i in range(len(data_i))]
          t_indices = [f"{chr(i+97+len(data_i))}{chr(i+97)}{chr(i+97+len(data_i)+1)}" for i in range(len(data_i)-1)]
          t_indices += [f"{chr(len(data_i)+97+len(data_i)-1)}{chr(len(data_i)+97-1)}{chr(97+len(data_i))}"]
          equation = f"{', '.join(base_indices + t_indices)} ->"
          temp = torch.einsum(equation, *data_i, *cs)
          output[i] = temp
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
      output = self.program_sketch(output_t, eqn_len)
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
      print(f"Correct: {correct} / {num_items} ({perc*100:.4f})")
      
      if self.save_model and self.best_acc < perc:
        self.best_acc = perc
        torch.save(self.network, model_dir+f"/{self.seed}_best.pkl")

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
    parser.add_argument("--n-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--method", type=str, choices=['tt', 'tensor_ring', 'cp', 'tucker'], default='cp')
    parser.add_argument("--max-len", type=int, default=7)
    parser.add_argument("--seed", type=int, default=3177)
    parser.add_argument("--rank", type=int, default=6)
    args = parser.parse_args()
 
    # Parameters
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    max_len = args.max_len
    tensor_method = args.method
    seed = args.seed
    loss = args.loss

    ranks = [3, 3, 3, 3]

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
    trainer = Trainer(tensor_method, train_loader, test_loader, model_dir, learning_rate, max_len)
    
    # Setup wandb
    config = vars(args)
    run = wandb.init(
        project=f"tensor_hwf",
        name=f'{args.rank}_{seed}',
        config=config)
    print(config)
          
    trainer.train(n_epochs)
    run.finish()