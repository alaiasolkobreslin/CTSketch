import torch
import torch.nn.functional as F
import torch.nn as nn

import math
import os
import json
import numpy as np
import time
from argparse import ArgumentParser

from datasets import SudokuDataset_RL
from models.transformer_sudoku import get_model
from sudoku_solver.board import Board
from sudoku_config import pretrain, distinct

import sys
import pathlib
sys.path.insert(0, f'{pathlib.Path(__file__).parent.parent.parent.absolute()}')
from tensor_sketch import TensorSketch

try:
    from pyswip import Prolog
except Exception:
    print('-->> Prolog not installed')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tensorsketch = TensorSketch('tt')
EPS = 1e-8

def program_oh(solution_boards, masking_boards, target):
    gt = target.argmax(dim=-1).t()
    masking_prob = masking_boards.sigmoid().unsqueeze(-1)
    logits = torch.cat(((1 - masking_prob), solution_boards * masking_prob), dim=-1)
  
    output = []
    for i, gt_i in enumerate(gt):
        t = []
        for j in range(3):
            ns = distinct(i, j)
            p = torch.einsum('ijk, bj -> bk', cores[0], logits[:, ns[0]])
            for c1, n2 in zip(cores[1:], ns[1:]):
                c2 = logits[:, n2]
                p = torch.einsum('bi, ika, bk -> ba', p, c1, c2)
            p = torch.einsum('bi, ika -> bk', p, cores[-1])
            t.append(p)
        t = (torch.stack(t, dim=-1)) # b x 3
        t = torch.where(torch.arange(9).unsqueeze(0).unsqueeze(-1).to(device) == gt_i.unsqueeze(-1).unsqueeze(-1), t, 0).sum(dim=-1)
        output.append(t)
    output = (torch.stack(output, dim=-1) + EPS).log()
    output = output.sum(dim=-1).sum(dim=-1).mean()
    loss = -output
    return loss

def loss_fn(solution_boards, masking_boards, target):
  prolog_instance = Prolog()
  prolog_instance.consult("sudoku/src/sudoku_solver/sudoku_prolog.pl") 
  final_boards = []
  rewards = []

  ground_truth_boards = torch.argmax(target,dim=2) # b x 81
  masking_prob = masking_boards.sigmoid().unsqueeze(-1)
  logits = torch.cat((masking_prob, solution_boards * (1-masking_prob)), dim=-1)
  cleaned_boards = logits.argmax(dim=-1) # b x 81

  for i in range(len(cleaned_boards)):
    board_to_solver = Board(cleaned_boards[i].reshape((9,9)).int().cpu())
    try:
      solver_success = board_to_solver.solve(solver ='prolog', prolog_instance=prolog_instance)
    except StopIteration:
      solver_success = False
    final_solution = torch.from_numpy(board_to_solver.board.reshape(81,)).to(device)
    if not solver_success:
      final_solution = cleaned_boards[i].to(device)
    reward = torch.where(final_solution == ground_truth_boards[i], 1, 0).sum() / 81
    rewards.append(reward)
    final_boards.append(final_solution)
  return torch.stack(rewards).mean(), final_boards

class Trainer():
  def __init__(self, model, train_loader, test_loader, model_dir, learning_rate, batch_size, seed, args):
    self.model_dir = model_dir
    self.network = model
    self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=learning_rate, weight_decay=args.weight_decay)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.best_loss = None
    self.best_reward = None
    self.batch_size = batch_size
    self.args = args
    self.seed = seed
  
  def adjust_learning_rate(self, epoch):
    lr = self.args.lr
    if hasattr(self.args, 'warmup') and epoch < self.args.warmup:
        lr = lr / (self.args.warmup - epoch)
    elif not self.args.disable_cos:
        lr *= 0.5 * (1. + math.cos(math.pi * (epoch - self.args.warmup) / (self.args.epochs - self.args.warmup)))
    for param_group in self.optimizer.param_groups:
        param_group['lr'] = lr
    return lr

  def train_epoch(self, epoch):
    self.network.train()
    for i, (data, target) in enumerate(self.train_loader):
      self.optimizer.zero_grad()
      data = data.to(device)
      target = target.to(device)
      board, mask = self.network(data)
      loss = program_oh(board, mask, target)
      loss.backward()
       
      self.optimizer.step()
      torch.cuda.empty_cache() 

    print(f"Epoch {epoch} Loss {loss.item()}")

  def test_epoch(self, epoch):
    reward = 0
    n = len(self.test_loader)
    self.network.eval()
    with torch.no_grad():
      for i, (data, target) in enumerate(self.test_loader):
        data = data.to(device)
        target = target.to(device)
        board, mask = self.network(data)
        r, pred = loss_fn(board, mask, target)
        reward += float(r * data.shape[0])
        torch.cuda.empty_cache()
        print(i)
      avg_reward = reward/n 
      print(f'----[rl][Epoch {epoch}] \t \t AvgReward {avg_reward:.4f}')

    if self.best_reward is None or reward > self.best_reward:
      self.best_reward = reward
      torch.save(self.network.state_dict(), f'{self.model_dir}/checkpoint_best_R_{self.seed}.pth')

  def train(self, n_epochs):
    # self.test_epoch(0)
    for epoch in range(1, n_epochs + 1):
      self.adjust_learning_rate(epoch)
      time1 = time.time()
      self.train_epoch(epoch)
      time2 = time.time()
      print(time2 - time1)
      time1 = time.time()
      self.test_epoch(epoch)
      time2 = time.time()
      torch.save(model.state_dict(), f'{self.model_dir}/checkpoint_{self.seed}_{epoch}.pth')

if __name__ == "__main__":
    parser = ArgumentParser('sudoku_cts')
    parser.add_argument('--solver', type=str, default='prolog')
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--seed', default=1234, type=int)

    parser.add_argument('--block-len', default=81, type=int)
    parser.add_argument('--data', type=str, default='satnet')
    parser.add_argument('--noise-setting', default='xxx/yyy.json', type=str)
    parser.add_argument('--train-only-mask', default = False, type = bool)
    parser.add_argument('--print-freq', default=100, type=int)

    parser.add_argument('--epochs', default=9, type=int)
    parser.add_argument('--warmup', default=10, type=int)
    parser.add_argument('-b', '--batch-size', default=200, type=int)
    parser.add_argument('--lr', default=0.000005, type=float)
    parser.add_argument('--weight-decay', default=3e-1, type=float)
    parser.add_argument('--clip-grad-norm', default=1., type=float)
    parser.add_argument('--disable-cos', action='store_true')
    args = parser.parse_args()

    gt = pretrain(device, 9).round().cpu()
    for r in [2]:
      e, cores = [], []
      rerr1, cores1, xhat1 = tensorsketch.approx_theta({'gt': gt, 'rank': r})
      e1 = (xhat1.to_numpy() - gt).abs().max()
      if e1 > 0: 
        e.append(e1)
        cores.append(cores1)
      rerr2, cores2, xhat2 = tensorsketch.approx_theta({'gt': gt, 'rank': r})
      e2 = (xhat2.to_numpy() - gt).abs().max()
      if e2 > 0: 
        e.append(e2)
        cores.append(cores2)
      rerr3, cores3, xhat3 = tensorsketch.approx_theta({'gt': gt, 'rank': r})
      e3 = (xhat3.to_numpy() - gt).abs().max()
      if e3 > 0: 
        e.append(e3)
        cores.append(cores3)  

    e = torch.stack(e)
    cores = cores[torch.argmin(e)]
    cores = [c_i.to(device) for c_i in cores]
    
    torch.manual_seed(args.seed)
    train_dataset = SudokuDataset_RL(args.data,'-train')
    test_dataset = SudokuDataset_RL(args.data,'-valid')

    # Model
    model = get_model(block_len=args.block_len)
    model.load_pretrained_models(args.data)
    model.to(device)

    model_dir = os.path.join('checkpoint', f'tensor')
    os.makedirs(model_dir, exist_ok=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)

    # load pre_trained models
    trainer = Trainer(model, train_loader, test_loader, model_dir, args.lr,  args.batch_size, args.seed, args)
    trainer.train(args.epochs)