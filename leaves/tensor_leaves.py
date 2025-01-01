import os
import random
from typing import *
import itertools
from time import time

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn

from argparse import ArgumentParser
import wandb

from leaves_config import leaves_loader, LeafNet, classify_11, classify_llm
from leaves_config import l11_margin, l11_shape, l11_texture, l11_labels
from leaves_config import l11_4_one, l11_4_two, l11_4_three

def program_dt():
    dims = len(l11_margin), len(l11_shape), len(l11_texture)
    rs = [list(range(d_i)) for d_i in dims]
    rs = list(itertools.product(*rs))
    gt = torch.zeros(tuple(dims))
    for (m, s, t) in rs:
      l = classify_11(l11_margin[m], l11_shape[s], l11_texture[t])
      gt[m, s, t] = l11_labels.index(l)
    return gt.long()

def program_gpt():
    dims = len(l11_4_one), len(l11_4_two), len(l11_4_three)
    rs = [list(range(d_i)) for d_i in dims]
    rs = list(itertools.product(*rs))
    gt = torch.zeros(tuple(dims))
    for (m, s, t) in rs:
      l = classify_llm(l11_4_one[m], l11_4_two[s], l11_4_three[t])
      gt[m, s, t] = l11_labels.index(l)
    return gt.long()

class LeavesNet(nn.Module):
  def __init__(self):
    super(LeavesNet, self).__init__()
    
    # features for classification
    self.net1 = LeafNet(len(l11_margin))
    self.net2 = LeafNet(len(l11_shape))
    self.net3 = LeafNet(len(l11_texture))

  def forward(self, x):
    has_margin = self.net1(x)
    has_shape = self.net2(x)
    has_texture = self.net3(x)
    return has_margin, has_shape, has_texture

class LeavesGPTNet(nn.Module):
  def __init__(self):
    super(LeavesGPTNet, self).__init__()

    # features for classification
    self.net1 = LeafNet(len(l11_4_one))
    self.net2 = LeafNet(len(l11_4_two))
    self.net3 = LeafNet(len(l11_4_three))

  def forward(self, x):
    has_f1 = self.net1(x)
    has_f2 = self.net2(x)
    has_f3 = self.net3(x)
    return has_f1, has_f2, has_f3 

class Trainer():
  def __init__(self, method, train_loader, test_loader, val_loader, learning_rate, gpu, model_dir, save_model=True):
    if gpu >= 0 and torch.cuda.is_available(): device = torch.device("cuda:%d" % gpu)
    else: device = torch.device("cpu")
    self.device = device
    self.method = method
    if method == "dt": self.network = LeavesNet().to(device)
    else: self.network = LeavesGPTNet().to(device)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.val_loader = val_loader
    self.save_model = save_model
    self.model_dir = model_dir
    self.best_acc = 0

  def pretrain(self):
    t0 = time()
    if self.method == 'dt': self.gt = program_dt()
    else: self.gt = program_gpt()
    t1 = time()
    wandb.log({"pretrain": t1 - t0})
  
  def program(self, *ps):
    gt = F.one_hot(self.gt, num_classes=len(l11_labels)).float().to(self.device)
    output = torch.einsum('ia, ib, ic, abcd -> id', ps[0], ps[1], ps[2], gt)
    return output

  def gt_program(self, *ps):
    ms = F.one_hot(ps[0].argmax(dim=-1), num_classes=ps[0].shape[1]).float()
    ss = F.one_hot(ps[1].argmax(dim=-1), num_classes=ps[1].shape[1]).float()
    ts = F.one_hot(ps[2].argmax(dim=-1), num_classes=ps[2].shape[1]).float()
    return self.program(ms, ss, ts)

  def loss_fn(self, output, ground_truth):
    dim = output.shape[1]
    gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in ground_truth]).to(self.device)
    return F.cross_entropy(output, gt)
  
  def train_epoch(self, epoch):
    self.network.train()
    num_items = 0
    total_correct = 0
    train_loss = 0
    for (i, (input, target)) in enumerate(self.train_loader):
      self.optimizer.zero_grad()
      input = input.to(self.device)
      target = target.to(self.device)
      output_t = self.network(input)
      output = self.program(*output_t)
      output = F.normalize(output, dim=-1)
      loss = self.loss_fn(output, target)
      loss.backward()
      self.optimizer.step()
      total_correct += (output.argmax(dim=1)==target).float().sum()
      num_items += output.shape[0]
      correct_perc = 100. * total_correct / num_items
      train_loss += loss.item()
      avg_loss = train_loss / (i + 1)
    print(f"[Train Epoch {epoch}] Loss: {avg_loss:.4f}, Overall Accuracy: {int(total_correct)}/{int(num_items)} {correct_perc:.2f}%")
    return train_loss

  def test_epoch(self, epoch):
    self.network.eval()
    num_items = 0
    num_correct = 0
    with torch.no_grad():
      for i, (input, target) in enumerate(val_loader):
        input = input.to(self.device)
        target = target.to(self.device)
        output_t = self.network(input)
        output = self.gt_program(*output_t)
        num_items += output.shape[0]
        num_correct += (output.argmax(dim=1)==target).float().sum()
      
      perc = float(num_correct/num_items)
      print(f"[Test Epoch {epoch}] {int(num_correct)}/{int(num_items)} ({perc * 100:.2f})%")
      
    return perc

  def train(self, n_epochs):
    self.pretrain()
    for epoch in range(1, n_epochs+1):
      t0 = time()
      train_loss = self.train_epoch(epoch)
      t1 = time()
      test_acc = self.test_epoch(epoch)
      t2 = time()

      wandb.log({
        "train_loss": train_loss,
        "test_acc": test_acc,
        "train time": t1 - t0,
        "test time": t2 - t1,
        "epoch time": t2 - t0})
      
      if self.save_model and self.best_acc < test_acc:
        self.best_acc = test_acc
        # torch.save(self.network.state_dict(), self.model_dir+"/best.pth")
        self.test()

  def test(self):
    self.network.load_state_dict(torch.load(self.model_dir+"/best.pth", weights_only=True))
    self.network.eval()
    num_items = 0
    num_correct = 0
    with torch.no_grad():
      for (input, target) in test_loader:
        input = input.to(self.device)
        target = target.to(self.device)
        output_t = self.network(input)
        output = self.gt_program(*output_t)
        num_items += output.shape[0]
        num_correct += (output.argmax(dim=1)==target).float().sum()
      perc = float(num_correct/num_items)
      print(f"[Final Acc] {int(num_correct)}/{int(num_items)} ({perc * 100:.2f})%")
    wandb.log({"final acc": perc})

if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("leaves")
  parser.add_argument("--n-epochs", type=int, default=100)
  parser.add_argument("--gpu", type=int, default=0)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--train-num", type=int, default=30)
  parser.add_argument("--val-num", type=int, default=10)
  parser.add_argument("--test-num", type=int, default=10)
  parser.add_argument("--learning-rate", type=float, default=1e-4)
  # parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--data-dir", type=str, default="leaf_11")
  parser.add_argument("--method", type=str, default="gpt")
  parser.add_argument("--cuda", action="store_true")
  args = parser.parse_args()

  for seed in [1234, 3177]:

    # Setup parameters
    torch.manual_seed(seed)
    random.seed(seed)

    # Load data
    data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
    model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../model/leaves_"+args.method))
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)
    (train_loader, val_loader, test_loader) = leaves_loader(data_root, args.data_dir, args.batch_size, args.train_num, args.test_num, args.val_num)
    trainer = Trainer(args.method, train_loader, test_loader, val_loader, args.learning_rate, args.gpu, model_dir)

    # setup wandb
    config = vars(args)
    run = wandb.init(
      project=f"tensor_leaf{args.method}",
      name = f'{seed}',
      config=config)
    print(config)

    # Run
    trainer.train(args.n_epochs)

    run.finish()