from functools import reduce
import os
import random
from typing import *
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim

from argparse import ArgumentParser
from sklearn.metrics import confusion_matrix
from mnist_config import MNISTSumNet, mnist_sum_loader
from tensor_sketch import TensorSketch

def full_theta4(digit, elems, output_dim, samples):
  x1 = torch.arange(0, elems).reshape(elems, 1, 1, 1)
  x2 = torch.arange(0, elems).reshape(1, elems, 1, 1)
  x3 = torch.arange(0, elems).reshape(1, 1, elems, 1)
  x4 = torch.arange(0, elems).reshape(1, 1, 1, elems)
  x = x1 + x2 + x3 + x4
  return x

def full_theta3(digit, elems, output_dim, samples):
  x1 = torch.arange(0, elems).reshape(elems, 1, 1)
  x2 = torch.arange(0, elems).reshape(1, elems, 1)
  x3 = torch.arange(0, elems).reshape(1, 1, elems)
  x = x1 + x2 + x3
  return x

def full_theta2(digit, elems, output_dim, samples):
  x1 = torch.arange(0, elems).reshape(elems, 1)
  x2 = torch.arange(0, elems).reshape(1, elems)
  x = x1 + x2
  return x

def sample_theta(digit, elems, output_dim, samples):
  xs = []
  for i in range(digit):
    xs.append(torch.randint(0, elems, (samples,)))
  xs = torch.stack(xs, dim=0)
  # t = torch.randint(0, output_dim, tuple([elems]*digit))
  t = torch.zeros(tuple([elems]*digit)).long()
  for i in range(samples):
    x_i = tuple(xs[:, i].tolist())
    t[x_i] = sum(x_i)
  return t

def full_theta(digits, input_dim, output_dim, samples):
  if digits == 4:
    return full_theta4(digits, input_dim, output_dim, samples)
  elif digits == 3:
    return full_theta3(digits, input_dim, output_dim, samples)
  elif digits == 2:
    return full_theta2(digits, input_dim, output_dim, samples)
  else: return sample_theta(digits, input_dim, output_dim, samples)

class Trainer():
  def __init__(self, model, tensor_method, digits, dims, train_loader, test_loader, model_dir, learning_rate, save_model=False):
    self.model_dir = model_dir
    self.all_digit = dims[-1]
    self.network = model(self.all_digit).to(device)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.best_loss = 10000000000
    self.digits = digits
    self.dims = [10] + [dim_i * 9 + 1 for dim_i in dims]
    self.tensorsketch = TensorSketch(tensor_method)
    self.save_model = save_model
    self.rerr = [1, 1, 1]

    self.gt = [] # cheating - shouldn't be saving this
    for i, digit_i in enumerate(self.digits):
      gt = full_theta(digit_i, self.dims[i], self.dims[i + 1], 1000) # 150000, 10000000, 250000
      self.gt.append(gt)
    self.t = [None, None, None] # self.gt

  def target_t(self, i, n, r, samples, *inputs):
    ps = []
    batch_size = inputs[0].shape[0]
    for _ in range(n):
      ps.append(torch.randint(0, r, (batch_size*samples,)))
    ps = torch.stack(ps, dim=-1)
    for sample_i in ps:
      s_i = tuple(sample_i.tolist())
      # self.t[s_i] = sum(s_i)
      self.gt[i][s_i] = sum(s_i)

  def sub_program(self, i, n, d, *base_inputs):
    t = self.t[i].to(device)
    p = base_inputs[0]
    batch_size = p.shape[0]
    for i in range(1, n):
      p1 = p.unsqueeze(-1)
      p2 = base_inputs[i].unsqueeze(1)
      eqn = f'{"".join([chr(j + 97) for j in range(0, i+2)])}, a{"".join([chr(i + 97) for i in range(i+1, i+3)])} -> {"".join([chr(j + 97) for j in range(0, i+1)])}{chr(i+97+2)}'
      p = torch.einsum(eqn, p1, p2)
    output = torch.zeros(batch_size, d).to(device).scatter_add_(1, t.flatten().repeat(batch_size, 1), p.flatten(1))
    return output
  
  def program(self, *inputs):
    ps = inputs
    digits, iters = 1, self.all_digit
    for i, (n_i, d_i) in enumerate(zip(self.digits, self.dims)):
      if self.t[i] is None:
        self.target_t(i, n_i, d_i, 100, *ps)
        digits *= n_i
        rerr, X_hat = self.tensorsketch.approx_theta({'gt': self.gt[i], 'digit': digits, 'output': self.dims[i+1], 'rank': 2})
        self.t[i] = X_hat
        self.rerr[i] = rerr
      ps2 = []
      iters = iters // n_i
      for j in range(iters):
        ps2.append(self.sub_program(i, n_i, self.dims[i+1], *tuple(ps[j*n_i : (j+1)*n_i])))
      ps = tuple(ps2)
    output = ps[0]
    rerr = sum(self.rerr)
    return rerr, output
  
  def loss(self, output, ground_truth):
    dim = output.shape[1]
    gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in ground_truth]).to(device)
    return F.cross_entropy(output, gt)
  
  def train_epoch(self, epoch):
    self.network.train()
    num_items = 0
    total_correct = 0
    total_correct2 = 0
    iter = tqdm(self.train_loader, total=len(self.train_loader))
    for (data, target, _) in iter:
      self.optimizer.zero_grad()
      output_t = self.network(tuple([data_i.to(device) for data_i in data]))
      rerr, output = self.program(*tuple(output_t))
      output = F.normalize(output, dim=-1)
      loss = self.loss(output, target.to(device))
      loss.backward()
      self.optimizer.step()
      output2 = torch.stack(output_t, dim=0).argmax(dim=-1).sum(dim=0)
      total_correct += (output2==target.to(device)).float().sum()
      total_correct2 += (output.argmax(dim=-1)==target.to(device)).float().sum()
      num_items += output.shape[0]
      correct_perc = 100. * total_correct / num_items
      correct_perc2 = 100. * total_correct2 / num_items
      iter.set_description(f"[Train {epoch}] Err:{rerr:.4f} Loss: {loss.item():.4f} Accuracy: {correct_perc:.4f}% {correct_perc2:.4f}%")

  def test_epoch(self, epoch):
    self.network.eval()
    num_items = len(self.test_loader.dataset)
    test_loss = 0
    correct = 0
    digits_pred = []
    digits_gt = []
    with torch.no_grad():
      iter = tqdm(self.test_loader, total=len(self.test_loader))
      for (data, target, digits) in iter:
        output_t = self.network(tuple([data_i.to(device) for data_i in data]))
        output = torch.stack(output_t, dim=0).argmax(dim=-1)
        digits_pred.extend(output.t().flatten().cpu())
        digits_gt.extend(digits.flatten().cpu())
        pred = output.sum(dim=0).to(device)
        target = target.to(device)
        # test_loss += self.loss(output2, target).item()
        correct += torch.where(pred == target, 1, 0).sum()
        perc = 100. * correct / num_items
        iter.set_description(f"[Test Epoch {epoch}] Accuracy: {correct}/{num_items} ({perc:.2f}%)")

      #cf_matrix = confusion_matrix(digits_gt, digits_pred)
      #print(cf_matrix)
    
    if self.save_model and test_loss < self.best_loss:
      self.best_loss = test_loss
      torch.save(self.network.state_dict(), self.model_dir+f"/best.pth")

  def train(self, n_epochs):
    for epoch in range(1, n_epochs + 1):
      self.train_epoch(epoch)
      self.test_epoch(epoch)
      
      if self.save_model: 
        torch.save(self.network.state_dict(), self.model_dir+f"/latest.pth")

if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("mnist_sum")
  parser.add_argument("--n-epochs", type=int, default=200)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--learning-rate", type=float, default=5e-4)
  parser.add_argument("--method", type=str, default='hooi')
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--jit", action="store_true")
  parser.add_argument("--digits", type=int, default=4)
  parser.add_argument("--dispatch", type=str, default="parallel")
  args = parser.parse_args()

  # Parameters
  n_epochs = args.n_epochs
  batch_size = args.batch_size
  learning_rate = args.learning_rate
  digit = args.digits
  method = args.method
  num_iters = 4
  num_iters2 = 3

  torch.manual_seed(args.seed)
  random.seed(args.seed)

  if torch.cuda.is_available(): device = torch.device('cuda')
  else: device = torch.device('cpu')

  # Data
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../data"))
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), f"../model/sum_{digit}"))
  os.makedirs(model_dir, exist_ok=True)

  # Dataloaders
  train_loader, test_loader = mnist_sum_loader(data_dir, batch_size, digit*num_iters*num_iters2)

  digits = [digit, num_iters, num_iters2]
  dims = [reduce(lambda x, y: x * y, digits[:i+1]) for i in range(len(digits))]

  # Create trainer and train
  trainer = Trainer(MNISTSumNet, method, digits, dims, train_loader, test_loader, model_dir, learning_rate)
  trainer.train(n_epochs)