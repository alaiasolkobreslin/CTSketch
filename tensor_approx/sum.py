import os
import random
from typing import *
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim

from argparse import ArgumentParser
from mnist_config import MNISTSumNet, mnist_sum_loader
from tensor_sketch import TensorSketch

def full_theta(digit, samples):
  xs = []
  for i in range(digit):
    xs.append(torch.randint(0, 10, (samples,)))
  xs = torch.stack(xs, dim=0)
  t = torch.randint(0, digit*9+1, tuple([10]*digit))
  for i in range(samples):
    x_i = tuple(xs[:, i].tolist())
    t[x_i] = sum(x_i)
  return t

class Trainer():
  def __init__(self, model, tensor_method, digit, train_loader, test_loader, model_dir, learning_rate, save_model=False):
    self.model_dir = model_dir
    self.network = model(digit).to(device)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.best_loss = 10000000000
    self.output_dim = digit*9 + 1
    self.digit = digit
    self.t = full_theta(digit, 0).detach()
    self.gt = self.t.clone() # cheating - shouldn't be saving this
    self.tensorsketch = TensorSketch(tensor_method)
    self.save_model = save_model
  
  def target_t(self, *inputs):
    ps = []
    for input_i in inputs:
      ps.append(torch.randint(0, 10, (input_i.shape[0]*self.digit*10,)).flatten())
    ps = torch.stack(ps, dim=-1)
    for sample_i in ps:
      s_i = tuple(sample_i.tolist())
      # self.t[s_i] = sum(s_i)
      self.gt[s_i] = sum(s_i)
  
  def program(self, *inputs):
    t = self.t.to(device)
    p = inputs[0]
    batch_size = p.shape[0]
    for i in range(1, self.digit):
      p1 = p.unsqueeze(-1)
      p2 = inputs[i].unsqueeze(1)
      eqn = f'{"".join([chr(j + 97) for j in range(0, i+2)])}, a{"".join([chr(i + 97) for i in range(i+1, i+3)])} -> {"".join([chr(j + 97) for j in range(0, i+1)])}{chr(i+97+2)}'
      p = torch.einsum(eqn, p1, p2)
    output = torch.zeros(batch_size, self.output_dim).to(device).scatter_add_(1, t.flatten().repeat(batch_size, 1), p.flatten(1))
    return output
  
  def loss(self, output, ground_truth):
    dim = output.shape[1]
    gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in ground_truth]).to(device)
    return F.cross_entropy(output, gt)
  
  def train_epoch(self, epoch):
    self.network.train()
    num_items = 0
    total_correct = 0
    iter = tqdm(self.train_loader, total=len(self.train_loader))
    rerr = 1.0
    for (data, target) in iter:
      self.optimizer.zero_grad()
      output_t = self.network(tuple([data_i.to(device) for data_i in data]))
      self.target_t(*tuple(output_t))
      rerr, X_hat = self.tensorsketch.approx_theta({'gt': self.gt, 'digit': self.digit})
      self.t = X_hat
      output = self.program(*tuple(output_t))
      output = F.normalize(output, dim=-1)
      loss = self.loss(output, target.to(device))
      loss.backward()
      self.optimizer.step()
      total_correct += (output.argmax(dim=1)==target.to(device)).float().sum()
      num_items += output.shape[0]
      correct_perc = 100. * total_correct / num_items
      iter.set_description(f"[Train {epoch}] Err: {rerr:.4f} Loss: {loss.item():.4f} Accuracy: {correct_perc:.4f}%")

  def test_epoch(self, epoch):
    self.network.eval()
    num_items = len(self.test_loader.dataset)
    test_loss = 0
    correct = 0
    with torch.no_grad():
      iter = tqdm(self.test_loader, total=len(self.test_loader))
      for (data, target) in iter:
        output_t = self.network(tuple([data_i.to(device) for data_i in data]))
        output = self.program(*tuple(output_t))
        output = F.normalize(output, dim=-1)
        test_loss += self.loss(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred).to(device)).sum()
        perc = 100. * correct / num_items
        iter.set_description(f"[Test Epoch {epoch}] Total loss: {test_loss:.4f}, Accuracy: {correct}/{num_items} ({perc:.2f}%)")
    
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
  parser.add_argument("--n-epochs", type=int, default=100)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--learning-rate", type=float, default=1e-4)
  parser.add_argument("--method", type=str, default='tt')
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--jit", action="store_true")
  parser.add_argument("--digits", type=int, default=2)
  parser.add_argument("--dispatch", type=str, default="parallel")
  args = parser.parse_args()

  # Parameters
  n_epochs = args.n_epochs
  batch_size = args.batch_size
  learning_rate = args.learning_rate
  digit = args.digits
  method = args.method

  torch.manual_seed(args.seed)
  random.seed(args.seed)

  if torch.cuda.is_available(): device = torch.device('cuda')
  else: device = torch.device('cpu')

  # Data
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../data"))
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), f"../model/sum_{digit}"))
  os.makedirs(model_dir, exist_ok=True)

  # Dataloaders
  train_loader, test_loader = mnist_sum_loader(data_dir, batch_size, digit)

  # Create trainer and train
  trainer = Trainer(MNISTSumNet, method, digit, train_loader, test_loader, model_dir, learning_rate)
  trainer.train(n_epochs)