import os
import random
from typing import *
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.optim as optim

from argparse import ArgumentParser
from mnist_config import MNISTNet, MNISTSumNet, mnist_sum_loader, mnist_loader
from tensor_sketch import TensorSketch

import sklearn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np

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

def full_theta_layer2(digit, components, samples):
  xs = []
  for i in range(int(digit/components)):
    xs.append(torch.randint(0, digit*9, (samples,)))
  xs = torch.stack(xs, dim=0)
  t = torch.randint(0, digit*components*9 + 1, tuple([digit*9+1]*components))
  for i in range(samples):
    x_i = tuple(xs[:, i].tolist())
    t[x_i] = sum(x_i)
  return t


class Trainer():
  def __init__(self, model, tensor_method, digit, components, train_loader, test_loader, mnist_loader, model_dir, learning_rate, save_model=False):
    self.model_dir = model_dir
    self.network = model(digit).to(device)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.mnist_loader = mnist_loader
    self.best_loss = 10000000000
    self.output_dim = digit*9 + 1
    self.digit = digit
    self.components = components
    self.t1 = full_theta(2, 0).detach() # hard code t1 to sum 2 digits (layer 1)
    self.t2 = full_theta_layer2(2, 2, 5000).detach() # hard code t2 to sum 4 digits (layer 2)
    self.t3 = full_theta_layer2(4, 2, 20000).detach() # hard code t3 to sum 8 digits (layer 3)
    self.gt1 = self.t1.clone() # cheating - shouldn't be saving this
    self.gt2 = self.t2.clone()
    self.gt3 = self.t3.clone()
    self.tensorsketch = TensorSketch(tensor_method)
    self.save_model = save_model
  
  def target_t1(self, digit, *inputs):
    ps = []
    for input_i in inputs[:2]:
      ps.append(torch.randint(0, 10, (input_i.shape[0]*digit*10,)).flatten())
    ps = torch.stack(ps, dim=-1)
    for sample_i in ps:
      s_i = tuple(sample_i.tolist())
      self.gt1[s_i] = sum(s_i)
      
  def target_t2(self, digit, *inputs):
    ps = []
    for input_i in inputs[:2]:
      ps.append(torch.randint(0, 9*digit+1, (input_i.shape[0]*digit*100,)).flatten())
    ps = torch.stack(ps, dim=-1)
    for sample_i in ps:
      s_i = tuple(sample_i.tolist())
      self.gt2[s_i] = sum(s_i)
      
  def target_t3(self, digit, *inputs):
    ps = []
    for input_i in inputs[:2]:
      ps.append(torch.randint(0, 18*digit+1, (input_i.shape[0]*digit*100,)).flatten())
    ps = torch.stack(ps, dim=-1)
    for sample_i in ps:
      s_i = tuple(sample_i.tolist())
      self.gt3[s_i] = sum(s_i)
    
  def program(self, *inputs):
    t1 = self.t1.to(device).clamp(0, 18)
    t2 = self.t2.to(device).clamp(0, 36)
    t3 = self.t3.to(device).clamp(0, 72)
    
    ps = inputs
    batch_size = inputs[0].shape[0]
    
    # 4 iterations
    p_outs = []
    for k in range(4):
        p_out = ps[k*2]
        i = 1
        p1 = p_out.unsqueeze(-1)
        p2 = ps[(k*2)+i].unsqueeze(1)
        eqn = f'{"".join([chr(j + 97) for j in range(0, i+2)])}, a{"".join([chr(i + 97) for i in range(i+1, i+3)])} -> {"".join([chr(j + 97) for j in range(0, i+1)])}{chr(i+97+2)}'
        p_out = torch.einsum(eqn, p1, p2)
        p_out = torch.zeros(batch_size, 19).to(device).scatter_add_(1, t1.flatten().repeat(batch_size, 1), p_out.flatten(1))
        p_outs.append(p_out)
        
    p_outs2 = []
    for k in range(2):
        p_out = p_outs[k*2]
        i = 1
        p1 = p_out.unsqueeze(-1)
        p2 = p_outs[(k*2)+i].unsqueeze(1)
        eqn = f'{"".join([chr(j + 97) for j in range(0, i+2)])}, a{"".join([chr(i + 97) for i in range(i+1, i+3)])} -> {"".join([chr(j + 97) for j in range(0, i+1)])}{chr(i+97+2)}'
        p_out = torch.einsum(eqn, p1, p2)
        p_out = torch.zeros(batch_size, 37).to(device).scatter_add_(1, t2.flatten().repeat(batch_size, 1), p_out.flatten(1))
        p_outs2.append(p_out)
        
    p1 = p_outs2[0]
    p2 = p_outs2[1]
    i = 1
    eqn = f'{"".join([chr(j + 97) for j in range(0, i+2)])}, a{"".join([chr(i + 97) for i in range(i+1, i+3)])} -> {"".join([chr(j + 97) for j in range(0, i+1)])}{chr(i+97+2)}'
    p_out = torch.einsum(eqn, p1.unsqueeze(-1), p2.unsqueeze(1))
    output = torch.zeros(batch_size, 73).to(device).scatter_add_(1, t3.flatten().repeat(batch_size, 1), p_out.flatten(1))
        
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
      self.target_t1(2, *tuple(output_t))
      self.target_t2(2, *tuple(output_t))
      self.target_t3(2, *tuple(output_t))
      rerr1, rerr2, rerr3, X_hat1, X_hat2, X_hat3 = self.tensorsketch.approx_theta({'gt1': self.gt1, 'gt2': self.gt2, 'gt3': self.gt3, 'digit': self.digit, 'components': 2})
      self.t1 = X_hat1
      self.t2 = X_hat2
      self.t3 = X_hat3
      output = self.program(*tuple(output_t))
      output = F.normalize(output, dim=-1)
      
      argmax_output_t = torch.stack([output_i.argmax(dim=1) for output_i in output_t])
      pred = torch.sum(argmax_output_t, dim=0)
      total_correct += pred.eq(target.data.view_as(pred).to(device)).sum()
      
      loss = self.loss(output, target.to(device))
      loss.backward()
      self.optimizer.step()
    #   total_correct += (output.argmax(dim=1)==target.to(device)).float().sum()
      num_items += output.shape[0]
      correct_perc = 100. * total_correct / num_items
      iter.set_description(f"[Train {epoch}] Err1: {rerr1:.4f} Err2: {rerr2:.4f} Err3: {rerr3:.4f} Loss: {loss.item():.4f} Accuracy: {correct_perc:.4f}%")

  def test_epoch(self, epoch):
    self.network.eval()
    num_items = len(self.test_loader.dataset)
    test_loss = 0
    correct = 0
    with torch.no_grad():
      iter = tqdm(self.test_loader, total=len(self.test_loader))
      for (data, target) in iter:
        output_t = self.network(tuple([data_i.to(device) for data_i in data]))
        argmax_output_t = torch.stack([output_i.argmax(dim=1) for output_i in output_t])
        output = torch.sum(argmax_output_t, dim=0)
        # test_loss += self.loss(output, target).item()
        test_loss = 0.0
        pred = output
        correct += pred.eq(target.data.view_as(pred).to(device)).sum()
        perc = 100. * correct / num_items
        iter.set_description(f"[Test Epoch {epoch}] Total loss: {test_loss:.4f}, Accuracy: {correct}/{num_items} ({perc:.2f}%)")
    
    if self.save_model and test_loss < self.best_loss:
      self.best_loss = test_loss
      torch.save(self.network.state_dict(), self.model_dir+f"/best.pth")
      
  def plot_confusion_matrix(self):
    # Get prediction result
    y_true, y_pred = [], []
    with torch.no_grad():
        for (imgs, digits) in self.mnist_loader:
            pred_digits = np.argmax(self.network.mnist_net(imgs), axis=1)
            y_true += [d.item() for d in digits]
            y_pred += [d.item() for d in pred_digits]

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(cm)

  def train(self, n_epochs):
    for epoch in range(1, n_epochs + 1):
      self.train_epoch(epoch)
      self.test_epoch(epoch)
      
      self.plot_confusion_matrix()
      
      if self.save_model: 
        torch.save(self.network.state_dict(), self.model_dir+f"/latest.pth")

if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("mnist_sum")
  parser.add_argument("--n-epochs", type=int, default=100)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--learning-rate", type=float, default=1e-4)
  parser.add_argument("--method", type=str, default='hooi')
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--jit", action="store_true")
  parser.add_argument("--digits", type=int, default=8)
  parser.add_argument("--components", type=int, default=2)
  parser.add_argument("--dispatch", type=str, default="parallel")
  args = parser.parse_args()

  # Parameters
  n_epochs = args.n_epochs
  batch_size = args.batch_size
  learning_rate = args.learning_rate
  digit = args.digits
  components = args.components
  method = args.method

  torch.manual_seed(args.seed)
  random.seed(args.seed)

  if torch.cuda.is_available(): device = torch.device('cuda')
  else: device = torch.device('cpu')

  # Data
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../data"))
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), f"../model/sum_16"))
  os.makedirs(model_dir, exist_ok=True)

  # Dataloaders
  train_loader, test_loader = mnist_sum_loader(data_dir, batch_size, digit)
  digit_loader = mnist_loader(data_dir, batch_size)

  # Create trainer and train
  trainer = Trainer(MNISTSumNet, method, digit, components, train_loader, test_loader, digit_loader, model_dir, learning_rate)
  trainer.train(n_epochs)