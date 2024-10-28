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

def full_theta_layer2(digit, components, samples):
  # digit is number of digits input to first layer
  #   each component has maximum output of digit*9 + 1
  # second layer has #inputs = components
  #   second layer has maximum output of digit * components)*9 + 1
  xs = []
  for i in range(digit):
    xs.append(torch.randint(0, digit*9+1, (samples,)))
  xs = torch.stack(xs, dim=0)
  t = torch.randint(0, digit*components*9 + 1, tuple([digit*9+1]*components))
  for i in range(samples):
    x_i = tuple(xs[:, i].tolist())
    t[x_i] = sum(x_i)
  return t
    

class Trainer():
  def __init__(self, model, tensor_method, digit, components, train_loader, test_loader, model_dir, learning_rate, save_model=False):
    self.model_dir = model_dir
    self.network = model(digit).to(device)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.best_loss = 10000000000
    self.output_dim = digit*9 + 1
    self.digit = digit
    self.components = components
    self.t1 = full_theta(2, 0).detach() # hard code t1 to sum 2 digits (layer 1)
    self.t2 = full_theta_layer2(2, 2, 0).detach() # hard code t2 to sum 4 digits (layer 2)
    self.gt1 = self.t1.clone() # cheating - shouldn't be saving this
    self.gt2 = self.t2.clone()
    self.tensorsketch = TensorSketch(tensor_method)
    self.save_model = save_model
  
  def target_t1(self, *inputs):
    ps = []
    for input_i in inputs[:int(self.digit/self.components)]:
      ps.append(torch.randint(0, 10, (input_i.shape[0]*self.digit*10,)).flatten())
    ps = torch.stack(ps, dim=-1)
    for sample_i in ps:
      s_i = tuple(sample_i.tolist())
      self.gt1[s_i] = sum(s_i)

  def target_t2(self, *inputs):
    ps = []
    batch_size = inputs[0].shape[0]
    for i in range(self.components):
      # self.digit * 10 would mean 40 -> 10x10x10x10
      # we want 19x19 which means we should have components * (9*digit +)
      ps.append(torch.randint(0, (9*(int(self.digit/self.components)))+1, (batch_size*(components * (9*(int(digit/components)) +1)),)).flatten())
    ps = torch.stack(ps, dim=-1)
    for sample_i in ps:
      s_i = tuple(sample_i.tolist())
      self.gt2[s_i] = sum(s_i)
  
  def program(self, *inputs):
    t1 = self.t1.to(device)
    t2 = self.t2.to(device)
    t1_len = len(t1.shape)
    t2_len = len(t2.shape)
    input_len = len(inputs)
    composed_len = int(input_len / t1_len)   
    prev_output_t1 = t1
    prev_output_t2 = t2
    
    # outputs = []
    # for i in range(composed_len):
    #     p = inputs[i]
    #     batch_size = p.shape[0]
    #     for j in range(1, t1_len):
    #         # first layer
    #         p1 = p.unsqueeze(-1)
    #         p2 = inputs[i * t1_len + j].unsqueeze(1)
    #         eqn = f'{"".join([chr(l + 97) for l in range(0, j+2)])}, a{"".join([chr(k + 97) for k in range(j+1, j+3)])} -> {"".join([chr(l + 97) for l in range(0, j+1)])}{chr(j+97+2)}'
    #         p = torch.einsum(eqn, p1, p2) # p1 shape is 16x10x1 and p2 shape is 16x1x10
    #     # second layer
    #     # shape is 16x19
    #     prev_output_t1 = torch.zeros(batch_size, 19).to(device).scatter_add_(1, prev_output_t1.flatten().repeat(batch_size, 1), p.flatten(1))
    #     # prev_output_t1.unsqueeze(1) shape is 16x1x19
    #     # Here's the problem: t2 shape is 19x19, but we want it to be of the form 1x19 or something like this
    #     prev_output_t2 = torch.zeros(batch_size, 19)
    #     # prev_output_t2 = torch.einsum(eqn, prev_output_t1.unsqueeze(1), prev_output_t2.unsqueeze(-1))
    #     pass
    #     # outputs.append(output)
    # pass
    
    p1 = inputs[0]
    p2 = inputs[1]
    p3 = inputs[2]
    p4 = inputs[3]
    
    batch_size = p1.shape[0]
    
    p1 = p1.unsqueeze(-1)
    p2 = p2.unsqueeze(1)
    p3 = p3.unsqueeze(-1)
    p4 = p4.unsqueeze(1)
    
    eqn = f'abc, acd -> abd'
    
    p_out1 = torch.einsum(eqn, p1, p2)
    p_out2 = torch.einsum(eqn, p3, p4)
    
    output1 = torch.zeros(batch_size, 19).to(device).scatter_add_(1, t1.flatten().repeat(batch_size, 1), p_out1.flatten(1))
    output2 = torch.zeros(batch_size, 19).to(device).scatter_add_(1, t1.flatten().repeat(batch_size, 1), p_out2.flatten(1))
    # both outputs are 16x19
    
    p1 = output1.unsqueeze(-1)
    p2 = output2.unsqueeze(1)
    p_out = torch.einsum(eqn, p1, p2)
    output = torch.zeros(batch_size, 37).to(device).scatter_add_(1, t2.flatten().repeat(batch_size, 1), p_out.flatten(1))
    
    return output
        
        
    # p = inputs[0]
    # batch_size = p.shape[0]
    # for i in range(1, self.digit):
    #   p1 = p.unsqueeze(-1)
    #   p2 = inputs[i].unsqueeze(1)
    #   eqn = f'{"".join([chr(j + 97) for j in range(0, i+2)])}, a{"".join([chr(i + 97) for i in range(i+1, i+3)])} -> {"".join([chr(j + 97) for j in range(0, i+1)])}{chr(i+97+2)}'
    #   p = torch.einsum(eqn, p1, p2)
    # output = torch.zeros(batch_size, self.output_dim).to(device).scatter_add_(1, t.flatten().repeat(batch_size, 1), p.flatten(1))
    # return output
  
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
      self.target_t1(*tuple(output_t))
      self.target_t2(*tuple(output_t))
      rerr1, rerr2, X_hat1, X_hat2 = self.tensorsketch.approx_theta({'gt1': self.gt1, 'gt2': self.gt2, 'digit': self.digit, 'components': 2})
      self.t1 = X_hat1
      self.t2 = X_hat2
      output = self.program(*tuple(output_t))
      output = F.normalize(output, dim=-1)
      loss = self.loss(output, target.to(device))
      loss.backward()
      self.optimizer.step()
      total_correct += (output.argmax(dim=1)==target.to(device)).float().sum()
      num_items += output.shape[0]
      correct_perc = 100. * total_correct / num_items
      iter.set_description(f"[Train {epoch}] Err: {rerr1:.4f} Loss: {loss.item():.4f} Accuracy: {correct_perc:.4f}%")

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
  parser.add_argument("--method", type=str, default='hooi')
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--jit", action="store_true")
  parser.add_argument("--digits", type=int, default=4)
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

  # Create trainer and train
  trainer = Trainer(MNISTSumNet, method, digit, components, train_loader, test_loader, model_dir, learning_rate)
  trainer.train(n_epochs)