import os
import random
from typing import *
from tqdm import tqdm
from functools import reduce
from argparse import ArgumentParser
from sklearn.metrics import confusion_matrix

import torch
import torch.nn.functional as F
import torch.optim as optim

from mnist_config import MNISTSumNet, mnist_multi_digit_sum2_loader
import sketch_config
from tensor_sketch import TensorSketch

# add two numbers with number of digits=digits
def full_thetas(digit, samples):
    # first calculate theta for the two-digit addition
    full_theta1 = sketch_config.full_theta2(digit=digit, elems=10, output_dim=19, samples=samples)
    # then calculate theta for the carry addition
    theta_i_0 = torch.cat((torch.arange(0, 10), torch.arange(0, 9)))
    theta_i_carry = torch.cat((torch.arange(1, 10), torch.arange(0, 10)))
    full_theta2 = torch.stack([theta_i_0, theta_i_carry])
    return full_theta1, full_theta2


class Trainer():
  def __init__(self, model, tensor_method, digits, train_loader, test_loader, model_dir, learning_rate, save_model=False):
    self.model_dir = model_dir
    self.network = model(digits * 2).to(device)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.best_loss = 10000000000
    self.digits = digits
    # self.dims = [10] + [dim_i * 9 + 1 for dim_i in dims]
    self.tensorsketch = TensorSketch(tensor_method)
    self.save_model = save_model
    self.rerr = [1] * (digits*2)
    self.t1 = None
    self.t2 = None
    self.full_theta1, self.full_theta2 = full_thetas(digits, 0)

  def sub_program(self, n, d, *base_inputs):
    t = self.t1.long().to(device)
    p = base_inputs[0]
    batch_size = p.shape[0]
    for i in range(1, n):
      p1 = p.unsqueeze(-1)
      p2 = base_inputs[i].unsqueeze(1)
      eqn = f'{"".join([chr(j + 97) for j in range(0, i+2)])}, a{"".join([chr(i + 97) for i in range(i+1, i+3)])} -> {"".join([chr(j + 97) for j in range(0, i+1)])}{chr(i+97+2)}'
      p = torch.einsum(eqn, p1, p2)
    output = torch.zeros(batch_size, d).to(device).scatter_add_(1, t.flatten().repeat(batch_size, 1), p.flatten(1))
    return output
  
  def sub_program_carry(self, sum_result, carry):
    t = self.t2.long().to(device)
    batch_size = sum_result.shape[0]
    p1 = sum_result.unsqueeze(-1)
    p2 = carry.unsqueeze(1)
    i = 1
    eqn = f'{"".join([chr(j + 97) for j in range(0, i+2)])}, a{"".join([chr(i + 97) for i in range(i+1, i+3)])} -> {"".join([chr(j + 97) for j in range(0, i+1)])}{chr(i+97+2)}'
    p = torch.einsum(eqn, p1, p2)
    output = torch.zeros(batch_size, 10).to(device).scatter_add_(1, t.flatten().repeat(batch_size, 1), p.flatten(1))
    return output
      
  def program(self, *inputs):
    ps = inputs
    # digits, iters = 1, self.all_digit
    rerr1, cores1, X_hat1 = self.tensorsketch.approx_theta({'gt': self.full_theta1, 'rank': 2})
    self.t1 = X_hat1
    rerr2, cores2, X_hat2 = self.tensorsketch.approx_theta({'gt': self.full_theta2, 'rank': 2})
    self.t2 = X_hat2
    assert(torch.all(self.t1 == self.full_theta1))
    assert(torch.all(self.t2 == self.full_theta2))
    
    first_sums = []
    carry = []
    # For each digit, we compute the sum2
    for i in range(self.digits):
      input1, input2 = ps[i], ps[i + self.digits]
      sum_output = self.sub_program(2, 19, *(input1, input2))
      first_sums.append(sum_output)
      carry_output = torch.stack([sum_output[:, :10].sum(dim=1), sum_output[:, 10:].sum(dim=1)], dim=1)
      carry.append(carry_output)
    
    final_sums = []
    previous_carry = torch.cat([torch.ones(16, 1).to(device), torch.zeros(16, 1).to(device)], dim=1)
    # For each digit, we compute the carry sum
    for i in range(self.digits):
      sum_pred = first_sums[i]
      final_sum = self.sub_program_carry(sum_pred, previous_carry)
      final_sums.append(final_sum)
      previous_carry = carry[i]
    # append 1 if the last sum gets the carry
    final_sums.append(F.pad(previous_carry, (0, 8)))
      
    output = torch.stack(final_sums).permute(1, 0, 2)
    rerr = rerr1 + rerr2
    return rerr, output
  
  def loss(self, output, ground_truth):
    # output_len = output.shape[1]
    dim = output.shape[2]
    gt = torch.stack([torch.stack([torch.tensor([1.0 if i == e else 0.0 for i in range(dim)]) for e in t]) for t in ground_truth]).to(device)
    return F.cross_entropy(output, gt)
  
  def train_epoch(self, epoch):
    self.network.train()
    num_items = 0
    total_correct = 0
    iter = tqdm(self.train_loader, total=len(self.train_loader))
    for (data, target, _) in iter:
      self.optimizer.zero_grad()
      output_t = self.network(tuple([data_i.to(device) for data_i in data]))
      rerr, output = self.program(*tuple(output_t))
      output = F.normalize(output, dim=-1)
      loss = self.loss(output, target.to(device))
      loss.backward()
      self.optimizer.step()
      total_correct += (output.argmax(dim=-1)==target.to(device)).float().sum()
      num_items += output.shape[0]
      correct_perc = 100. * total_correct / num_items
      iter.set_description(f"[Train {epoch}] Err:{rerr:.4f} Loss: {loss.item():.4f} Accuracy: {correct_perc:.4f}%")

  def test_epoch(self, epoch):
    self.network.eval()
    num_items = len(self.test_loader.dataset)
    correct = 0
    digit_correct = 0
    digits_pred = []
    digits_gt = []
    with torch.no_grad():
      iter = tqdm(self.test_loader, total=len(self.test_loader))
      for (data, target, digits) in iter:
        output_t = self.network(tuple([data_i.to(device) for data_i in data]))
        output = torch.stack(output_t, dim=0).argmax(dim=-1)
        pred = output.sum(dim=0).to(device)
        digits_pred.extend(output.t().flatten().cpu())
        digits_gt.extend(digits.flatten().cpu())
        target = target.to(device)
        correct += (pred == target).sum()
        digit_correct += (output.t().flatten() == digits.flatten().to(device)).sum()
        perc = 100. * correct / num_items
        perc2 = 100. * digit_correct / (num_items * self.digits * 2)
        iter.set_description(f"[Test Epoch {epoch}] Accuracy: {correct}/{num_items} ({perc:.2f}%) Digit Accuracy ({perc2:.2f}%)")

      # cf_matrix = confusion_matrix(digits_gt, digits_pred)
      # print(cf_matrix)
    
    if self.save_model:
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
  parser.add_argument("--learning-rate", type=float, default=1e-3)
  parser.add_argument("--method", type=str, default='tt')
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
  digit = 15

  torch.manual_seed(args.seed)
  random.seed(args.seed)

  if torch.cuda.is_available(): device = torch.device('cuda')
  else: device = torch.device('cpu')

  # Data
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../data"))
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), f"../model/sum_{digit}"))
  os.makedirs(model_dir, exist_ok=True)

  # Dataloaders
  train_loader, test_loader = mnist_multi_digit_sum2_loader(data_dir, batch_size, digit)

  # Create trainer and train
  trainer = Trainer(MNISTSumNet, method, digit, train_loader, test_loader, model_dir, learning_rate)
  trainer.train(n_epochs)