import os
import random
from typing import *
from tqdm import tqdm
from functools import reduce
from argparse import ArgumentParser
from sklearn.metrics import confusion_matrix
from time import time
import wandb

import torch
import torch.nn.functional as F
import torch.optim as optim

from mnist_config import MNISTSumNet, mnist_multi_digit_sum2_loader
import sketch_config
from tensor_sketch import TensorSketch

def pretrain(digit, samples):
    t0 = time()
    # first calculate theta for the two-digit addition
    full_theta1 = sketch_config.full_theta2(digit=digit, elems=10, output_dim=19, samples=samples)
    # then calculate theta for the carry addition
    theta_i_0 = torch.arange(0, 19).unsqueeze(1).repeat(1, 10)
    theta_i_carry = torch.arange(1, 20).unsqueeze(1).repeat(1, 10)
    full_theta2 = torch.cat((theta_i_0, theta_i_carry), dim=1)
    t1 = time()
    wandb.log({"pretrain": t1 - t0})
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
    self.tensorsketch = TensorSketch(tensor_method)
    self.save_model = save_model
    self.rerr = [1] * (digits*2)
    self.t1 = None
    self.t2 = None
    self.full_theta1, self.full_theta2 = pretrain(digits, 0)

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
    output = torch.zeros(batch_size, 20).to(device).scatter_add_(1, t.flatten().repeat(batch_size, 1), p.flatten(1))
    return output
      
  def program(self, *inputs):
    ps = inputs
    batch_size = inputs[0].shape[0]
    rerr1, cores1, X_hat1 = self.tensorsketch.approx_theta({'gt': self.full_theta1, 'rank': 2})
    self.t1 = torch.clamp(X_hat1, min=0)
    rerr2, cores2, X_hat2 = self.tensorsketch.approx_theta({'gt': self.full_theta2, 'rank': 2})
    self.t2 = torch.clamp(X_hat2, min=0)
    assert(torch.all(self.t1 == self.full_theta1)) # This assertion sometimes fails - 0 value to -1
    assert(torch.all(self.t2 == self.full_theta2))
    mapping = torch.tensor([i % 10 for i in range(20)]).to(device)
    
    first_sums = []
    # For each digit, we compute the sum2
    for i in range(self.digits):
      index = self.digits - 1 - i
      input1, input2 = ps[index], ps[index + self.digits]
      sum_output = self.sub_program(2, 19, *(input1, input2))
      first_sums.append(sum_output)
    
    final_sums = []
    previous_sum = torch.cat([torch.ones(batch_size, 1).to(device), torch.zeros(batch_size, 19).to(device)], dim=1)
    # For each digit, we compute the carry sum
    for i in range(self.digits):
      sum_pred = first_sums[i]
      final_sum = self.sub_program_carry(sum_pred, previous_sum)
      grouped_final_sum = torch.zeros(batch_size, 10).to(device).index_add(1, mapping, final_sum)
      final_sums.append(grouped_final_sum)
      previous_sum = final_sum
    # append 1 if the last sum gets the carry
    tens_mapping = torch.tensor([i // 10 for i in range(20)]).to(device)
    final_sums.append(torch.zeros(batch_size, 10).to(device).index_add(1, tens_mapping, previous_sum))
      
    output = torch.stack(final_sums).permute(1, 0, 2) # most significant digit at end, least significant at beginning
    rerr = rerr1 + rerr2
    return rerr, output
  
  def loss(self, output, ground_truth):
    dim = output.shape[2]
    return F.nll_loss(torch.flatten((output + 1e-8).log(), start_dim=0, end_dim=1), torch.flatten(ground_truth))
  
  def train_epoch(self, epoch):
    self.network.train()
    num_items = 0
    total_correct = 0
    train_loss = 0
    
    pred_powers_of_10 = torch.arange(self.digits + 1, device=device).unsqueeze(0)
    target_powers_of_10 = torch.arange(self.digits + 1).unsqueeze(0)
    for (i, (data, target, _)) in enumerate(train_loader):
      self.optimizer.zero_grad()
      output_t = self.network(tuple([data_i.to(device) for data_i in data]))
      rerr, output = self.program(*tuple(output_t))
      loss = self.loss(output, target.to(device))
      loss.backward()
      self.optimizer.step()
      final_pred = torch.sum(output.argmax(dim=-1) * (10 ** pred_powers_of_10), dim=1)
      final_target = torch.sum(target * (10 ** target_powers_of_10), dim=1)
      total_correct += (final_pred.to('cpu') == final_target).sum()
      train_loss += loss.item()
      avg_loss = train_loss / (i + 1)
      num_items += output.shape[0]
      correct_perc = 100. * total_correct / num_items
    print(f"[Train Epoch {epoch}] Loss: {avg_loss:.4f}, Overall Accuracy: {int(total_correct)}/{int(num_items)} {correct_perc:.2f}%")
    return train_loss

  def test_epoch(self, epoch):
    self.network.eval()
    num_items = len(self.test_loader.dataset)
    correct = 0
    digit_correct = 0
    digits_pred = []
    digits_gt = []
    pred_powers_of_10 = torch.arange(self.digits - 1, -1 , -1, device=device).unsqueeze(0)
    target_powers_of_10 = torch.arange(0, self.digits + 1).unsqueeze(0)
    with torch.no_grad():
      for (data, target, digits) in test_loader:
        output_t = self.network(tuple([data_i.to(device) for data_i in data]))
        output = torch.stack(output_t, dim=0).argmax(dim=-1)
        # split output into two tensors
        output1 = torch.sum(output[:self.digits].T * (10 ** pred_powers_of_10), dim=1)
        output2 = torch.sum(output[self.digits:].T * (10 ** pred_powers_of_10), dim=1)
        final_pred = output1 + output2
        final_target = torch.sum(target * (10 ** target_powers_of_10), dim=1)
        digits_pred.extend(output.t().flatten().cpu())
        digits_gt.extend(digits.flatten().cpu())
        target = target.to(device)
        correct += (final_pred.to('cpu') == final_target).sum()
        
        digit_correct += (output.t().flatten() == digits.flatten().to(device)).sum()
        perc = 100. * correct / num_items
        perc2 = 100. * digit_correct / (num_items * self.digits * 2)

    print(f"[Test Epoch {epoch}] Total correct {int(correct)}/{int(num_items)} ({perc:.2f}) Digit correct {perc2:.2f}%")
    
    if self.save_model:
      torch.save(self.network.state_dict(), self.model_dir+f"/best.pth")
    
    return perc, perc2

  def train(self, n_epochs):
    for epoch in range(1, n_epochs + 1):
      t0 = time()
      train_loss = self.train_epoch(epoch)
      t1 = time()
      test_acc, test_digit_acc = self.test_epoch(epoch)
      t2 = time()
      
      wandb.log({
        "train_loss": train_loss,
        "test_acc": test_acc,
        "test_digit_acc": test_digit_acc,
        "train time": t1 - t0,
        "test time": t2 - t1,
        "epoch time": t2 - t0})
      
      if self.save_model: 
        torch.save(self.network.state_dict(), self.model_dir+f"/latest.pth")

if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("mnist_sum")
  parser.add_argument("--n-epochs", type=int, default=20)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--learning-rate", type=float, default=1e-3)
  parser.add_argument("--method", type=str, default='tt')
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--jit", action="store_true")
  parser.add_argument("--digits", type=int, default=15)
  parser.add_argument("--dispatch", type=str, default="parallel")
  args = parser.parse_args()

  # Parameters
  n_epochs = args.n_epochs
  batch_size = args.batch_size
  learning_rate = args.learning_rate
  digit = args.digits
  method = args.method
  
  for seed in [4321]:

    torch.manual_seed(seed)
    random.seed(seed)

    if torch.cuda.is_available(): device = torch.device('cuda')
    else: device = torch.device('cpu')

    # Data
    data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../data"))
    model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), f"../model/sum_{digit}"))
    os.makedirs(model_dir, exist_ok=True)

    # Dataloaders
    train_loader, test_loader = mnist_multi_digit_sum2_loader(data_dir, batch_size, digit)
    
    # Setup wandb
    config = vars(args)
    run = wandb.init(
      project=f'add-two-{digit}-digit',
      name = f'{seed}',
      config=config
    )
    
    # Create and run trainer
    trainer = Trainer(MNISTSumNet, method, digit, train_loader, test_loader, model_dir, learning_rate)
    trainer.train(n_epochs)
    
    run.finish()