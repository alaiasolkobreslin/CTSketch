import os
import random
from typing import *
from tqdm import tqdm
from time import time

import torch
import torch.nn.functional as F
import torch.optim as optim

from argparse import ArgumentParser
import wandb
from sklearn.metrics import confusion_matrix

from mnist_config import MNISTSumNet, mnist_sum_loader
from sketch_configs import full_theta2, full_theta3, full_theta4, sample_theta

def full_theta(digits, input_dim, output_dim, samples):
  if digits == 4:
    return full_theta4(digits, input_dim, output_dim, samples)
  elif digits == 3:
    return full_theta3(digits, input_dim, output_dim, samples)
  elif digits == 2:
    return full_theta2(digits, input_dim, output_dim, samples)
  else: 
    raise Exception("not implemented")
    #return sample_theta(digits, input_dim, output_dim, samples)

class Trainer():
  def __init__(self, model, tensor_method, digits, dims, train_loader, test_loader, val_loader, model_dir, learning_rate, save_model=True):
    self.model_dir = model_dir
    self.all_digit = dims[-1]
    self.network = model(self.all_digit).to(device)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.val_loader = val_loader
    self.digits = digits
    self.dims = [10] + [dim_i * 9 + 1 for dim_i in dims]
    self.tensorsketch = TensorSketch(tensor_method)
    self.rerr = [1] * len(digits)
    self.t = [None] * len(digits)
    self.digit_acc = 0
    self.best_acc = 0
    self.save_model = save_model

    t0 = time()
    self.gt = []
    for i, digit_i in enumerate(self.digits):
      gt = full_theta(digit_i, self.dims[i], self.dims[i + 1], 0)
      self.gt.append(gt)
    t1 = time()
    wandb.log({"pretrain": t1 - t0})

  def target_t(self, i, n, r, samples, *inputs):
    ps = []
    batch_size = inputs[0].shape[0]
    for _ in range(n):
      ps.append(torch.randint(0, r, (batch_size*samples,)))
    ps = torch.stack(ps, dim=-1)
    for sample_i in ps:
      s_i = tuple(sample_i.tolist())
      self.gt[i][s_i] = sum(s_i)

  def sub_program(self, i, n, d, *base_inputs):
    t = self.t[i].long().to(device)
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
    if len(self.digits) >= 8: ranks = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    else: ranks = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
    for i, n_i in enumerate(self.digits):
      if self.t[i] is None:
        # self.target_t(i, n_i, d_i, 100, *ps) - assuming full pre-training for now
        digits *= n_i
        rerr, _, X_hat = self.tensorsketch.approx_theta({'gt': self.gt[i], 'digit': digits, 'output': self.dims[i+1], 'rank': ranks[i]})
        self.t[i] = X_hat.to_numpy().round().clamp(0, self.dims[i+1]).long()
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
    train_loss = 0
    iter = tqdm(self.train_loader, total=len(self.train_loader))
    for (data, target, _) in iter:
      self.optimizer.zero_grad()
      output_t = self.network(tuple([data_i.to(device) for data_i in data]))
      rerr, output = self.program(*tuple(output_t))
      output = F.normalize(output, dim=-1)
      loss = self.loss(output, target.to(device))
      train_loss += loss
      loss.backward()
      self.optimizer.step()
      total_correct += (output.argmax(dim=-1)==target.to(device)).float().sum()
      num_items += output.shape[0]
      correct_perc = 100. * total_correct / num_items
      iter.set_description(f"[Train {epoch}] Err:{rerr:.4f} Loss: {loss.item():.4f} Accuracy: {correct_perc:.4f}%")
    return train_loss, rerr

  def test_epoch(self, epoch):
    self.network.eval()
    num_items = len(self.val_loader.dataset)
    correct = 0
    digit_correct = 0
    digits_pred = []
    digits_gt = []
    with torch.no_grad():
      iter = tqdm(self.val_loader, total=len(self.val_loader))
      for (data, target, digits) in iter:
        output_t = self.network(tuple([data_i.to(device) for data_i in data]))
        output = torch.stack(output_t, dim=0).argmax(dim=-1)
        pred = output.sum(dim=0).to(device)
        digits_pred.extend(output.t().flatten().cpu())
        digits_gt.extend(digits.flatten().cpu())
        target = target.to(device)
        correct += (pred == target).sum()
        digit_correct += (output.t().flatten() == digits.flatten().to(device)).sum()
        perc = float(correct/num_items)
        digit_perc = float(digit_correct / (num_items * self.all_digit))
        iter.set_description(f"[Test Epoch {epoch}] Accuracy: {correct}/{num_items} ({perc * 100:.2f}%) Digit Accuracy ({digit_perc * 100:.2f}%)")

      # cf_matrix = confusion_matrix(digits_gt, digits_pred)
      # print(cf_matrix)
    
    return perc, digit_perc

  def test(self):
    # self.network.load_state_dict(torch.load(self.model_dir+"/best.pth"))
    self.network.eval()
    num_items = len(self.test_loader.dataset)
    correct = 0
    digit_correct = 0
    with torch.no_grad():
      for (data, target, digits) in self.test_loader:
        target = target.to(device)
        output_t = self.network(tuple([data_i.to(device) for data_i in data]))
        output = torch.stack(output_t, dim=0).argmax(dim=-1)
        pred = output.sum(dim=0).to(device)
        correct += (pred == target).sum()
        digit_correct += (output.t().flatten() == digits.flatten().to(device)).sum()
      perc = float(correct/num_items)
      digit_perc = float(digit_correct / (num_items * self.all_digit))
      print(f"[Final Acc] {int(correct)}/{int(num_items)} ({perc * 100:.2f})% Digit Accuracy ({digit_perc * 100:.2f}%)")
    wandb.log({"final acc": perc, "final digit acc": digit_perc})

  def train(self, n_epochs):
    for epoch in range(1, n_epochs+1):
      t0 = time()
      train_loss, rerr = self.train_epoch(epoch)
      t1 = time()
      test_acc, test_digit_acc = self.test_epoch(epoch)
      t2 = time()
 
      wandb.log({
        "train_loss": train_loss,
        "rerr": rerr,
        "test_acc": test_acc,
        "test_digit_acc": test_digit_acc,
        "train time": t1 - t0,
        "test time": t2 - t1,
        "epoch time": t2 - t0})     
          
      if self.save_model and self.best_acc < test_acc:
        self.best_acc = test_acc
        self.test()

if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("mnist_sum")
  parser.add_argument("--n-epochs", type=int, default=100)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--learning-rate", type=float, default=5e-4)
  parser.add_argument("--method", type=str, default='tt')
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--depth", type=int, default=1)
  args = parser.parse_args()

  # Parameters
  n_epochs = args.n_epochs
  batch_size = args.batch_size
  learning_rate = args.learning_rate
  depth = args.depth
  method = args.method

  digits = [2] * depth
  dims = [2 ** (1 + i) for i in range(depth)]

  torch.manual_seed(args.seed)
  random.seed(args.seed)

  if torch.cuda.is_available(): device = torch.device('cuda')
  else: device = torch.device('cpu')

  # Data
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), f"../../model/tensor_tree{depth}_{args.seed}"))
  os.makedirs(model_dir, exist_ok=True)

  # Dataloaders
  train_loader, val_loader, test_loader = mnist_sum_loader(data_dir, batch_size, dims[-1])

  # setup wandb
  config = vars(args)
  wandb.init(
    project=f"tensor_tree{depth}",
    name=f"{args.seed}",
    config=config)
  print(config)

  # Create trainer and train
  trainer = Trainer(MNISTSumNet, method, digits, dims, train_loader, test_loader, val_loader, model_dir, learning_rate)
  trainer.train(n_epochs)