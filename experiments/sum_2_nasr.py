

import os
import random
from typing import *

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from argparse import ArgumentParser
from tqdm import tqdm
import json
import numpy as np

from src import input
from src import output

from torch_modules import nasr

mnist_img_transform = torchvision.transforms.Compose([
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(
    (0.1307,), (0.3081,)
  )
])

def sum_2(a, b):
  return a + b


def validation(a, b):
  a = a.argmax(dim=1)
  b = b.argmax(dim=1)

  predictions = torch.stack([torch.tensor(sum_2(a[i], b[i])) for i in range(len(a))])
  return predictions

def final_output(model,ground_truth, args, a, b):
  d_a = torch.distributions.categorical.Categorical(a)
  d_b = torch.distributions.categorical.Categorical(b)

  s_a = d_a.sample()
  s_b = d_b.sample()

  model.saved_log_probs = d_a.log_prob(s_a)+d_b.log_prob(s_b)

  predictions = []
  for i in range(len(s_a)):
    prediction = sum_2(s_a[i], s_b[i])
    predictions.append(prediction)
    reward = nasr.compute_reward(prediction,ground_truth[i])
    model.rewards.append(reward)
  
  return torch.stack(predictions)

class MNISTSum2Dataset(torch.utils.data.Dataset):
  def __init__(
    self,
    root: str,
    length: int,
    train: bool = True,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False,
  ):
    # Contains a MNIST dataset
    self.mnist_dataset = torchvision.datasets.MNIST(
      root,
      train=train,
      transform=transform,
      target_transform=target_transform,
      download=download,
    )
    self.length = length
    self.index_map = list(range(len(self.mnist_dataset)))
    random.shuffle(self.index_map)

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    # Get two data points
    (a_img, a_digit) = self.mnist_dataset[self.index_map[idx * 2]]
    (b_img, b_digit) = self.mnist_dataset[self.index_map[idx * 2 + 1]]

    # Each data has two images and the GT is the sum of two digits
    return (a_img, b_img, a_digit + b_digit)

  @staticmethod
  def collate_fn(batch):
    a_imgs = torch.stack([item[0] for item in batch])
    b_imgs = torch.stack([item[1] for item in batch])
    digits = torch.stack([torch.tensor(item[2]).long() for item in batch])
    return ((a_imgs, b_imgs), digits)


def mnist_sum_2_loader(data_dir, batch_size_train, batch_size_test):
    train_dataset = MNISTSum2Dataset(data_dir, length=5000, train=True, download=True, transform=mnist_img_transform)
    train_set_size = len(train_dataset)
    train_indices = list(range(train_set_size))
    split = int(train_set_size * 0.8)
    train_indices, val_indices = train_indices[:split], train_indices[split:]
    train_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_dataset, train_indices), collate_fn=MNISTSum2Dataset.collate_fn, batch_size=batch_size_train, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(train_dataset, val_indices), collate_fn=MNISTSum2Dataset.collate_fn, batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        MNISTSum2Dataset(
        data_dir,
        length=500,
        train=False,
        download=True,
        transform=mnist_img_transform,
        ),
        collate_fn=MNISTSum2Dataset.collate_fn,
        batch_size=batch_size_test,
        shuffle=True
    )

    return train_loader, valid_loader, test_loader


class MNISTNet(nn.Module):
  def __init__(self):
    super(MNISTNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
    self.fc1 = nn.Linear(1024, 1024)
    self.fc2 = nn.Linear(1024, 10)

  def forward(self, x):
    x = F.max_pool2d(self.conv1(x), 2)
    x = F.max_pool2d(self.conv2(x), 2)
    x = x.view(-1, 1024)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, p = 0.5, training=self.training)
    x = self.fc2(x)
    return F.softmax(x, dim=1)


class MNISTSum2Net(nn.Module):
  def __init__(self):
    super(MNISTSum2Net, self).__init__()

    # MNIST Digit Recognition Network
    self.mnist_net = MNISTNet()


  def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
    (a_imgs, b_imgs) = x

    a_distrs = self.mnist_net(a_imgs)
    b_distrs = self.mnist_net(b_imgs)

    return (a_distrs, b_distrs)

class RLSum2Net(nn.Module):
  def __init__(self, **kwargs):
    super().__init__()
    self.saved_log_probs = []
    self.rewards = []
    self.perception = MNISTSum2Net()

  def forward(self, x):
    return self.perception.forward(x)


def bce_loss(output, ground_truth):
  (_, dim) = output.shape
  gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in ground_truth])
  return F.binary_cross_entropy(output, gt)


class Trainer():
  def __init__(self, train_loader, test_loader, valid_loader, model, model_dir, final_output, args):
    self.model_dir = model_dir
    self.network = MNISTSum2Net()
    self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.valid_loader = valid_loader
    self.model = model
    self.best_loss = 10000000000
    self.loss = bce_loss
    self.final_output = final_output
    self.args = args


  def train_epoch(self, epoch):
    self.network.train()
    num_items = 0
    train_loss = 0
    
    iter = tqdm(self.train_loader, total=len(self.train_loader))

    eps = np.finfo(np.float32).eps.item()
    for i, (images, target) in enumerate(iter):
      images = tuple(image.to(self.args.gpu_id) for image in images)
      target = target.to(self.args.gpu_id)
      preds = self.network(images)
      self.final_output(self.model,target,self.args,*preds)
      rewards = np.array(self.model.rewards)
      rewards_mean = rewards.mean()
      rewards = (rewards - rewards.mean())/(rewards.std() + eps)
      policy_loss = torch.zeros(len(rewards), requires_grad=True)
      
      for n, (reward, log_prob) in enumerate(zip(rewards, self.model.saved_log_probs)):
        policy_loss[n].data += (-log_prob.cpu()*reward)
      self.optimizer.zero_grad()
      
      policy_loss = policy_loss.sum()

      num_items += images[0].size(0)
      train_loss += float(policy_loss.item() * images[0].size(0))
      policy_loss.backward()

      self.optimizer.step()
      
      avg_loss = train_loss/num_items
      iter.set_description(f"[Train Epoch {epoch}] AvgLoss: {avg_loss:.4f} AvgRewards: {rewards_mean:.4f}")
      
      if self.args.print_freq >= 0 and i % self.args.print_freq == 0:
        stats2 = {'epoch': epoch, 'train': i, 'avr_train_loss': avg_loss, 'avr_train_reward': rewards_mean}
        with open(f"model/nasr/detail_log.txt", "a") as f:
          f.write(json.dumps(stats2) + "\n")
      self.model.rewards = []
      self.model.shared_log_probs = []
      torch.cuda.empty_cache()
    
    return (train_loss/num_items), rewards_mean

  def test_epoch(self, epoch):
    self.network.eval()
    num_items = 0
    test_loss = 0
    rewards_value = 0
    num_correct = 0
    
    iter = tqdm(self.test_loader, total=len(self.test_loader))

    eps = np.finfo(np.float32).eps.item()
    with torch.no_grad():
      for i, (images, target) in enumerate(iter):
        images = tuple(image.to(self.args.gpu_id) for image in images)
        target = target.to(self.args.gpu_id)
        
        preds = self.network(images)
        output = self.final_output(self.model,target,self.args,*preds)

        rewards = np.array(self.model.rewards)
        rewards_mean = rewards.mean()
        rewards_value += float(rewards_mean * images[0].size(0))
        rewards = (rewards - rewards.mean())/(rewards.std() + eps)

        policy_loss = []
        for reward, log_prob in zip(rewards, self.model.saved_log_probs):
          policy_loss.append(-log_prob*reward)
        policy_loss = (torch.stack(policy_loss)).sum()

        num_items += images[0].size(0)
        test_loss += float(policy_loss.item() * images[0].size(0))
        self.model.rewards = []
        self.model.saved_log_probs = []
        torch.cuda.empty_cache()

        num_correct += (output==target).sum()
        perc = 100.*num_correct/num_items
        
        iter.set_description(f"[Test Epoch {epoch}] {int(num_correct)}/{int(num_items)} ({perc:.2f})%")
        
        if self.best_loss is None or test_loss < self.best_loss:
          self.best_loss = test_loss
          torch.save(self.network.state_dict(), f'{self.model_dir}/checkpoint_best_L.pth')
    
    avg_loss = (test_loss / num_items)
    
    return avg_loss, rewards_mean, perc


  def train(self, n_epochs):
    ckpt_path = os.path.join('outputs', 'nasr/')
    best_loss = None
    best_reward = None
    with open(f"{self.model_dir}/log.txt", 'w'): pass
    with open(f"{self.model_dir}/detail_log.txt", 'w'): pass
    for epoch in range(1, n_epochs+1):
        lr = nasr.adjust_learning_rate(self.optimizer, epoch, self.args)
        train_loss, train_rewards = self.train_epoch(epoch)
        loss, valid_rewards = nasr.validate(self.valid_loader, self.model, self.final_output, self.args)
        test_loss, _, test_accuracy = self.test_epoch(epoch)
      
        if best_reward is None or valid_rewards > best_reward :
            best_reward = valid_rewards
            torch.save(self.model.state_dict(), f'{ckpt_path}/checkpoint_best_R.pth')
        
        if best_loss is None or loss < best_loss :
            best_loss = loss
            torch.save(self.model.state_dict(), f'{ckpt_path}/checkpoint_best_L.pth')
      
        stats = {'epoch': epoch, 'lr': lr, 'train_loss': train_loss, 
                    'val_loss': loss, 'best_loss': best_loss , 
                    'train_rewards': train_rewards, 'valid_rewards': valid_rewards,
                    'test_loss': test_loss, 'test_accuracy': test_accuracy.item()}
        with open(f"{self.model_dir}/log.txt", "a") as f:
            f.write(json.dumps(stats) + "\n")
    torch.save(self.network.state_dict(), f'{self.model_dir}/checkpoint_last.pth')
    
    # Testing the best model
    self.model.load_state_dict(torch.load(f'{ckpt_path}/checkpoint_best_R.pth'))
    avg_loss, rewards_mean, perc = self.test_epoch(n_epochs+1)
    print(f"Best Model Test Loss: {avg_loss}")
    print(f"Best Model Test Reward: {rewards_mean}")
    print(f"Best Model Test Accuracy: {perc}")


if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("mnist_sum_2")
  parser.add_argument("--epochs", type=int, default=100)
  parser.add_argument('-b', '--batch-size', default=16, type=int)
  parser.add_argument("--learning-rate", type=float, default=0.0001)
  parser.add_argument('--weight-decay', default=3e-1, type=float)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument('--print-freq', default=5, type=int)
  parser.add_argument('--disable-cos', action='store_true')
  parser.add_argument('--warmup', default=0, type=int)
  parser.add_argument('--gpu-id', default='cpu', type=str)
  args = parser.parse_args()

  torch.manual_seed(args.seed)
  random.seed(args.seed)

  # Data
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../model/mnist_sum_2"))
  os.makedirs(model_dir, exist_ok=True)

  # Dataloaders
  (train_loader, valid_loader, test_loader) = mnist_sum_2_loader(data_dir, args.batch_size, args.batch_size)

  model = RLSum2Net()
  model.to(args.gpu_id)

  # Create trainer and train
  trainer = Trainer(train_loader, valid_loader, test_loader, model, model_dir, final_output, args)
  trainer.train(args.epochs)