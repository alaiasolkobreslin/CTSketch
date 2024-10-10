import os
import random
from typing import *
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from argparse import ArgumentParser

mnist_img_transform = torchvision.transforms.Compose([
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(
    (0.1307,), (0.3081,)
  )
])

class MNISTSumDataset(torch.utils.data.Dataset):
  def __init__(
    self,
    root: str,
    digit: int,
    train: bool = True,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False,
  ):
    self.digit = digit
    # Contains a MNIST dataset
    if train: self.length = 60000 # min(5000 * digit, 60000)
    else: self.length = 10000 # min(500 * digit, 10000)
    self.digit = digit
    self.mnist_dataset = torch.utils.data.Subset(
      torchvision.datasets.MNIST(
        root,
        train=train,
        transform=transform,
        target_transform=target_transform,
        download=download,
      ),
      range(self.length)
    )
    self.index_map = list(range(len(self.mnist_dataset)))
    random.shuffle(self.index_map)

    self.sum_dataset = []
    for i in range(len(self.mnist_dataset)//self.digit):
      self.sum_dataset.append([])
      for j in range(self.digit):
        self.sum_dataset[i].append(self.mnist_dataset[self.index_map[i*self.digit + j]])

  def __len__(self):
    return len(self.sum_dataset)

  def __getitem__(self, idx):
    # Get two data points
    item = self.sum_dataset[idx]
    data, target = [], []
    for (d,t) in item:
      data.append(d)
      target.append(t)
    
    target = sum(tuple(target))

    # Each data has two images and the GT is the sum of two digits
    return (*tuple(data), target)

  @staticmethod
  def collate_fn(batch):
    imgs = []
    for i in range(len(batch[0])-1):
      imgs.append(torch.stack([item[i] for item in batch]))
    digits = torch.stack([torch.tensor(item[-1]).long() for item in batch])
    return (tuple(imgs), digits)


def mnist_sum_loader(data_dir, batch_size, digit):
  train_loader = torch.utils.data.DataLoader(
    MNISTSumDataset(
      data_dir,
      digit,
      train=True,
      download=True,
      transform=mnist_img_transform,
    ),
    collate_fn=MNISTSumDataset.collate_fn,
    batch_size=batch_size,
    shuffle=True
  )

  test_loader = torch.utils.data.DataLoader(
    MNISTSumDataset(
      data_dir,
      digit,
      train=False,
      download=True,
      transform=mnist_img_transform,
    ),
    collate_fn=MNISTSumDataset.collate_fn,
    batch_size=batch_size,
    shuffle=True
  )

  return train_loader, test_loader

class MNISTNet(nn.Module):
  def __init__(self):
    super(MNISTNet, self).__init__()
    self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
    self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
    self.fc1 = nn.Linear(256, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
    x = F.max_pool2d(F.relu(self.conv1(x)), 2)
    x = F.max_pool2d(F.relu(self.conv2(x)), 2)
    x = x.view(-1, 256)
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return F.softmax(x, dim=1)

def full_theta():
  x = torch.randint(0, 10, (50,))
  y = torch.randint(0, 10, (50,))
  t = torch.zeros(10, 10, 19)
  for (xi, yi) in zip(x, y):
    s = xi + yi
    t[xi, yi, s] = 1
  return t   

class MNISTSumNet(nn.Module):
  def __init__(self, digit):
    super(MNISTSumNet, self).__init__()

    self.mnist_net = MNISTNet()
    self.digit = digit

  def forward(self, x):
    batch_size = x[0].shape[0]
    x = torch.cat(x, dim=0)
    x = self.mnist_net(x)
    x = [x[i*batch_size:(i + 1) * batch_size,:] for i in range(self.digit)]
    return tuple(x)

class Trainer():
  def __init__(self, model, digit, train_loader, test_loader, model_dir, learning_rate):
    self.model_dir = model_dir
    self.network = model(digit)
    self.optimizer = optim.AdamW(self.network.parameters(), lr=learning_rate, weight_decay=1e-5)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.best_loss = 10000000000
    self.tensor_loss_fn = nn.MSELoss()
    self.t1 = nn.Parameter(torch.randn(19, 10, 1), requires_grad=True)
    self.t2 = nn.Parameter(torch.randn(19, 1, 10), requires_grad=True)
    self.tensor_optimizer = optim.AdamW([self.t1, self.t2], lr = 0.05, weight_decay = 0.0001)
    self.t3 = full_theta().permute(2, 0, 1).detach()

  def approx_theta(self):
      iter = tqdm()
      for n in range(10000):
        self.tensor_optimizer.zero_grad()
        t = self.t1.bmm(self.t2)
        t = F.gumbel_softmax(t, dim=0, hard=True)
        l = self.tensor_loss_fn(t, self.t3)
        l.backward()
        self.tensor_optimizer.step()
        iter.set_description(f"[Train Epoch {n}] Loss: {l.item():.4f}%")
  
  def program(self, p1, p2):
    t = self.t1.bmm(self.t2)
    t = F.one_hot(torch.argmax(t, dim=0), num_classes = t.shape[0]).float()
    p1 = p1.unsqueeze(-1)
    p2 = p2.unsqueeze(1)
    p = p1.bmm(p2)
    output = torch.einsum('ijk, jkl -> il', p, t)
    return output
  
  def target_t(self, pa, pb):
    a = pa.argmax(dim=-1) # torch.multinomial(pa, 1).flatten()
    b = pb.argmax(dim=-1)
    for (a_i, b_i) in zip(a, b):
      self.t3[:, a_i, b_i] = F.one_hot(a_i + b_i, num_classes = self.t3.shape[0])
  
  def tensor_loss(self):
    target = self.t3
    for n in range(1):
      self.tensor_optimizer.zero_grad()
      p = self.t1.bmm(self.t2)
      p = F.gumbel_softmax(p, dim=0, hard=True)
      l = self.tensor_loss_fn(p, target)
      l.backward()
      self.tensor_optimizer.step()
    return l

  def loss(self, output, ground_truth):
    dim = output.shape[1]
    gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in ground_truth])
    return F.cross_entropy(output, gt)
  
  def train2(self, epoch):
    self.network.train()
    num_items = 0
    total_correct = 0
    iter = tqdm(self.train_loader, total=len(self.train_loader))
    for (data, target) in iter:
      self.optimizer.zero_grad()
      self.tensor_optimizer.zero_grad()
      output_t = self.network(data)
      output = self.program(*tuple(output_t))
      output = F.normalize(output, dim=-1)
      loss = self.loss(output, target)
      loss.backward()
      self.optimizer.step()
      self.tensor_optimizer.step()
      total_correct += (output.argmax(dim=1)==target).float().sum()
      num_items += output.shape[0]
      correct_perc = 100. * total_correct / num_items
      iter.set_description(f"[Train Epoch {epoch}] Loss: {loss.item():.4f} Accuracy: {correct_perc:.4f}%")

  def train_epoch(self, epoch):
    self.network.train()
    num_items = 0
    total_correct = 0
    iter = tqdm(self.train_loader, total=len(self.train_loader))
    for (data, target) in iter:
      self.optimizer.zero_grad()
      output_t = self.network(data)
      if self.t3.sum() < 100:
        self.target_t(*tuple(output_t))
      loss_t = self.tensor_loss()
      output = self.program(*tuple(output_t))
      output = F.normalize(output, dim=-1)
      loss = self.loss(output, target)
      loss.backward()
      self.optimizer.step()
      total_correct += (output.argmax(dim=1)==target).float().sum()
      num_items += output.shape[0]
      correct_perc = 100. * total_correct / num_items
      iter.set_description(f"[Train Epoch {epoch}] LossT: {loss_t.item():.4f} Loss: {loss.item():.4f} Accuracy: {correct_perc:.4f}%")

  def test_epoch(self, epoch):
    self.network.eval()
    num_items = len(self.test_loader.dataset)
    test_loss = 0
    correct = 0
    with torch.no_grad():
      iter = tqdm(self.test_loader, total=len(self.test_loader))
      for (data, target) in iter:
        output_t = self.network(data)
        output = self.program(*tuple(output_t))
        output = F.normalize(output, dim=-1)
        test_loss += self.loss(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        perc = 100. * correct / num_items
        iter.set_description(f"[Test Epoch {epoch}] Total loss: {test_loss:.4f}, Accuracy: {correct}/{num_items} ({perc:.2f}%)")

  def train(self, n_epochs):
    self.approx_theta()
    for epoch in range(1, n_epochs + 1):
      self.train_epoch(epoch)
      self.test_epoch(epoch)
    torch.save(self.network.state_dict(), self.model_dir+"/model.pth")

if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("mnist_sum")
  parser.add_argument("--n-epochs", type=int, default=100)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--learning-rate", type=float, default=5e-4)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--jit", action="store_true")
  parser.add_argument("--dispatch", type=str, default="parallel")
  args = parser.parse_args()

  # Parameters
  n_epochs = args.n_epochs
  batch_size = args.batch_size
  learning_rate = args.learning_rate
  digit = 2
  torch.manual_seed(args.seed)
  random.seed(args.seed)

  # Data
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../data"))
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../model/mnist_sum"))
  os.makedirs(model_dir, exist_ok=True)

  # Dataloaders
  train_loader, test_loader = mnist_sum_loader(data_dir, batch_size, digit)

  # Create trainer and train
  trainer = Trainer(MNISTSumNet, digit, train_loader, test_loader, model_dir, learning_rate)
  trainer.train(n_epochs)