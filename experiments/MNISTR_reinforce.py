

import os
import random
from typing import *
import inspect

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from argparse import ArgumentParser
from tqdm import tqdm

from src import input
from src import output

from torch_modules import reinforce

mnist_img_transform = torchvision.transforms.Compose([
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(
    (0.1307,), (0.3081,)
  )
])

class MNISTRDataset(torch.utils.data.Dataset):
  def __init__(
    self,
    root: str,
    digit: int,
    bbox: Callable,
    train: bool = True,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False,
  ):
    self.digit = digit
    self.bbox = bbox
    # Contains a MNIST dataset
    if train: self.length = min(5000 * digit, 60000)
    else: self.length = min(500 * digit, 10000)
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
    
    target = self.bbox(*tuple(target))

    # Each data has two images and the GT is the sum of two digits
    return (*tuple(data), target)

  @staticmethod
  def collate_fn(batch):
    imgs = []
    for i in range(len(batch[0])-1):
      imgs.append(torch.stack([item[i] for item in batch]))
    digits = torch.stack([torch.tensor(item[-1]).long() for item in batch])
    return (tuple(imgs), digits)


def mnist_r_loader(data_dir, batch_size, digit, bbox):
  train_loader = torch.utils.data.DataLoader(
    MNISTRDataset(
      data_dir,
      digit,
      bbox,
      train=True,
      download=True,
      transform=mnist_img_transform,
    ),
    collate_fn=MNISTRDataset.collate_fn,
    batch_size=batch_size,
    shuffle=True
  )

  test_loader = torch.utils.data.DataLoader(
    MNISTRDataset(
      data_dir,
      digit,
      bbox,
      train=False,
      download=True,
      transform=mnist_img_transform,
    ),
    collate_fn=MNISTRDataset.collate_fn,
    batch_size=batch_size,
    shuffle=True
  )

  return train_loader, test_loader


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


class MNISTRNet(nn.Module):
  def __init__(self, k, digit, bbox):
    super(MNISTRNet, self).__init__()

    self.digit = digit
    self.f = bbox
    # MNIST Digit Recognition Network
    self.mnist_net = MNISTNet()

    self.bbox = reinforce.REINFORCE(**{
      "bbox": bbox,
      "n_samples": k,
      "input_mappings": [input.DiscreteInputMapping(list(range(10))) for _ in range(digit)],
    })

  def forward(self, gt: torch.Tensor, x):
    batch_size = x[0].shape[0]
    x = torch.cat(x, dim=0)

    # First recognize the digits
    x = self.mnist_net(x)
    x = [x[i*batch_size:(i + 1) * batch_size,:] for i in range(self.digit)]
    
    # Wrap predictions in SingleInput objects
    inputs = [input.SingleInput(x[i]) for i in range(self.digit)]
    
    # Then execute the reasoning module
    return self.bbox(gt, *tuple(inputs))

  def get_test_preds(self, x):
    distrs = [self.mnist_net(x[i]) for i in range(self.digit)]
    preds = [d.argmax(dim=1) for d in distrs]
    return self.f(*tuple(preds))


def bce_loss(output, ground_truth):
  (_, dim) = output.shape
  gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in ground_truth])
  return F.binary_cross_entropy(output, gt)


class Trainer():
  def __init__(self,
               train_loader,
               test_loader,
               model_dir,
               learning_rate,
               k,
               digit,
               bbox):
    self.model_dir = model_dir
    self.network = MNISTRNet(k, digit, bbox)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.best_loss = 10000000000
    self.loss = bce_loss


  def train_epoch(self, epoch):
    self.network.train()
    iter = tqdm(self.train_loader, total=len(self.train_loader))
    for (data, target) in iter:
      self.optimizer.zero_grad()
      loss = self.network(target, data)
      loss.backward()
      self.optimizer.step()
      iter.set_description(f"[Train Epoch {epoch}] Loss: {loss.item():.4f}")

  def test_epoch(self, epoch):
    self.network.eval()
    num_items = len(self.test_loader.dataset)
    test_loss = 0
    correct = 0
    with torch.no_grad():
      iter = tqdm(self.test_loader, total=len(self.test_loader))
      for (data, target) in iter:
        test_loss += self.network(target, data)
        pred = self.network.get_test_preds(data)
        correct += pred.eq(target.data.view_as(pred)).sum()
        perc = 100. * correct / num_items
        iter.set_description(f"[Test Epoch {epoch}] Total loss: {test_loss:.4f}, Accuracy: {correct}/{num_items} ({perc:.2f}%)")
      if test_loss < self.best_loss:
        self.best_loss = test_loss

  def train(self, n_epochs):
    self.test_epoch(0)
    for epoch in range(1, n_epochs + 1):
      self.train_epoch(epoch)
      self.test_epoch(epoch)


if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("mnist_r")
  parser.add_argument("--n-epochs", type=int, default=10)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--learning-rate", type=float, default=0.0001)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--bbox", type=str, default="lambda a, b: a + b")
  parser.add_argument("--k", type=int, default=100)
  args = parser.parse_args()

  # Parameters
  n_epochs = args.n_epochs
  batch_size = args.batch_size
  learning_rate = args.learning_rate
  k = args.k
  torch.manual_seed(args.seed)
  random.seed(args.seed)
  bbox = eval(args.bbox)
  
  # Determine number of digits
  signature = inspect.signature(bbox)
  digit = len(signature.parameters)

  # Data
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../model/mnist_r"))
  os.makedirs(model_dir, exist_ok=True)

  # Dataloaders
  train_loader, test_loader = mnist_r_loader(data_dir, batch_size, digit, bbox)

  # Create trainer and train
  trainer = Trainer(train_loader, test_loader, model_dir, learning_rate, k, digit, bbox)
  trainer.train(n_epochs)