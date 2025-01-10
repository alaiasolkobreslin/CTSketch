import random
from typing import *

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

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
    r = (self.digit * 6000) % 60000
    q = (self.digit * 6000) // 60000

    # Contains a MNIST dataset
    if train: self.length = r
    else: self.length = r // 6
    self.digit = digit
    mnist_sub = torch.utils.data.Subset(
      torchvision.datasets.MNIST(
        root,
        train=train,
        transform=transform,
        target_transform=target_transform,
        download=download,
      ),
      range(self.length)
    )
    mnist = torchvision.datasets.MNIST(
        root,
        train=train,
        transform=transform,
        target_transform=target_transform,
        download=download,
      )
    self.mnist_dataset = torch.utils.data.ConcatDataset([mnist]*q + [mnist_sub])
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
    
    target_sum = sum(tuple(target))

    # Each data has two images and the GT is the sum of two digits
    return (*tuple(data), target_sum, target)

  @staticmethod
  def collate_fn(batch):
    imgs = []
    for i in range(len(batch[0])-2):
      imgs.append(torch.stack([item[i] for item in batch]))
    sum = torch.stack([torch.tensor(item[-2]).long() for item in batch])
    digits = torch.stack([torch.tensor(item[-1]).long() for item in batch])
    return (tuple(imgs), sum, digits)


def mnist_sum_loader(data_dir, batch_size, digit):
  dataset = MNISTSumDataset(
      data_dir,
      digit,
      train=True,
      download=True,
      transform=mnist_img_transform,
    )
  train_dataset, val_dataset = torch.utils.data.random_split(dataset, [5000, 1000])
  train_loader = torch.utils.data.DataLoader(
    train_dataset,
    collate_fn=MNISTSumDataset.collate_fn,
    batch_size=batch_size,
    shuffle=True
  )

  val_loader = torch.utils.data.DataLoader(
    val_dataset,
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

  return train_loader, val_loader, test_loader

class MNISTNet(nn.Module):
  def __init__(self, with_softmax=True):
    super(MNISTNet, self).__init__()
    self.with_softmax = with_softmax
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
    if self.with_softmax: x = F.softmax(x, dim=-1)
    return x

class MNISTSumNet(nn.Module):
  def __init__(self, digit, with_softmax=True):
    super(MNISTSumNet, self).__init__()

    self.mnist_net = MNISTNet(with_softmax)
    self.digit = digit

  def forward(self, x):
    batch_size = x[0].shape[0]
    x = torch.cat(x, dim=0)
    x = self.mnist_net(x)
    x = [x[i*batch_size:(i + 1) * batch_size,:] for i in range(self.digit)]
    return tuple(x)