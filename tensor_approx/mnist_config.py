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
    mnist = torchvision.datasets.MNIST(
        root,
        train=train,
        transform=transform,
        target_transform=target_transform,
        download=download,
      )
    if self.digit >= 200:
      self.mnist_dataset = torch.utils.data.ConcatDataset([mnist, mnist, mnist, mnist, mnist])
    elif self.digit >= 100:
      self.mnist_dataset = torch.utils.data.ConcatDataset([mnist, mnist, mnist, mnist])
    elif self.digit >= 50:
      self.mnist_dataset = torch.utils.data.ConcatDataset([mnist, mnist, mnist])
    elif self.digit >= 25:
      self.mnist_dataset = torch.utils.data.ConcatDataset([mnist, mnist])
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
    sum =torch.stack([torch.tensor(item[-2]).long() for item in batch])
    digits = torch.stack([torch.tensor(item[-1]).long() for item in batch])
    return (tuple(imgs), sum, digits)


class MNISTMultiDigitSum2Dataset(torch.utils.data.Dataset):
  def __init__(
    self,
    root: str,
    digit: int,
    train: bool = True,
    transform: Optional[Callable] = None,
    target_transform: Optional[Callable] = None,
    download: bool = False,
  ):
    self.digit = digit # number of digits for each of the 2 numbers
    # Contains a MNIST dataset
    if train: self.length = 60000 # min(5000 * digit, 60000)
    else: self.length = 10000 # min(500 * digit, 10000)
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
    mnist = torchvision.datasets.MNIST(
        root,
        train=train,
        transform=transform,
        target_transform=target_transform,
        download=download,
      )
    total_digits = self.digit * 2
    if total_digits >= 200:
      self.mnist_dataset = torch.utils.data.ConcatDataset([mnist, mnist, mnist, mnist, mnist])
    elif total_digits >= 100:
      self.mnist_dataset = torch.utils.data.ConcatDataset([mnist, mnist, mnist, mnist])
    elif total_digits >= 50:
      self.mnist_dataset = torch.utils.data.ConcatDataset([mnist, mnist, mnist])
    elif total_digits >= 25:
      self.mnist_dataset = torch.utils.data.ConcatDataset([mnist, mnist])
    self.index_map = list(range(len(self.mnist_dataset)))
    random.shuffle(self.index_map)

    self.sum_dataset = []
    for i in range(len(self.mnist_dataset)//total_digits):
      self.sum_dataset.append([])
      for j in range(total_digits):
        self.sum_dataset[i].append(self.mnist_dataset[self.index_map[i*total_digits + j]])

  def __len__(self):
    return len(self.sum_dataset)

  def __getitem__(self, idx):
    # Get two data points
    item = self.sum_dataset[idx]
    data, target = [], []
    for (d,t) in item:
      data.append(d)
      target.append(t)
    target1 = int(''.join(str(t) for t in target[:len(target)//2]))
    target2 = int(''.join(str(t) for t in target[len(target)//2:]))
    
    target_sum = str(target1 + target2)
    if len(target_sum) <= self.digit:
      # add padding
      padding = self.digit - len(target_sum) + 1
      target_sum = ('0' * padding) + target_sum
    target_sum = torch.tensor([int(s) for s in reversed(target_sum)])

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

def mnist_multi_digit_sum2_loader(data_dir, batch_size, digit):
  train_loader = torch.utils.data.DataLoader(
    MNISTMultiDigitSum2Dataset(
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
    MNISTMultiDigitSum2Dataset(
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
    return F.softmax(x, dim=-1)

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