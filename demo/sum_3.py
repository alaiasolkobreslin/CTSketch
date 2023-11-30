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

import blackbox

mnist_img_transform = torchvision.transforms.Compose([
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(
    (0.1307,), (0.3081,)
  )
])

class MNISTSum3Dataset(torch.utils.data.Dataset):
  def __init__(
    self,
    root: str,
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
    self.index_map = list(range(len(self.mnist_dataset)))
    random.shuffle(self.index_map)

  def __len__(self):
    return int(len(self.mnist_dataset) / 3)

  def __getitem__(self, idx):
    # Get two data points
    (a_img, a_digit) = self.mnist_dataset[self.index_map[idx * 3]]
    (b_img, b_digit) = self.mnist_dataset[self.index_map[idx * 3 + 1]]
    (c_img, c_digit) = self.mnist_dataset[self.index_map[idx * 3 + 2]]

    # Each data has two images and the GT is the sum of two digits
    return (a_img, b_img, c_img, a_digit + b_digit + c_digit)

  @staticmethod
  def collate_fn(batch):
    a_imgs = torch.stack([item[0] for item in batch])
    b_imgs = torch.stack([item[1] for item in batch])
    c_imgs = torch.stack([item[2] for item in batch])
    digits = torch.stack([torch.tensor(item[3]).long() for item in batch])
    return ((a_imgs, b_imgs, c_imgs), digits)


def mnist_sum_3_loader(data_dir, batch_size):
  train_loader = torch.utils.data.DataLoader(
    MNISTSum3Dataset(
      data_dir,
      train=True,
      download=True,
      transform=mnist_img_transform,
    ),
    collate_fn=MNISTSum3Dataset.collate_fn,
    batch_size=batch_size,
    shuffle=True
  )

  test_loader = torch.utils.data.DataLoader(
    MNISTSum3Dataset(
      data_dir,
      train=False,
      download=True,
      transform=mnist_img_transform,
    ),
    collate_fn=MNISTSum3Dataset.collate_fn,
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

def sum_3(xa, xb, xc):
  y_dim = xa.shape[1] + xb.shape[1] + xc.shape[1]
  y_pred = torch.argmax(xa, dim=1) + torch.argmax(xb, dim=1) + torch.argmax(xc, dim=1)
  return F.one_hot(y_pred, num_classes = y_dim).float()

class MNISTSum3Net(nn.Module):
  def __init__(self):
    super(MNISTSum3Net, self).__init__()

    # MNIST Digit Recognition Network
    self.mnist_net = MNISTNet()
    self.bbox = blackbox.BlackBoxFunction.apply

  def forward(self, x: Tuple[torch.Tensor, torch.Tensor]):
    (a_imgs, b_imgs, c_imgs) = x

    # First recognize the two digits
    a_distrs = self.mnist_net(a_imgs) # Tensor 64 x 10
    b_distrs = self.mnist_net(b_imgs) # Tensor 64 x 10
    c_distrs = self.mnist_net(c_imgs)

    # Then execute the reasoning module; the result is a size 19 tensor
    blackbox.BlackBoxFunction.fn = sum_3
    return self.bbox(a_distrs, b_distrs, c_distrs) # Tensor 64 x 19


class Trainer():
  def __init__(self, train_loader, test_loader, model_dir, learning_rate):
    self.model_dir = model_dir
    self.network = MNISTSum3Net()
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.best_loss = 10000000000

  def loss(self, output, ground_truth):
    (_, dim) = output.shape
    gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in ground_truth])
    return F.binary_cross_entropy(output, gt)

  def train_epoch(self, epoch):
    self.network.train()
    iter = tqdm(self.train_loader, total=len(self.train_loader))
    for (data, target) in iter:
      self.optimizer.zero_grad()
      output = self.network(data)
      loss = self.loss(output, target)
      loss.backward()
      self.optimizer.step()
      iter.set_description(f"[Train Epoch {epoch}] Loss: {loss.item():.4f} Accuracy: {(output.argmax(dim=1)==target).float().mean():.2f}")

  def test_epoch(self, epoch):
    self.network.eval()
    num_items = len(self.test_loader.dataset)
    test_loss = 0
    correct = 0
    with torch.no_grad():
      iter = tqdm(self.test_loader, total=len(self.test_loader))
      for (data, target) in iter:
        output = self.network(data)
        test_loss += self.loss(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        perc = 100. * correct / num_items
        iter.set_description(f"[Test Epoch {epoch}] Total loss: {test_loss:.4f}, Accuracy: {correct}/{num_items} ({perc:.2f}%)")

  def train(self, n_epochs):
    for epoch in range(1, n_epochs + 1):
      self.train_epoch(epoch)
      self.test_epoch(epoch)


if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("mnist_sum_3")
  parser.add_argument("--n-epochs", type=int, default=20)
  parser.add_argument("--batch-size", type=int, default=128)
  parser.add_argument("--learning-rate", type=float, default=0.0004)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--jit", action="store_true")
  parser.add_argument("--dispatch", type=str, default="parallel")
  args = parser.parse_args()

  # Parameters
  n_epochs = args.n_epochs
  batch_size = args.batch_size
  learning_rate = args.learning_rate
  torch.manual_seed(args.seed)
  random.seed(args.seed)

  # Data
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../model/mnist_sum_3"))
  os.makedirs(model_dir, exist_ok=True)

  # Dataloaders
  train_loader, test_loader = mnist_sum_3_loader(data_dir, batch_size)

  # Create trainer and train
  trainer = Trainer(train_loader, test_loader, model_dir, learning_rate)
  trainer.train(n_epochs)
