import os
import random
from typing import *
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torch.optim as optim

from argparse import ArgumentParser

from torch_modules.finite_difference import FiniteDifference, DiscreteOutputMapping, DiscreteInputMapping

def n_plus_one(num):
    return num + 1

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
    
    target = n_plus_one(*tuple(target))

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

class MNISTLinearNet(nn.Module):
  def __init__(self):
    super(MNISTLinearNet, self).__init__()
    self.linear1 = nn.Linear(10, 10)

  def forward(self, x):
    x = F.softmax(self.linear1(x), dim=1)
    return x 

class MNISTSumNet(nn.Module):
  def __init__(self, digit):
    super(MNISTSumNet, self).__init__()

    # MNIST Digit Recognition Network
    self.mnist_net = MNISTNet()
    self.digit = digit
    self.input_mapping = [DiscreteInputMapping(list(range(10)))]*self.digit

    # Scallop Context
    self.bbox = FiniteDifference(**{
        "bbox": n_plus_one,
        "input_mappings": tuple(self.input_mapping),
        "output_mapping": DiscreteOutputMapping(list(range(1, 11)))
        })

  def forward(self, x):
    batch_size = x[0].shape[0]
    x = torch.cat(x, dim=0)

    # First recognize the two digits
    x = self.mnist_net(x)
    x = [x[i*batch_size:(i + 1) * batch_size,:] for i in range(self.digit)]
    
    # Then execute the reasoning module; the result is a size 19 tensor
    return self.bbox(*tuple(x))

class Trainer():
  def __init__(self, model, digit, train_loader, test_loader, model_dir, learning_rate):
    self.model_dir = model_dir
    self.network = model(digit)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.best_loss = 10000000000
    self.linear_net = MNISTLinearNet()
    self.linear_optimizer = optim.Adam(self.linear_net.parameters(), lr=0.001)

  def loss(self, output, ground_truth):
    dim = output.shape[1]
    gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in ground_truth])
    return F.binary_cross_entropy(output, gt)

  def train_epoch(self, epoch):
    self.network.train()
    self.linear_net.train()
    num_items = 0
    total_correct = 0
    iter = tqdm(self.train_loader, total=len(self.train_loader))
    for (data, target) in iter:
      self.optimizer.zero_grad()
      self.linear_optimizer.zero_grad()
      _, output = self.network(data)
      output = self.linear_net(output)
      loss = self.loss(output, target)
      loss.backward()
      self.linear_optimizer.step()
      self.optimizer.step()
      total_correct += (output.argmax(dim=1)==target).float().sum()
      num_items += output.shape[0]
      correct_perc = 100. * total_correct / num_items
      iter.set_description(f"[Train Epoch {epoch}] Loss: {loss.item():.4f} Overall Accuracy: {correct_perc:.4f}%")

  def test_epoch(self, epoch):
    self.network.eval()
    self.linear_net.eval()
    num_items = len(self.test_loader.dataset)
    test_loss = 0
    correct = 0
    with torch.no_grad():
      iter = tqdm(self.test_loader, total=len(self.test_loader))
      for (data, target) in iter:
        _, output = self.network(data)
        output = self.linear_net(output)
        test_loss += self.loss(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
        perc = 100. * correct / num_items
        iter.set_description(f"[Test Epoch {epoch}] Total loss: {test_loss:.4f}, Accuracy: {correct}/{num_items} ({perc:.2f}%)")

  def train(self, n_epochs):
    # self.network.load_state_dict(torch.load(self.model_dir+"/model.pth"))
    for epoch in range(1, n_epochs + 1):
      self.train_epoch(epoch)
      self.test_epoch(epoch)
    torch.save(self.network.state_dict(), self.model_dir+"/model.pth")

if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("mnist_sum")
  parser.add_argument("--n-epochs", type=int, default=200)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--learning-rate", type=float, default=1e-3)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--jit", action="store_true")
  parser.add_argument("--dispatch", type=str, default="parallel")
  args = parser.parse_args()

  # Parameters
  n_epochs = args.n_epochs
  batch_size = args.batch_size
  learning_rate = args.learning_rate
  digit = 1
  torch.manual_seed(args.seed)
  random.seed(args.seed)

  # Data
  data_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../model/mnist_sum"))
  os.makedirs(model_dir, exist_ok=True)

  # Dataloaders
  train_loader, test_loader = mnist_sum_loader(data_dir, batch_size, digit)

  # Create trainer and train
  trainer = Trainer(MNISTSumNet, digit, train_loader, test_loader, model_dir, learning_rate)
  trainer.train(n_epochs)
