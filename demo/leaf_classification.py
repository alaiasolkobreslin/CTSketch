import os
import random
from typing import *

import torch
import torchvision
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from PIL import Image

from argparse import ArgumentParser
from tqdm import tqdm

import blackbox
import task_program

leaves_img_transform = torchvision.transforms.Compose([
  torchvision.transforms.ToTensor(),
  torchvision.transforms.Normalize(
    (0.1307,), (0.3081,)
  )
])

class LeavesDataset(torch.utils.data.Dataset):
  """
  :param data_root, the root directory of the data folder
  """
  def __init__(
    self,
    data_root: str,
    data_dir: str,
    transform: Optional[Callable] = leaves_img_transform,
  ):
    self.transform = transform
    
    # Get all image paths and their labels
    self.samples = []
    n_group = 0
    data_dir = os.path.join(data_root, data_dir)
    for sample_group in os.listdir(data_dir):
      sample_group_dir = os.path.join(data_dir, sample_group)
      if not os.path.isdir(sample_group_dir):
        continue
      for sample_group_file in os.listdir(sample_group_dir):
        sample_img_path = os.path.join(sample_group_dir, sample_group_file)
        if sample_img_path.endswith('JPG') or sample_img_path.endswith('png'):
          self.samples.append((sample_img_path, n_group))
      n_group += 1
    
    self.index_map = list(range(len(self.samples)))
    random.shuffle(self.index_map)

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    (img_path, label) = self.samples[self.index_map[idx]]
    img = Image.open(open(img_path, "rb"))
    img = self.transform(img)
    return (img, label)
  
  @staticmethod
  def collate_fn(batch):
    imgs = torch.stack([item[0] for item in batch])
    labels = torch.stack([torch.tensor(item[1]).long() for item in batch])
    return (imgs, labels)

def leaves_loader(data_root, data_dir, batch_size, train_percentage):
  dataset = LeavesDataset(data_root, data_dir)
  num_train = int(len(dataset) * train_percentage)
  num_test = len(dataset) - num_train
  (train_dataset, test_dataset) = torch.utils.data.random_split(dataset, [num_train, num_test])
  train_loader = torch.utils.data.DataLoader(train_dataset, collate_fn=LeavesDataset.collate_fn, batch_size=batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_dataset, collate_fn=LeavesDataset.collate_fn, batch_size=batch_size, shuffle=True)
  return (train_loader, test_loader)

class LeafNet(nn.Module):
  def __init__(self, num_features):
    super(LeafNet, self).__init__()
    self.num_features = num_features
    self.dim = task_program.l11_dim

    # CNN
    self.cnn = nn.Sequential(
      nn.Conv2d(3, 32, 10, 1),
      nn.ReLU(),
      nn.MaxPool2d(3),
      nn.Conv2d(32, 64, 5, 1),
      nn.ReLU(),
      nn.MaxPool2d(3),
      nn.Conv2d(64, 128, 3, 1),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Conv2d(128, 128, 3, 1),
      nn.ReLU(),
      nn.MaxPool2d(2),
      nn.Flatten(),
    )

    # Fully connected for 'features'
    self.features_fc = nn.Sequential(
      nn.Linear(self.dim, self.dim),
      nn.ReLU(),
      nn.Linear(self.dim, self.num_features),
      nn.Softmax(dim=1)
    )
  
  def forward(self, x):
    x = self.cnn(x)
    x = self.features_fc(x)   
    return x

class LeavesNet(nn.Module):
  def __init__(self):
    super(LeavesNet, self).__init__()
    
    # features for classification
    self.margin = task_program.l11_margin
    self.shape = task_program.l11_shape
    self.texture = task_program.l11_texture
    self.labels = task_program.l11_labels
    self.dim = task_program.l11_dim

    self.net1 = LeafNet(len(self.margin))
    self.net2 = LeafNet(len(self.shape))
    self.net3 = LeafNet(len(self.texture))

    # Blackbox encoding identification chart
    self.bbox = blackbox.BlackBox(
      task_program.classify_11,
      (blackbox.DiscreteInputMapping(self.margin),
       blackbox.DiscreteInputMapping(self.shape),
       blackbox.DiscreteInputMapping(self.texture)),
      blackbox.DiscreteOutputMapping(self.labels))

  def forward(self, x):
    has_margin = self.net1(x)
    has_shape = self.net2(x)
    has_texture = self.net3(x)
    return self.bbox(has_margin, has_shape, has_texture)

class Trainer():
  def __init__(self, train_loader, test_loader, learning_rate, gpu, save_model=False):
    if gpu >= 0:
      device = torch.device("cuda:%d" % gpu)
    else:
      device = torch.device("cpu")
    self.device = device
    self.network = LeavesNet() #.to(self.device)
    self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.save_model = save_model

    # Aggregated loss (initialized to be a huge number)
    self.min_test_loss = 100000000.0

  def loss_fn(self, output, ground_truth):
    dim = output.shape[1]
    gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in ground_truth])
    return F.binary_cross_entropy(output, gt)
  
  def train_epoch(self, epoch):
    self.network.train()
    num_items = 0
    total_correct = 0
    iter = tqdm(self.train_loader, total=len(self.train_loader))
    for (input, target) in iter:
      self.optimizer.zero_grad()
      input = input.to(self.device)
      target = target.to(self.device)
      output = self.network(input)
      loss = self.loss_fn(output, target)
      loss.backward()
      self.optimizer.step()
      total_correct += (output.argmax(dim=1)==target).float().sum()
      num_items += output.shape[0]
      correct_perc = 100. * total_correct / num_items
      iter.set_description(f"[Train Epoch {epoch}] Loss: {loss.item():.4f}, Overall Accuracy: {correct_perc:.4f}%")

  def test_epoch(self, epoch):
    self.network.eval()
    num_items = len(self.test_loader.dataset)
    num_correct = 0
    test_loss = 0
    with torch.no_grad():
      iter = tqdm(self.test_loader, total=len(self.test_loader))
      for (input, target) in iter:
        input = input.to(self.device)
        target = target.to(self.device)
        output = self.network(input)
        test_loss += self.loss_fn(output, target).item()
        num_correct += (output.argmax(dim=1)==target).float().sum()
        perc = 100.*num_correct/num_items
        iter.set_description(f"[Test Epoch {epoch}] Total loss: {test_loss:.4f}, Accuracy: {num_correct}/{num_items} ({perc:.2f})%")
    
    # Save the model
    # if self.save_model and test_loss < self.min_test_loss:
    #  self.min_test_loss = test_loss
    #  torch.save(self.network, "../model/leaves/leaves_net.pkl")

  def train(self, n_epochs):
    for epoch in range(1, n_epochs+1):
      self.train_epoch(epoch)
      self.test_epoch(epoch)

if __name__ == "__main__":
  # Argument parser
  parser = ArgumentParser("leaves")
  parser.add_argument("--model-name", type=str, default="leaves.pkl")
  parser.add_argument("--n-epochs", type=int, default=30)
  parser.add_argument("--gpu", type=int, default=-1)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--train-percentage", type=float, default=0.8)
  parser.add_argument("--learning-rate", type=float, default=0.0002)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--data-dir", type=str, default="leaf_11")
  parser.add_argument("--cuda", action="store_true")
  args = parser.parse_args()

  # Setup parameters
  torch.manual_seed(args.seed)
  random.seed(args.seed)

  # Load data
  data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../model/leaves"))
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  (train_loader, test_loader) = leaves_loader(data_root, args.data_dir, args.batch_size, args.train_percentage)
  trainer = Trainer(train_loader, test_loader, args.learning_rate, args.gpu)

  # Run
  trainer.train(args.n_epochs)