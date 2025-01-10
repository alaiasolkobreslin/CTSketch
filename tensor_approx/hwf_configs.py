import os
from typing import *
import json
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class HWFDataset(torch.utils.data.Dataset):
  def __init__(self, root: str, prefix: str, split: str, max_len):
    super(HWFDataset, self).__init__()
    self.root = root
    self.split = split
    self.metadata = json.load(open(os.path.join(root, f"HWF/{prefix}_{split}.json")))
    self.metadata = [m for m in self.metadata if len(m['img_paths']) <= max_len]
    self.img_transform = torchvision.transforms.Compose([
      torchvision.transforms.ToTensor(),
      torchvision.transforms.Normalize((0.5,), (1,))])
  
  def __len__(self):
    return len(self.metadata)

  def __getitem__(self, index):
    sample = self.metadata[index]

    # Input is a sequence of images
    img_seq = []
    for img_path in sample["img_paths"]:
      img_full_path = os.path.join(self.root, "HWF/Handwritten_Math_Symbols", img_path)
      img = Image.open(img_full_path).convert("L")
      img = self.img_transform(img)
      img_seq.append(img)

    # Output is the "res" in the sample of metadata
    res = sample["res"]
    img_seq_len = len(img_seq)

    # Return (input, output) pair
    return (img_seq, img_seq_len, res)

  @staticmethod
  def collate_fn(batch):
    # max_len = max([img_seq_len for (_, img_seq_len, _) in batch])
    # zero_img = torch.zeros_like(batch[0][0][0])
    # pad_zero = lambda img_seq: img_seq + [zero_img] * (max_len - len(img_seq))
    # img_seqs = torch.stack([torch.stack(pad_zero(img_seq)) for (img_seq, _, _) in batch])
    img_seqs = torch.cat([torch.stack(img_seq, dim=0) for (img_seq, _, _) in batch], dim=0)
    img_seq_len = torch.stack([torch.tensor(img_seq_len).long() for (_, img_seq_len, _) in batch])
    results = torch.stack([torch.tensor(res) for (_, _, res) in batch])
    return (img_seqs, img_seq_len, results)

def hwf_loader(data_dir, batch_size, prefix, max_len):
  train_loader = torch.utils.data.DataLoader(HWFDataset(data_dir, prefix, "train", max_len),
                                            collate_fn=HWFDataset.collate_fn, 
                                            batch_size=batch_size, 
                                            shuffle=True)
  test_loader = torch.utils.data.DataLoader(HWFDataset(data_dir, prefix, "test", max_len), 
                                            collate_fn=HWFDataset.collate_fn, 
                                            batch_size=batch_size, 
                                            shuffle=True)
  return (train_loader, test_loader)

class SymbolNet(nn.Module):
    def __init__(self):
        super(SymbolNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.fc1 = nn.Linear(30976, 128)
        self.fc2 = nn.Linear(128, 14)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.dropout(x, p=0.25, training=self.training)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc2(x)
        return x # F.softmax(x, dim=1)

def eval_result_eq(a, b, threshold=0.01):
    result = abs(a - b) < threshold
    return result

def hwf_eval(symbols: List[str]):
  for i, s in enumerate(symbols):
    if i % 2 == 0 and not s.isdigit(): raise Exception("BAD")
    if i % 2 == 1 and s not in ["+", "-", "*", "/"]: raise Exception("BAD")
  return eval("".join(symbols))