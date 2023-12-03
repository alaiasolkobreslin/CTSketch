import torch
import torch.nn.functional as F

def sum_2(xa, xb):
  y_dim = xa.shape[1] + xb.shape[1] - 1
  y_pred = torch.argmax(xa, dim=1) + torch.argmax(xb, dim=1)
  return F.one_hot(y_pred, num_classes = y_dim).float()

def sum_3(xa, xb, xc):
  y_dim = xa.shape[1] + xb.shape[1] + xc.shape[1] - 2
  y_pred = torch.argmax(xa, dim=1) + torch.argmax(xb, dim=1) + torch.argmax(xc, dim=1)
  return F.one_hot(y_pred, num_classes = y_dim).float()

def sum_4(xa, xb, xc, xd):
  y_dim = xa.shape[1] + xb.shape[1] + xc.shape[1] + xd.shape[1] - 3
  y_pred = torch.argmax(xa, dim=1) + torch.argmax(xb, dim=1) + torch.argmax(xc, dim=1) + torch.argmax(xd, dim=1)
  return F.one_hot(y_pred, num_classes = y_dim).float()