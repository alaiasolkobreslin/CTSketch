import torch

def theta2(elems):
  x1 = torch.arange(0, elems).reshape(elems, 1).expand(elems, elems)
  x2 = torch.arange(0, elems).reshape(1, elems).expand(elems, elems)
  return x1, x2

def full_theta2(digit, elems, output_dim, samples):
  x1, x2 = theta2(elems)
  x = x1 + x2
  return x