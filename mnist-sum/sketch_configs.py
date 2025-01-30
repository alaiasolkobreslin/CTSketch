import torch

def theta2(elems):
  x1 = torch.arange(0, elems).reshape(elems, 1).expand(elems, elems)
  x2 = torch.arange(0, elems).reshape(1, elems).expand(elems, elems)
  return x1, x2

def theta3(elems):
  x1 = torch.arange(0, elems).reshape(elems, 1, 1).expand(elems, elems, elems)
  x2 = torch.arange(0, elems).reshape(1, elems, 1).expand(elems, elems, elems)
  x3 = torch.arange(0, elems).reshape(1, 1, elems).expand(elems, elems, elems)
  return x1, x2, x3

def theta4(elems):
  x1 = torch.arange(0, elems).reshape(elems, 1, 1, 1).expand(elems, elems, elems, elems)
  x2 = torch.arange(0, elems).reshape(1, elems, 1, 1).expand(elems, elems, elems, elems)
  x3 = torch.arange(0, elems).reshape(1, 1, elems, 1).expand(elems, elems, elems, elems)
  x4 = torch.arange(0, elems).reshape(1, 1, 1, elems).expand(elems, elems, elems, elems)
  return x1, x2, x3, x4

def stream_theta2(digit, elems, output_dim, samples):
  x1, x2 = theta2(elems)
  return x1.flatten(), x2.flatten()

def full_theta2(digit, elems, output_dim, samples):
  x1, x2 = theta2(elems)
  x = x1 + x2
  return x

def stream_theta3(digit, elems, output_dim, samples):
  x1, x2, x3 = theta3(elems)
  return x1.flatten(), x2.flatten(), x3.flatten()

def full_theta3(digit, elems, output_dim, samples):
  x1, x2, x3 = theta3(elems)
  x = x1 + x2 + x3
  return x

def stream_theta4(digit, elems, output_dim, samples):
  x1, x2, x3, x4 = theta4(elems)
  return x1.flatten(), x2.flatten(), x3.flatten(), x4.flatten()

def full_theta4(digit, elems, output_dim, samples):
  x1, x2, x3, x4 = theta4(elems)
  x = x1 + x2 + x3 + x4
  return x

def random_theta(digit, elems, output_dim, samples):
  xs = []
  for i in range(digit):
    xs.append(torch.randint(0, elems, (samples,)))
  xs = torch.stack(xs, dim=0).t()
  t = torch.ones(tuple([elems]*digit)).long()*(-1)
  ys = torch.randint(0, output_dim, (samples, ))
  for x, y in zip(xs, ys):
    t[tuple(x)] = y
  return t

def sample_theta(digit, elems, output_dim, samples):
  xs = []
  for i in range(digit):
    xs.append(torch.randint(0, elems, (samples,)))
  xs = torch.stack(xs, dim=0)
  # t = torch.randint(0, output_dim, tuple([elems]*digit))
  t = torch.zeros(tuple([elems]*digit)).long()
  for i in range(samples):
    x_i = tuple(xs[:, i].tolist())
    t[x_i] = sum(x_i)
  return t
