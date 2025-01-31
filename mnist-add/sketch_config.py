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

def theta5(elems):
  x1 = torch.arange(0, elems).reshape(elems, 1, 1, 1, 1)
  x2 = torch.arange(0, elems).reshape(1, elems, 1, 1, 1)
  x3 = torch.arange(0, elems).reshape(1, 1, elems, 1, 1)
  x4 = torch.arange(0, elems).reshape(1, 1, 1, elems, 1)
  x5 = torch.arange(0, elems).reshape(1, 1, 1, 1, elems)
  return x1, x2, x3, x4, x5

def theta6(elems):
  x1 = torch.arange(0, elems).reshape(elems, 1, 1, 1, 1, 1).expand(elems, elems, elems, elems, elems, elems)
  x2 = torch.arange(0, elems).reshape(1, elems, 1, 1, 1, 1).expand(elems, elems, elems, elems, elems, elems)
  x3 = torch.arange(0, elems).reshape(1, 1, elems, 1, 1, 1).expand(elems, elems, elems, elems, elems, elems)
  x4 = torch.arange(0, elems).reshape(1, 1, 1, elems, 1, 1).expand(elems, elems, elems, elems, elems, elems)
  x5 = torch.arange(0, elems).reshape(1, 1, 1, 1, elems, 1).expand(elems, elems, elems, elems, elems, elems)
  x6 = torch.arange(0, elems).reshape(1, 1, 1, 1, 1, elems).expand(elems, elems, elems, elems, elems, elems)
  return x1, x2, x3, x4, x5, x6

def theta7(elems):
  x1 = torch.arange(0, elems).reshape(elems, 1, 1, 1, 1, 1, 1).expand(elems, elems, elems, elems, elems, elems, elems)
  x2 = torch.arange(0, elems).reshape(1, elems, 1, 1, 1, 1, 1).expand(elems, elems, elems, elems, elems, elems, elems)
  x3 = torch.arange(0, elems).reshape(1, 1, elems, 1, 1, 1, 1).expand(elems, elems, elems, elems, elems, elems, elems)
  x4 = torch.arange(0, elems).reshape(1, 1, 1, elems, 1, 1, 1).expand(elems, elems, elems, elems, elems, elems, elems)
  x5 = torch.arange(0, elems).reshape(1, 1, 1, 1, elems, 1, 1).expand(elems, elems, elems, elems, elems, elems, elems)
  x6 = torch.arange(0, elems).reshape(1, 1, 1, 1, 1, elems, 1).expand(elems, elems, elems, elems, elems, elems, elems)
  x7 = torch.arange(0, elems).reshape(1, 1, 1, 1, 1, 1, elems).expand(elems, elems, elems, elems, elems, elems, elems)
  return x1, x2, x3, x4, x5, x6, x7

def theta8(elems):
  x1 = torch.arange(0, elems).reshape(elems, 1, 1, 1, 1, 1, 1, 1).expand(elems, elems, elems, elems, elems, elems, elems, elems)
  x2 = torch.arange(0, elems).reshape(1, elems, 1, 1, 1, 1, 1, 1).expand(elems, elems, elems, elems, elems, elems, elems, elems)
  x3 = torch.arange(0, elems).reshape(1, 1, elems, 1, 1, 1, 1, 1).expand(elems, elems, elems, elems, elems, elems, elems, elems)
  x4 = torch.arange(0, elems).reshape(1, 1, 1, elems, 1, 1, 1, 1).expand(elems, elems, elems, elems, elems, elems, elems, elems)
  x5 = torch.arange(0, elems).reshape(1, 1, 1, 1, elems, 1, 1, 1).expand(elems, elems, elems, elems, elems, elems, elems, elems)
  x6 = torch.arange(0, elems).reshape(1, 1, 1, 1, 1, elems, 1, 1).expand(elems, elems, elems, elems, elems, elems, elems, elems)
  x7 = torch.arange(0, elems).reshape(1, 1, 1, 1, 1, 1, elems, 1).expand(elems, elems, elems, elems, elems, elems, elems, elems)
  x8 = torch.arange(0, elems).reshape(1, 1, 1, 1, 1, 1, 1, elems).expand(elems, elems, elems, elems, elems, elems, elems, elems)
  return x1, x2, x3, x4, x5, x6, x7, x8

def theta9(elems):
  x1 = torch.arange(0, elems).reshape(elems, 1, 1, 1, 1, 1, 1, 1, 1).expand(elems, elems, elems, elems, elems, elems, elems, elems, elems)
  x2 = torch.arange(0, elems).reshape(1, elems, 1, 1, 1, 1, 1, 1, 1).expand(elems, elems, elems, elems, elems, elems, elems, elems, elems)
  x3 = torch.arange(0, elems).reshape(1, 1, elems, 1, 1, 1, 1, 1, 1).expand(elems, elems, elems, elems, elems, elems, elems, elems, elems)
  x4 = torch.arange(0, elems).reshape(1, 1, 1, elems, 1, 1, 1, 1, 1).expand(elems, elems, elems, elems, elems, elems, elems, elems, elems)
  x5 = torch.arange(0, elems).reshape(1, 1, 1, 1, elems, 1, 1, 1, 1).expand(elems, elems, elems, elems, elems, elems, elems, elems, elems)
  x6 = torch.arange(0, elems).reshape(1, 1, 1, 1, 1, elems, 1, 1, 1).expand(elems, elems, elems, elems, elems, elems, elems, elems, elems)
  x7 = torch.arange(0, elems).reshape(1, 1, 1, 1, 1, 1, elems, 1, 1).expand(elems, elems, elems, elems, elems, elems, elems, elems, elems)
  x8 = torch.arange(0, elems).reshape(1, 1, 1, 1, 1, 1, 1, elems, 1).expand(elems, elems, elems, elems, elems, elems, elems, elems, elems)
  x9 = torch.arange(0, elems).reshape(1, 1, 1, 1, 1, 1, 1, 1, elems).expand(elems, elems, elems, elems, elems, elems, elems, elems, elems)
  return x1, x2, x3, x4, x5, x6, x7, x8, x9

def stream_theta2(digit, elems, output_dim, samples):
  x1, x2 = theta2(elems)
  return x1.flatten(), x2.flatten()

def full_theta2(digit, elems, output_dim, samples):
  x1, x2 = theta2(elems)
  x = x1 + x2
  # r = random_theta(digit, elems, output_dim, samples)
  #r = torch.where(r > 0, r, x)
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

def stream_theta5(digit, elems, output_dim, samples):
  x1, x2, x3, x4, x5 = theta5(elems)
  return x1.flatten(), x2.flatten(), x3.flatten(), x4.flatten(), x5.flatten()

def full_theta5(digit, elems, output_dim, samples):
  x1, x2, x3, x4, x5 = theta5(elems)
  x = x1 + x2 + x3 + x4 + x5
  return x

def stream_theta6(digit, elems, output_dim, samples):
  x1, x2, x3, x4, x5, x6 = theta6(elems)
  return x1.flatten(), x2.flatten(), x3.flatten(), x4.flatten(), x5.flatten(), x6.flatten()

def full_theta6(digit, elems, output_dim, samples):
  x1, x2, x3, x4, x5, x6 = theta6(elems)
  x = x1 + x2 + x3 + x4 + x5 + x6
  return x

def stream_theta7(digit, elems, output_dim, samples):
  x1, x2, x3, x4, x5, x6, x7 = theta7(elems)
  return x1.flatten(), x2.flatten(), x3.flatten(), x4.flatten(), x5.flatten(), x6.flatten(), x7.flatten()

def full_theta7(digit, elems, output_dim, samples):
  x1, x2, x3, x4, x5, x6, x7 = theta7(elems)
  x = x1 + x2 + x3 + x4 + x5 + x6 + x7
  return x

def stream_theta8(digit, elems, output_dim, samples):
  x1, x2, x3, x4, x5, x6, x7, x8 = theta8(elems)
  return x1.flatten(), x2.flatten(), x3.flatten(), x4.flatten(), x5.flatten(), x6.flatten(), x7.flatten(), x8.flatten()

def full_theta8(digit, elems, output_dim, samples):
  x1, x2, x3, x4, x5, x6, x7, x8 = theta8(elems)
  x = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8
  return x

def stream_theta9(digit, elems, output_dim, samples):
  x1, x2, x3, x4, x5, x6, x7, x8, x9 = theta9(elems)
  return x1.flatten(), x2.flatten(), x3.flatten(), x4.flatten(), x5.flatten(), x6.flatten(), x7.flatten(), x8.flatten(), x9.flatten()

def full_theta9(digit, elems, output_dim, samples):
  x1, x2, x3, x4, x5, x6, x7, x8, x9 = theta9(elems)
  x = x1 + x2 + x3 + x4 + x5 + x6 + x7 + x8 + x9
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