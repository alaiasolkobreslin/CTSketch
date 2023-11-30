import torch
import torch.nn.functional as F

def finite_difference_2(fn, *inputs):
  argmax_inputs, inputs_distr = [], []
  k = 0
  for x in inputs:
    n = x.shape[1]
    k += n
    argmax_inputs.append(F.one_hot(x.argmax(dim=1), num_classes = n))
    inputs_distr.append(x)

  y_pred = fn(*tuple(argmax_inputs))

  jacobian = []
  for x, count in zip(inputs, range(len(inputs))):
    batch_size, n = x.shape
    jacobian_i = torch.zeros(batch_size, k, n)
    for i in torch.arange(0,n):
      inputs_i = argmax_inputs.copy()
      inputs_i[count] = F.one_hot(i, num_classes=n).float().repeat(batch_size,1)
      y = fn(*tuple(inputs_i)) - y_pred
      
      probs = argmax_inputs[1-count].argmax(dim=1)
      for probs_i, batch_i in zip(probs, range(batch_size)):
        freq = torch.where(probs==probs_i, 1, 0).sum()
        prob = inputs_distr[1-count][:,probs_i].unsqueeze(1)/freq
        jacobian_i[:,:,i] += y[batch_i].unsqueeze(0).repeat(batch_size, 1)*prob/n
    jacobian.append(jacobian_i)

  return tuple(jacobian)

def finite_difference_3(fn, *inputs):
  argmax_inputs, inputs_distr = [], []
  k = 0
  for x in inputs:
    n = x.shape[1]
    k += n
    argmax_inputs.append(F.one_hot(x.argmax(dim=1), num_classes = n))
    inputs_distr.append(x)

  y_pred = fn(*tuple(argmax_inputs))

  jacobian = []
  for x, count in zip(inputs, range(len(inputs))):
    batch_size, n = x.shape
    jacobian_i = torch.zeros(batch_size, k, n)
    for i in torch.arange(0,n):
      inputs_i = argmax_inputs.copy()
      inputs_i[count] = F.one_hot(i, num_classes=n).float().repeat(batch_size,1)
      y = fn(*tuple(inputs_i)) - y_pred
      
      argmax_i, inputs_distr_i = [], []
      for t in range(len(inputs)):
        if t != count:
          argmax_i.append(argmax_inputs[t].argmax(dim=1))
          inputs_distr_i.append(inputs_distr[t])
      
      for b, c, indices in zip(argmax_i[0], argmax_i[1], range(argmax_i[0].shape[0])):
        freq_b = torch.where(argmax_i[0]==b, 1, 0)
        freq_c = torch.where(argmax_i[1]==c, 1, 0)
        freq = (freq_b * freq_c).sum()
        prob_b = inputs_distr_i[0][:,b].unsqueeze(1)
        prob_c = inputs_distr_i[1][:,c].unsqueeze(1)
        jacobian_i[:,:,i] += y[indices].unsqueeze(0).repeat(batch_size, 1)*prob_b*prob_c/(n*freq)
    jacobian.append(jacobian_i)
  return tuple(jacobian)

class BlackBoxSum2(torch.autograd.Function):
  def sum_2(xa, xb):
    y_dim = xa.shape[1] + xb.shape[1]
    y_pred = torch.argmax(xa, dim=1) + torch.argmax(xb, dim=1)
    return F.one_hot(y_pred, num_classes = y_dim).float()

  @staticmethod
  def forward(ctx, *inputs):
      ctx.save_for_backward(*inputs)
      output = BlackBoxSum2.sum_2(*inputs)
      return output

  @staticmethod
  def backward(ctx, grad_output):
      inputs = ctx.saved_tensors
      js = finite_difference_2(BlackBoxSum2.sum_2, *inputs)
      js = [grad_output.unsqueeze(1).matmul(j).squeeze(1) for j in js]
      return tuple(js)


class BlackBoxSum3(torch.autograd.Function):

  def sum_3(xa, xb, xc):
    y_dim = xa.shape[1] + xb.shape[1] + xc.shape[1]
    y_pred = torch.argmax(xa, dim=1) + torch.argmax(xb, dim=1) + torch.argmax(xc, dim=1)
    return F.one_hot(y_pred, num_classes = y_dim).float()

  @staticmethod
  def forward(ctx, *inputs):
      ctx.save_for_backward(*inputs)
      output = BlackBoxSum3.sum_3(*inputs)
      return output

  @staticmethod
  def backward(ctx, grad_output):
      inputs = ctx.saved_tensors
      js = finite_difference_3(BlackBoxSum3.sum_3, *inputs)
      js = [grad_output.unsqueeze(1).matmul(j).squeeze(1) for j in js]
      return tuple(js)