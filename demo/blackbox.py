import torch
import torch.nn.functional as F

def compute_prob(argmax, inputs):
  n = len(argmax)
  batch_size = argmax[0].shape[0]
  probs, freqs = torch.zeros(n,batch_size,batch_size,1), torch.zeros(n,batch_size,batch_size)
  for i in range(n):
    freq = torch.zeros(batch_size,batch_size)
    prob = torch.ones(batch_size,batch_size,1)
    for j in range(batch_size):
      input_i = argmax[i][j]
      prob[j] = inputs[i][:,input_i].unsqueeze(1)
      freq[j] = torch.where(argmax[i]==input_i,1,0)
    probs[i] = prob
    freqs[i] = freq
  return probs, freqs

def finite_difference(fn, *inputs):
  argmax_inputs, inputs_distr, argmax_x = [], [], []
  k = 0
  for x in inputs:
    n = x.shape[1]
    k += n
    argmax_x.append(x.argmax(dim=1))
    argmax_inputs.append(F.one_hot(x.argmax(dim=1), num_classes = n))
    inputs_distr.append(x)

  y_pred = fn(*tuple(argmax_inputs))
  probs, freqs = compute_prob(argmax_x, inputs_distr)

  jacobian = []
  for x, count in zip(inputs, range(len(inputs))):
    batch_size, n = x.shape
    jacobian_i = torch.zeros(batch_size, k, n)
    probs_c = torch.cat((probs[:count],probs[count+1:]))
    freqs_c = torch.cat((freqs[:count],freqs[count+1:]))
    for i in torch.arange(0,n):
      inputs_i = argmax_inputs.copy()
      inputs_i[count] = F.one_hot(i, num_classes=n).float().repeat(batch_size,1) 
      y = fn(*tuple(inputs_i)) - y_pred   
      for batch_i in range(batch_size):
        probs_i = probs_c[:,batch_i,:].prod(dim=0)
        freqs_i = freqs_c[:,batch_i,:].prod(dim=0).sum()
        jacobian_i[:,:,i] += y[batch_i].unsqueeze(0).repeat(batch_size, 1)*probs_i/(n*freqs_i)
    jacobian.append(jacobian_i)
  return tuple(jacobian)

class BlackBoxFunction(torch.autograd.Function):
  fn = lambda x : x

  @staticmethod
  def forward(ctx, *inputs):
      ctx.save_for_backward(*inputs)
      output = BlackBoxFunction.fn(*inputs)
      return output

  @staticmethod
  def backward(ctx, grad_output):
      inputs = ctx.saved_tensors
      js = finite_difference(BlackBoxFunction.fn, *inputs)
      js = [grad_output.unsqueeze(1).matmul(j).squeeze(1) for j in js]
      return tuple(js)