import torch
import torch.nn.functional as F

def finite_difference(fn, output, *inputs):
  # Prepare the inputs to the black-box function
  argmax_inputs = []
  for input_i in inputs: 
    argmax_inputs.append(F.one_hot(input_i.argmax(dim=1), num_classes = input_i.shape[1]))

  # Compute the probability and the frequency of the inputs
  batch_size, num_inputs = argmax_inputs[0].shape[0], len(argmax_inputs)
  probs, freqs = torch.zeros(num_inputs,batch_size,batch_size), torch.zeros(num_inputs,batch_size,batch_size)
  for i in range(num_inputs):
    for j in range(batch_size):
      argmax_inputs_j = argmax_inputs[i][j].argmax()
      probs[i][j] = inputs[i][:,argmax_inputs_j]
      freqs[i][j] = torch.where(argmax_inputs[i].argmax(dim=1)==argmax_inputs_j,1,0)

  # Compute the jacobian for each input
  jacobian = []
  for input_i, n in zip(inputs, range(num_inputs)):
    input_dim = input_i.shape[1]
    jacobian_i = torch.zeros(batch_size, output.shape[1], input_dim)
    
    # Remove the information about the nth input
    probs_n = torch.cat((probs[:n],probs[n+1:]))
    freqs_n = torch.cat((freqs[:n],freqs[n+1:]))

    for i in torch.arange(0,input_dim):
      # Perturb the nth input to be i
      inputs_i = argmax_inputs.copy()
      inputs_i[n] = F.one_hot(i, num_classes=input_dim).float().repeat(batch_size,1) 
      
      # Compute the corresponding change in the output
      y = fn(*tuple(inputs_i)) - output  
      
      # Update the jacobian, weighted by the probability of the unchanged inputs
      for batch_i in range(batch_size):
        probs_i = probs_n[:,batch_i,:].prod(dim=0)
        freqs_i = freqs_n[:,batch_i,:].prod(dim=0).sum()
        jacobian_i[:,:,i] += y[batch_i].unsqueeze(0).repeat(batch_size,1)*(probs_i.unsqueeze(1))/(input_dim*freqs_i)
    
    jacobian.append(jacobian_i)
  
  return tuple(jacobian)

class BlackBoxFunction(torch.autograd.Function):
  # Placeholder function; to be initialized before calling
  fn = lambda x : x

  @staticmethod
  def forward(ctx, *inputs):
      output = BlackBoxFunction.fn(*inputs)

      # To use during backward propagation
      ctx.save_for_backward(*inputs, output)
      return output

  @staticmethod
  def backward(ctx, grad_output):
      inputs, output = ctx.saved_tensors[:-1], ctx.saved_tensors[-1]
      js = finite_difference(BlackBoxFunction.fn, output, *inputs)

      # L1 normalization
      js = [F.normalize(grad_output.unsqueeze(1).matmul(j).squeeze(1), dim=1, p=1) for j in js]
      return tuple(js)