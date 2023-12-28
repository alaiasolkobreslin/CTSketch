import torch
import torch.nn.functional as F
from typing import *

RESERVED_FAILURE = "__RESERVED_FAILURE__"

class ListInput:
    """
    The struct holding vectorized list input
    """
    def __init__(self, tensor: torch.Tensor, lengths: List[int]):
        self.tensor = tensor
        self.lengths = lengths

class InputMapping:
  def __init__(self): pass

  def get_elements(self): pass

class ListInputMapping(InputMapping):
  def __init__(self, max_length: int, element_input_mapping: InputMapping):
    self.max_length = max_length
    self.element_input_mapping = element_input_mapping  

class DiscreteInputMapping(InputMapping):
  def __init__(self, elements: List[Any]):
    self.elements = elements
  
  def get_elements(self, idx):
    return self.elements[idx]

class OutputMapping:
  def __init__(self): pass

  def vectorize(self, results: List):
    """
    An output mapping should implement this function to vectorize the results
    """
    pass

class DiscreteOutputMapping(OutputMapping):
  def __init__(self, elements: List[Any]):
    self.elements = elements
    self.element_indices = {e: i for (i, e) in enumerate(elements)}
  
  def vectorize(self, results: List) -> torch.Tensor:
    batch_size = len(results)
    result_tensor = torch.zeros((batch_size, len(self.elements)), requires_grad=True)
    for i in range(batch_size):
      if results[i][0] != RESERVED_FAILURE:
        result_tensor[i][self.element_indices[results[i][0]]].data += 1
    return result_tensor

class UnknownDiscreteOutputMapping(OutputMapping):
    def __init__(self):
        pass

def finite_difference(fn, output, *inputs):
  # Prepare the inputs to the black-box function
  argmax_inputs = []
  for input_i in inputs: 
    ys = torch.multinomial(torch.ones(input_i.shape[1]), 1)
    argmax_inputs.append(F.one_hot(input_i.topk(dim=1,k=input_i.shape[1]).indices[:,ys], num_classes=input_i.shape[1]).squeeze(1))

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

    for i in torch.multinomial(torch.ones(input_dim),input_dim).sort().values:
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
  
class BlackBox(torch.nn.Module):
  def __init__(
    self,
    function: Callable,
    input_mappings: Tuple[InputMapping],
    output_mapping: OutputMapping):
    super(BlackBox, self).__init__()
    assert type(input_mappings) == tuple, "input_mappings must be a tuple"
    self.function = function
    self.input_mappings = input_mappings
    self.output_mapping = output_mapping
  
  def forward(self, *inputs):
    num_inputs = len(inputs)
    assert num_inputs == len(self.input_mappings), "inputs and input_mappings must have the same length"

    # Get the batch size
    batch_size = self.get_batch_size(inputs[0])
    for i in range(1, num_inputs):
      assert batch_size == self.get_batch_size(inputs[i]), "all inputs must have the same batch size"

    BlackBoxFunction.fn = self.dec_fn

    return BlackBoxFunction.apply(*inputs)

  def get_batch_size(self, input: Any):
    if type(input) == torch.Tensor:
      return input.shape[0]
    elif type(input) == ListInput:
      return len(input.lengths)
    raise Exception("Unknown input type")

  def zip_batched_inputs(self, batched_inputs):
    result = [list(zip(*lists)) for lists in zip(*batched_inputs)]
    return result
  
  def invoke_function_on_inputs(self, inputs):
    """
      Given a list of inputs, invoke the black-box function on each of them.
      Note that function may fail on some inputs, and we skip those.
    """
    for r in inputs:
      try:
        y = self.function(*r)
        yield y
      except:
        yield RESERVED_FAILURE

  def invoke_function_on_batched_inputs(self, batched_inputs):
      return [list(self.invoke_function_on_inputs(batch)) for batch in batched_inputs]

  def dec_fn(self, *inputs):
    # Prepare the inputs to the black-box functionBlackBoxFunction.fn = self.function    
    to_compute_inputs = []
    for (input_i, input_mappings_i) in zip(inputs, self.input_mappings):
      argmax_index_i = input_i.argmax(dim=1).unsqueeze(-1)
      sampled_element_i = [[input_mappings_i.get_elements(i) for i in argmax_indices_for_task_i] for argmax_indices_for_task_i in argmax_index_i]
      to_compute_inputs.append(sampled_element_i)
    to_compute_inputs = self.zip_batched_inputs(to_compute_inputs)

    # Get the outputs from the black-box function
    results = self.invoke_function_on_batched_inputs(to_compute_inputs)
    output = self.output_mapping.vectorize(results)

    return output