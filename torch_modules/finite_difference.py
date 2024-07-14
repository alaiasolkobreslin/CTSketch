import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *

from src import constants

class FiniteDifference(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.function = kwargs['bbox']
        self.input_mappings = kwargs['input_mappings']
        self.output_mapping = kwargs['output_mapping']

    def forward(self, *inputs):
        output = BlackBoxFunction.apply(self.function, self.input_mappings, self.output_mapping, *inputs)
        return output

class BlackBoxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fn, input_mappings, output_mapping, *inputs):
        output = BlackBoxFunction.decorated_fn(fn, input_mappings, output_mapping, *inputs)

        # To use during backward propagation
        ctx.save_for_backward(*inputs, output)
        ctx.bbox_fn = fn
        ctx.input_mappings = input_mappings
        ctx.output_mapping = output_mapping
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inputs, output = ctx.saved_tensors[:-1], ctx.saved_tensors[-1]
        bbox_fn = ctx.bbox_fn
        input_mappings = ctx.input_mappings
        output_mapping = ctx.output_mapping
        js = BlackBoxFunction.finite_difference(bbox_fn, input_mappings, output_mapping, output, *inputs)

        jacobians = []
        # normalization
        for i, j in enumerate(js): 
            gumbel = torch.autograd.functional.jacobian(F.gumbel_softmax, inputs[i])
            t = F.normalize(grad_output.unsqueeze(1).matmul(j).squeeze(1), dim=1, p=1)
            jacobians.append(torch.einsum('ab,abcd->cd', t, gumbel))
            
        return None, None, None, *tuple(jacobians) 
    
    def decorated_fn(fn, input_mappings, output_mapping, *inputs):
        def zip_batched_inputs(batched_inputs):
            result = [[lists] for lists in zip(*batched_inputs)]
            return result
            
        def invoke_function_on_inputs(inputs):
            for r in inputs:
                try:
                    y = fn(*r)
                    yield y
                except:
                    yield constants.RESERVED_FAILURE

        def invoke_function_on_batched_inputs(batched_inputs):
            return [list(invoke_function_on_inputs(batch)) for batch in batched_inputs]
            
        # Prepare the inputs to the black-box function  
        to_compute_inputs = []
        for (input_i, input_mappings_i) in zip(inputs, input_mappings):
            sampled_element_i = [input_mappings_i.get_elements(input_i_example) for input_i_example in input_i]
            to_compute_inputs.append(sampled_element_i)
        to_compute_inputs = zip_batched_inputs(to_compute_inputs)

        # Get the outputs from the black-box function
        results = invoke_function_on_batched_inputs(to_compute_inputs)
        output = output_mapping.vectorize(results)

        return F.softmax(output, dim=1)
    
    def finite_difference(fn, input_mappings, output_mapping, output, *inputs):
        k = 10
        batch_size = inputs[0].shape[0]

        # Compute the jacobian for each input
        jacobian = []
        for n, (input_i, mapping_i) in enumerate(zip(inputs, input_mappings)):
            input_dim = input_i.shape[1]
            jacobian_i = torch.zeros(batch_size, output.shape[1], input_dim)
            
            for i in mapping_i.get_perturbations(k, batch_size):
                # Perturb the nth input to be i
                inputs_i = list(inputs).copy()
                inputs_i[n] = i
                
                # Compute the corresponding change in the output
                y = BlackBoxFunction.decorated_fn(fn, input_mappings, output_mapping, *tuple(inputs_i)) - output  
                i = i[0].argmax()

                # Update the jacobian, weighted by the probability of the unchanged inputs
                p = (input_i - input_i[:,i].unsqueeze(-1))
                N = torch.distributions.Normal(0,2)
                p = N.cdf(-p).prod(dim=1).unsqueeze(-1).unsqueeze(-1)
                t = y.unsqueeze(1).repeat(1, batch_size, 1)*p
                
                jacobian_i[:,:,i] += t.sum(dim=0)
                c3 = input_i.argmax(dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1,batch_size,output.shape[1],input_dim)
                v3 = torch.arange(input_dim).unsqueeze(0).unsqueeze(1).unsqueeze(2)
                jacobian_i -= torch.where(c3==v3, t.unsqueeze(-1).repeat(1,1,1,input_dim), 0).sum(dim=0)
                
            jacobian.append(jacobian_i)
        
        return tuple(jacobian)


class InputMapping:
  def __init__(self): pass

  def get_elements(self): pass

  def get_perturbations(self): pass

class DiscreteInputMapping(InputMapping):
  def __init__(self, elements: List[Any]):
    self.elements = elements
  
  def get_elements(self, probs):
    idx = probs.argmax()
    return self.elements[idx]
  
  def get_perturbations(self, k, batch_size):
    x = torch.multinomial(torch.ones(len(self.elements)), k).sort().values
    return F.one_hot(x, num_classes=len(self.elements)).unsqueeze(1).repeat(1,batch_size,1)

class OutputMapping:
  def __init__(self): pass

  def vectorize(self): pass

class DiscreteOutputMapping(OutputMapping):
  def __init__(self, elements: List[Any]):
    self.elements = elements
    self.element_indices = {e: i for (i, e) in enumerate(elements)}
  
  def vectorize(self, results: List) -> torch.Tensor:
    batch_size = len(results)
    result_tensor = torch.zeros((batch_size, len(self.elements)))
    for i in range(batch_size):
      if results[i][0] != constants.RESERVED_FAILURE:
        result_tensor[i][self.element_indices[results[i][0]]] += 1
    return result_tensor