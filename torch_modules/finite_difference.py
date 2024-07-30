import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import *

RESERVED_FAILURE = "__RESERVED_FAILURE__"

class FiniteDifference(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.function = kwargs['bbox']
        self.input_mappings = kwargs['input_mappings']
        self.output_mapping = kwargs['output_mapping']
        self.y_elements = kwargs['elements'] if 'elements' in kwargs.keys() else []

    def forward(self, *inputs):
        output = BlackBoxFunction.apply(self.function, self.input_mappings, self.output_mapping, self.y_elements, *inputs)
        return BlackBoxFunction.elements, output

class BlackBoxFunction(torch.autograd.Function):
    elements = []
    
    @staticmethod
    def forward(ctx, fn, input_mappings, output_mapping, elements, *inputs):
        y_elements, output = BlackBoxFunction.decorated_fn(fn, input_mappings, output_mapping, elements, *inputs)
        output, perts, ys, y_elements = BlackBoxFunction.get_perturbations(fn, input_mappings, output_mapping, output, y_elements, *inputs)

        # To use during backward propagation
        ctx.save_for_backward(*inputs, output)
        ctx.input_mappings = input_mappings
        ctx.output_mapping = output_mapping
        ctx.perts = perts
        ctx.ys = ys
        
        BlackBoxFunction.elements = y_elements
        return output

    @staticmethod
    def backward(ctx, grad_output):
        inputs, output = ctx.saved_tensors[:-1], ctx.saved_tensors[-1]
        input_mappings = ctx.input_mappings
        output_mapping = ctx.output_mapping
        perts = ctx.perts
        ys = ctx.ys
        
        js = BlackBoxFunction.finite_difference(input_mappings, output_mapping, output, perts, ys, *inputs)

        jacobians = []
        # normalization
        for i, j in enumerate(js): 
            gumbel = torch.autograd.functional.jacobian(F.gumbel_softmax, inputs[i])
            t = F.normalize(grad_output.unsqueeze(1).matmul(j).squeeze(1), dim=1, p=1)
            jacobians.append(torch.einsum('ab,abcd->cd', t, gumbel))
            
        return None, None, None, None, *tuple(jacobians) 
    
    def decorated_fn(fn, input_mappings, output_mapping, elements, *inputs):
        def zip_batched_inputs(batched_inputs):
            result = [[lists] for lists in zip(*batched_inputs)]
            return result
            
        def invoke_function_on_inputs(inputs):
            for r in inputs:
                try:
                    y = fn(*r)
                    yield y
                except:
                    yield RESERVED_FAILURE

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
        elements, output = output_mapping.vectorize(results, elements)

        return elements, output
    
    def get_perturbations(fn, input_mappings, output_mapping, output, y_elements, *inputs):
      perts = []
      ys = []

      for n, (input_i, mapping_i) in enumerate(zip(inputs, input_mappings)):
        k = min(len(mapping_i.elements), 100)
        pert = mapping_i.get_perturbations(k, input_i)
        perts.append(pert)
        
        ys_i = []
        for i in pert:
          # Perturb the nth input to be i
          inputs_i = list(inputs).copy()
          inputs_i[n] = i

          y_elements, y = BlackBoxFunction.decorated_fn(fn, input_mappings, output_mapping, y_elements, *tuple(inputs_i)) 
          ys_i.append(y)
        
        ys.append(ys_i)
          
      if type(output_mapping) == UnknownDiscreteOutputMapping:
        output = F.one_hot(output.argmax(dim=1), num_classes = len(y_elements)).float() 
        output = F.softmax(output, dim=-1)
      elif type(output_mapping) == DiscreteOutputMapping:
        output = F.softmax(output, dim=-1)
      return output, perts, ys, y_elements  
    
    def finite_difference(input_mappings, output_mapping, output, perts, ys, *inputs):
        batch_size = inputs[0].shape[0]

        # Compute the jacobian for each input
        jacobian = []
        for input_i, pert_i, y_i, mapping_i in zip(inputs, perts, ys, input_mappings):
          input_dim = input_i.shape[1]
          jacobian_i = torch.zeros(batch_size, output.shape[1], input_dim)
            
          for i, y in zip(pert_i, y_i):                
            # Compute the corresponding change in the output
            if type(output_mapping) == UnknownDiscreteOutputMapping:
              y = F.pad(y, (0, output.shape[1]-y.shape[1], 0, 0)) 
              y = F.softmax(y, dim=-1)
            elif type(output_mapping) == DiscreteOutputMapping:
              y = F.softmax(y, dim=-1)
            y = y - output  

            # Update the jacobian, weighted by the probability of the unchanged inputs
            if type(mapping_i) == DiscreteInputMapping:
              i = i[0].argmax()
              t = y.unsqueeze(1).repeat(1, batch_size, 1)*(input_i[:,i].unsqueeze(-1).unsqueeze(-1))
              jacobian_i[:,:,i] += t.sum(dim=0)

              c3 = input_i.argmax(dim=1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1,batch_size,output.shape[1],input_dim)
              v3 = torch.arange(input_dim).unsqueeze(0).unsqueeze(1).unsqueeze(2)
              jacobian_i -= torch.where(c3==v3, t.unsqueeze(-1).repeat(1,1,1,input_dim), 0).sum(dim=0)
                
            elif type(mapping_i) == ListInputMapping:
              element_len = len(mapping_i.return_elements()) 
              max_det = inputs[0].shape[1] // element_len
                
              for j in range(max_det):
                i_j = i[:,j*element_len:(j+1)*element_len]
                input_i_j = input_i[:,j*element_len:(j+1)*element_len]
                for b in range(batch_size):
                  if torch.all(i_j[b] == input_i_j[b]):
                    continue

                  i2 = i_j[b].argmax()     
                  t = y[b].unsqueeze(0)*(input_i_j[:i2].unsqueeze(-1).unsqueeze(-1))
                  jacobian_i[:,:,j*element_len+i2] += t

                  c3 = input_i_j.argmax(dim=1).unsqueeze(-1).unsqueeze(-1)
                  v3 = torch.arange(element_len).unsqueeze(0).unsqueeze(1)
                  jacobian_i[:,:,j*element_len:(j+1)*element_len] -= torch.where(c3==v3, t.unsqueeze(-1), 0)

          jacobian.append(jacobian_i)
        
        return tuple(jacobian)


class InputMapping:
  def __init__(self): pass

  def get_elements(self): pass

  def get_perturbations(self): pass

  def return_elements(self):
     return self.elements

class DiscreteInputMapping(InputMapping):
  def __init__(self, elements: List[Any]):
    self.elements = elements
  
  def get_elements(self, probs):
    if len(probs.shape)==1:
      return self.elements[probs.argmax()]
    
    elements = []
    for idx in probs.argmax(dim=-1):
      elements.append(self.elements[idx])
    return elements
  
  def get_perturbations(self, k, probs):
    max_probs = probs.max(dim=1).values.unsqueeze(-1)
    probs = torch.where(max_probs == probs, 0, probs + max_probs/(probs.shape[1]-1))
    x = torch.multinomial(probs, k).sort().values
    return F.one_hot(x, num_classes=len(self.elements)).permute(1,0,2)

class ListInputMapping(InputMapping):
  def __init__(self, lengths, max_len, element_input_mapping: InputMapping):
    self.element_input_mapping = element_input_mapping  
    self.lengths = lengths
    self.max_len = max_len
    self.elements = self.element_input_mapping.return_elements()
  
  def get_elements(self, probs):
    elements = []
    len_elements = probs.shape[1]//self.max_len
    for l in range(len(self.lengths)):
      t = []
      for i in range(self.lengths[l]):
        t.append(self.element_input_mapping.get_elements(probs[l][i*len_elements:(i+1)*len_elements]))
      elements.append(t)
    return elements
  
  def get_perturbations(self, k, probs):
    batch_size = probs.shape[0]
    ps = torch.zeros(k, batch_size, self.max_len, len(self.elements))
    for l in range(len(self.lengths)):
      for i in range(batch_size):
        if self.lengths[i] > l:
          p = probs[i,l*len(self.elements):(l+1)*len(self.elements)].unsqueeze(0)
          if p.sum() == 0: continue
          ps[:,i,l] += self.element_input_mapping.get_perturbations(k, p).squeeze(1)
    return ps.flatten(2)

class OutputMapping:
  def __init__(self): pass

  def vectorize(self): pass

class DiscreteOutputMapping(OutputMapping):
  def __init__(self, elements: List[Any]):
    self.elements = elements
    self.element_indices = {e: i for (i, e) in enumerate(elements)}
  
  def vectorize(self, results: List, elements) -> torch.Tensor:
    batch_size = len(results)
    result_tensor = torch.zeros((batch_size, len(self.elements)))
    for i in range(batch_size):
      if results[i][0] != RESERVED_FAILURE:
        result_tensor[i][self.element_indices[results[i][0]]] += 1
    return self.elements, result_tensor
  
class UnknownDiscreteOutputMapping(OutputMapping):
  def __init__(self):
    pass
    
  def vectorize(self, results:List, y_elements) -> torch.Tensor:
    batch_size = len(results)

    # Get the unique elements
    elements = y_elements + list(set([elem for batch in results for elem in batch if (elem != RESERVED_FAILURE and elem not in y_elements)]))
    element_indices = {e: i for (i, e) in enumerate(elements)}

    result_tensor = torch.zeros((batch_size, len(elements)))
    for i in range(batch_size):
      if results[i][0] != RESERVED_FAILURE:
        result_tensor[i][element_indices[results[i][0]]] += 1
    return (elements, result_tensor) 

class BinaryOutputMapping(OutputMapping):
  def __init__(self):
    pass

  def vectorize(self, results:List, y_elements) -> torch.Tensor:
    batch_size = len(results)
    result_tensor = torch.zeros((batch_size, 1))
    for i, result_i in enumerate(results):
      if result_i[0] == True:
        result_tensor[i] += 1
    return [1], result_tensor